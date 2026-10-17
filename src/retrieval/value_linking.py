"""
Value linking (exact match) for GraphRAG.

Questions often name a cell VALUE instead of a column ("countries in Asia" ->
country.Continent, "played in 2013" -> matches.year), which column-name
embeddings cannot catch. This module matches the question against the distinct
values stored in the database and returns the matching column node ids, which
retrieval.py adds as extra anchors after Stage 2 (before graph traversal).

Cost: the index is built ONCE per database (one SELECT DISTINCT ... LIMIT N per
eligible column, read-only) and cached in-process; per question it is one dict
lookup per n-gram / number. No embeddings, no GPU.

Text values (always on with use_value_linking):
  - column_type == "text", not a primary key, not part of a foreign-key edge,
    name not ending in "id"/"code" (codes like "ABW" never appear in questions)
  - content checks on the sampled values: mean length <= value_max_chars
    (drops description columns), numeric ratio <= value_max_numeric_ratio
  - normalized exactly like the question in retrieval._extract_phrases()
    (lowercase, punctuation removed, whitespace collapsed)

Number values (only with value_link_numbers):
  - "number" columns and text columns that are mostly numeric (same PK/FK/id
    exclusions); "time" columns contribute the 4-digit years in their values
  - canonical form so 2013, 2013.0 and '2013' are equal
  - a number found in more than value_max_columns_per_number columns is dropped:
    small numbers (3, 10) sit in many columns and say nothing, a year like 2013
    usually sits in one
  - question numbers are read from the RAW question (the n-gram cleaner turns
    "3.5" into "35"); numbers right after value_number_skip_after words
    ("top 3", "first 5") are LIMITs, not values, and are ignored
"""

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

from src.core.config import PipelineConfig
from src.core.schema import normalize_identifier

logger = logging.getLogger(__name__)

_NUMERIC = re.compile(r"[\d\s]+")
_YEAR = re.compile(r"\b(1[5-9]\d\d|20\d\d)\b")
_QUESTION_NUMBER = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?![\w])")


def normalize_value(value) -> str:
    """Same normalization as retrieval._extract_phrases() applies to the question."""
    clean = re.sub(r"[^\w\s]", "", str(value).lower())
    return " ".join(clean.split())


def canonical_number(value) -> str | None:
    """'2013', 2013.0, ' 2013 ' -> '2013'; '3.50' -> '3.5'; non-numbers -> None."""
    try:
        f = float(str(value).replace(",", "").strip())
    except ValueError:
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return str(int(f)) if f.is_integer() else f"{f:.10g}"


@dataclass
class ValueIndex:
    db_name: str
    value_to_nodes: dict[str, list[str]] = field(default_factory=dict)
    number_to_nodes: dict[str, list[str]] = field(default_factory=dict)
    indexed_columns: list[str] = field(default_factory=list)         # text value columns
    indexed_number_columns: list[str] = field(default_factory=list)  # number/time columns
    skipped_columns: dict[str, str] = field(default_factory=dict)    # node id -> reason


def _fk_nodes(graph: nx.Graph, db_name: str) -> set[str]:
    out = set()
    for u, v, d in graph.edges(data=True):
        if d.get("relation") == "foreign_key" and graph.nodes[u].get("database") == db_name:
            out.update((u, v))
    return out


def _key_column_reason(node_data: dict, is_fk: bool) -> str | None:
    if node_data.get("is_pk"):
        return "primary key"
    if is_fk:
        return "foreign key"
    last = normalize_identifier(node_data["column"]).split()[-1:]
    if last and last[0] in ("id", "code"):
        return "id/code name"
    return None


def build_value_index(graph: nx.Graph, db_name: str, db_path: Path, cfg: PipelineConfig) -> ValueIndex:
    """Read distinct values of every eligible column of `db_name` once (read-only)."""
    idx = ValueIndex(db_name)
    if not db_path.exists():
        logger.warning("Value linking: %s not found, no values indexed for %s", db_path, db_name)
        return idx

    fk = _fk_nodes(graph, db_name)
    nodes = [(n, d) for n, d in graph.nodes(data=True)
             if d.get("database") == db_name and d.get("type") == "column"]
    want_numbers = cfg.value_link_numbers
    stop = set(cfg.query_stopwords)

    conn = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    # Spider has a few badly encoded rows (wta_1, IMPLEMENTATION_DECISIONS.md poin 16)
    conn.text_factory = lambda b: b.decode("utf-8", errors="ignore")
    value_cols: dict[str, set[str]] = {}
    number_cols: dict[str, set[str]] = {}

    def add_numbers(node: str, values) -> None:
        found = {n for v in values if (n := canonical_number(v)) is not None}
        if not found:
            idx.skipped_columns[node] = "no parseable numbers"
            return
        idx.indexed_number_columns.append(node)
        for n in found:
            number_cols.setdefault(n, set()).add(node)

    try:
        for node, d in nodes:
            ctype = d.get("column_type")
            reason = _key_column_reason(d, node in fk)
            if reason is None and ctype not in ("text", "number", "time"):
                reason = f"type={ctype}"
            if reason is None and ctype in ("number", "time") and not want_numbers:
                reason = f"type={ctype}"
            if reason:
                idx.skipped_columns[node] = reason
                continue
            try:
                rows = conn.execute(
                    f'SELECT DISTINCT "{d["column"]}" FROM "{d["table"]}" '
                    f'WHERE "{d["column"]}" IS NOT NULL LIMIT ?',
                    (cfg.value_max_rows_per_column,),
                ).fetchall()
            except sqlite3.Error as e:
                idx.skipped_columns[node] = f"sqlite error: {e}"
                continue

            raw = [str(r[0]) for r in rows if str(r[0]).strip()]
            if not raw:
                idx.skipped_columns[node] = "empty"
                continue

            if ctype == "number":
                add_numbers(node, raw)
                continue
            if ctype == "time":
                add_numbers(node, {y for v in raw for y in _YEAR.findall(v)})
                continue

            # text column
            if sum(len(v) for v in raw) / len(raw) > cfg.value_max_chars:
                idx.skipped_columns[node] = "long text"
                continue
            values = [normalize_value(v) for v in raw]
            if sum(1 for v in values if _NUMERIC.fullmatch(v)) / len(values) > cfg.value_max_numeric_ratio:
                if want_numbers:
                    add_numbers(node, raw)
                else:
                    idx.skipped_columns[node] = "mostly numeric"
                continue

            idx.indexed_columns.append(node)
            for v in values:
                if (cfg.value_min_chars <= len(v) <= cfg.value_max_chars
                        and not _NUMERIC.fullmatch(v)
                        and len(v.split()) <= cfg.max_ngram   # longer values can never match an n-gram
                        and v not in stop):
                    value_cols.setdefault(v, set()).add(node)
    finally:
        conn.close()

    # Values found in too many columns ("yes", "other", the number 3) are not evidence of anything.
    idx.value_to_nodes = {
        v: sorted(cols) for v, cols in value_cols.items()
        if len(cols) <= cfg.value_max_columns_per_value
    }
    idx.number_to_nodes = {
        n: sorted(cols) for n, cols in number_cols.items()
        if len(cols) <= cfg.value_max_columns_per_number
    }
    return idx


def extract_question_numbers(question: str, cfg: PipelineConfig) -> list[str]:
    """Canonical numbers in the raw question, minus LIMIT-like ones ("top 3")."""
    skip = {w.lower() for w in cfg.value_number_skip_after}
    out: dict[str, None] = {}
    for m in _QUESTION_NUMBER.finditer(question):
        before = re.findall(r"[A-Za-z]+", question[: m.start()])
        if before and before[-1].lower() in skip:
            continue
        n = canonical_number(m.group(1))
        if n is not None:
            out.setdefault(n)
    return list(out)


def link_values(phrases: list[str], index: ValueIndex, cfg: PipelineConfig,
                numbers: list[str] = ()) -> list[str]:
    """
    Column node ids whose values exactly equal a question n-gram (longer n-grams
    first, more specific) or a question number. At most
    cfg.value_max_linked_columns columns, in a deterministic order.
    """
    out: dict[str, None] = {}
    hits = [index.value_to_nodes.get(p, ()) for p in sorted(phrases, key=lambda p: (-len(p.split()), p))]
    hits += [index.number_to_nodes.get(n, ()) for n in numbers]
    for nodes in hits:
        for node in nodes:
            out.setdefault(node)
            if len(out) >= cfg.value_max_linked_columns:
                return list(out)
    return list(out)


# In-process cache: build once per (database, settings that change the index).
_CACHE: dict[tuple, ValueIndex] = {}


def get_value_index(graph: nx.Graph, db_name: str, cfg: PipelineConfig) -> ValueIndex:
    key = (str(cfg.db_dir), db_name, cfg.value_max_rows_per_column, cfg.value_min_chars,
           cfg.value_max_chars, cfg.value_max_numeric_ratio, cfg.value_max_columns_per_value,
           cfg.value_link_numbers, cfg.value_max_columns_per_number,
           cfg.max_ngram, tuple(cfg.query_stopwords))
    if key not in _CACHE:
        db_path = cfg.db_dir / db_name / f"{db_name}.sqlite"
        _CACHE[key] = build_value_index(graph, db_name, db_path, cfg)
    return _CACHE[key]
