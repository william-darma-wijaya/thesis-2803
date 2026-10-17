"""
Value linking (exact match) for GraphRAG.

Questions often name a cell VALUE instead of a column ("countries in Asia" ->
country.Continent), which column-name embeddings cannot catch. This module
matches question n-grams against the distinct values stored in the database
and returns the matching column node ids, which retrieval.py adds as extra
anchors after Stage 2 (before graph traversal).

Cost: the index is built ONCE per database (one SELECT DISTINCT ... LIMIT N per
eligible column, read-only) and cached in-process; per question it is one dict
lookup per n-gram. No embeddings, no GPU.

Eligible columns (decided automatically, see build_value_index()):
  - column_type == "text" in tables.json (number/time/boolean/others skipped:
    numbers match far too many columns, dates rarely match the question form)
  - not a primary key, not part of a foreign-key edge, name not ending in
    "id"/"code" (codes like "ABW" never appear in questions)
  - content checks on the sampled values: mean length <= value_max_chars
    (drops description columns), numeric ratio <= value_max_numeric_ratio
Values are normalized exactly like the question in retrieval._extract_phrases()
(lowercase, punctuation removed, whitespace collapsed) so equal strings meet.
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


def normalize_value(value) -> str:
    """Same normalization as retrieval._extract_phrases() applies to the question."""
    clean = re.sub(r"[^\w\s]", "", str(value).lower())
    return " ".join(clean.split())


@dataclass
class ValueIndex:
    db_name: str
    value_to_nodes: dict[str, list[str]] = field(default_factory=dict)
    indexed_columns: list[str] = field(default_factory=list)   # node ids
    skipped_columns: dict[str, str] = field(default_factory=dict)  # node id -> reason


def _fk_nodes(graph: nx.Graph, db_name: str) -> set[str]:
    out = set()
    for u, v, d in graph.edges(data=True):
        if d.get("relation") == "foreign_key" and graph.nodes[u].get("database") == db_name:
            out.update((u, v))
    return out


def _skip_reason_by_metadata(node_data: dict, is_fk: bool) -> str | None:
    if node_data.get("column_type") != "text":
        return f"type={node_data.get('column_type')}"
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

    conn = sqlite3.connect(Path(db_path).resolve().as_uri() + "?mode=ro", uri=True)
    # Spider has a few badly encoded rows (wta_1, IMPLEMENTATION_DECISIONS.md poin 16)
    conn.text_factory = lambda b: b.decode("utf-8", errors="ignore")
    value_cols: dict[str, set[str]] = {}
    try:
        for node, d in nodes:
            reason = _skip_reason_by_metadata(d, node in fk)
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
            if sum(len(v) for v in raw) / len(raw) > cfg.value_max_chars:
                idx.skipped_columns[node] = "long text"
                continue
            values = [normalize_value(v) for v in raw]
            if sum(1 for v in values if _NUMERIC.fullmatch(v)) / len(values) > cfg.value_max_numeric_ratio:
                idx.skipped_columns[node] = "mostly numeric"
                continue

            idx.indexed_columns.append(node)
            stop = set(cfg.query_stopwords)
            for v in values:
                if (cfg.value_min_chars <= len(v) <= cfg.value_max_chars
                        and not _NUMERIC.fullmatch(v)
                        and len(v.split()) <= cfg.max_ngram   # longer values can never match an n-gram
                        and v not in stop):
                    value_cols.setdefault(v, set()).add(node)
    finally:
        conn.close()

    # Values found in too many columns ("yes", "other", "male") are not evidence of anything.
    idx.value_to_nodes = {
        v: sorted(cols) for v, cols in value_cols.items()
        if len(cols) <= cfg.value_max_columns_per_value
    }
    return idx


def link_values(phrases: list[str], index: ValueIndex, cfg: PipelineConfig) -> list[str]:
    """
    Column node ids whose values exactly equal a question n-gram. Longer n-grams
    are tried first (more specific), at most cfg.value_max_linked_columns
    columns are returned, in a deterministic order.
    """
    out: dict[str, None] = {}
    for phrase in sorted(phrases, key=lambda p: (-len(p.split()), p)):
        for node in index.value_to_nodes.get(phrase, ()):
            out.setdefault(node)
            if len(out) >= cfg.value_max_linked_columns:
                return list(out)
    return list(out)


# In-process cache: build once per (database, settings that change the index).
_CACHE: dict[tuple, ValueIndex] = {}


def get_value_index(graph: nx.Graph, db_name: str, cfg: PipelineConfig) -> ValueIndex:
    key = (str(cfg.db_dir), db_name, cfg.value_max_rows_per_column, cfg.value_min_chars,
           cfg.value_max_chars, cfg.value_max_numeric_ratio, cfg.value_max_columns_per_value,
           cfg.max_ngram, tuple(cfg.query_stopwords))
    if key not in _CACHE:
        db_path = cfg.db_dir / db_name / f"{db_name}.sqlite"
        _CACHE[key] = build_value_index(graph, db_name, db_path, cfg)
    return _CACHE[key]
