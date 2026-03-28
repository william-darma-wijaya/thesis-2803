"""
GraphRAG schema linking: semantic retrieval + graph path tracing.

Two improvements over the baseline:
  1. Top-k column selection   — instead of a hard similarity threshold, keep
                                 only the k best column matches per query phrase.
  2. Two-stage retrieval      — first narrow down to the top-k most relevant
                                 *tables*, then retrieve columns only within
                                 those tables. This bounds precision regardless
                                 of database size.

Both features are controlled via PipelineConfig and degrade gracefully to the
original single-stage threshold behaviour when disabled (set to 0).
"""

import logging
import re
from dataclasses import dataclass
from typing import Optional

import networkx as nx
import torch
from sentence_transformers import SentenceTransformer, util

from config import PipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Precomputed schema index (cached per database)
# ---------------------------------------------------------------------------

@dataclass
class SchemaIndex:
    """Holds precomputed embeddings for a single database, for both stages."""
    db_name: str
    # Stage 1 — table-level
    tables: list[str]                    # unique table names
    table_embeddings: Optional[torch.Tensor]  # shape (n_tables, d)
    # Stage 2 — column-level
    columns: list[str]                   # column names, parallel to col_node_ids
    col_node_ids: list[str]              # full "db.table.col" node IDs
    col_table_map: list[str]             # which table each column belongs to
    col_embeddings: Optional[torch.Tensor]    # shape (n_cols, d)


def build_schema_index(
    graph: nx.Graph,
    db_name: str,
    embed_model: SentenceTransformer,
) -> SchemaIndex:
    """
    Precompute table-level and column-level embeddings for `db_name`.
    Call once per database before running queries against it.
    """
    col_nodes = [
        (n, d)
        for n, d in graph.nodes(data=True)
        if d.get("database") == db_name and d.get("type") == "column"
    ]

    if not col_nodes:
        return SchemaIndex(db_name, [], None, [], [], [], None)

    # --- Column index ---
    col_node_ids = [n for n, _ in col_nodes]
    columns = [d["column"] for _, d in col_nodes]
    col_table_map = [d["table"] for _, d in col_nodes]
    # Encode as "table.column" — gives the embedder full relational context.
    # "employees.name" is unambiguous; "column name" loses the table context.
    col_texts = [
        f"{t}.{c}" for t, c in zip(col_table_map, columns)
    ]
    col_embeddings = embed_model.encode(col_texts, convert_to_tensor=True)

    # --- Table index (unique tables, deduplicated) ---
    tables = list(dict.fromkeys(col_table_map))  # preserves order, deduplicates
    # Raw table name — no prefix needed, BGE-M3 understands bare names
    table_embeddings = embed_model.encode(tables, convert_to_tensor=True)

    return SchemaIndex(
        db_name=db_name,
        tables=tables,
        table_embeddings=table_embeddings,
        columns=columns,
        col_node_ids=col_node_ids,
        col_table_map=col_table_map,
        col_embeddings=col_embeddings,
    )


# Keep the old signature as a thin compatibility shim used in pipeline.py
def precompute_column_embeddings(
    graph: nx.Graph,
    db_name: str,
    embed_model: SentenceTransformer,
) -> tuple[list[str], Optional[torch.Tensor]]:
    """Thin shim — prefer build_schema_index() for new code."""
    idx = build_schema_index(graph, db_name, embed_model)
    return idx.columns, idx.col_embeddings


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extract_phrases(query: str, max_ngram: int) -> list[str]:
    """Return all unique word n-grams (up to max_ngram) from a cleaned query."""
    clean = re.sub(r"[^\w\s]", "", query.lower())
    words = clean.split()
    return list(
        {
            " ".join(words[i : i + length])
            for length in range(1, max_ngram + 1)
            for i in range(len(words) - length + 1)
        }
    )


def _top_k_indices(score_row: torch.Tensor, k: int) -> list[int]:
    """Return indices of the k highest scores in a 1-D tensor."""
    k = min(k, score_row.size(0))
    return torch.topk(score_row, k).indices.tolist()


# ---------------------------------------------------------------------------
# Stage 1 — table retrieval
# ---------------------------------------------------------------------------

def retrieve_candidate_tables(
    query: str,
    index: SchemaIndex,
    embed_model: SentenceTransformer,
    cfg: PipelineConfig,
) -> list[str]:
    """
    Return the top-k most relevant table names for `query`.

    If `cfg.top_k_tables` is 0, all tables are returned (two-stage disabled).
    """
    if cfg.top_k_tables == 0 or index.table_embeddings is None:
        return index.tables

    phrases = _extract_phrases(query, cfg.max_ngram)
    if not phrases:
        return index.tables

    phrase_emb = embed_model.encode(phrases, convert_to_tensor=True)
    scores = util.cos_sim(phrase_emb, index.table_embeddings)  # (n_phrases, n_tables)

    # Max-pool across phrases: best score any phrase gave each table
    max_scores, _ = scores.max(dim=0)  # (n_tables,)

    selected_indices = _top_k_indices(max_scores, cfg.top_k_tables)
    selected_tables = [index.tables[i] for i in selected_indices]

    logger.debug("Stage 1 — candidate tables for '%s': %s", query[:50], selected_tables)
    return selected_tables


# ---------------------------------------------------------------------------
# Stage 2 — column retrieval (within candidate tables)
# ---------------------------------------------------------------------------

def retrieve_candidate_columns(
    query: str,
    index: SchemaIndex,
    candidate_tables: list[str],
    embed_model: SentenceTransformer,
    cfg: PipelineConfig,
) -> list[str]:
    """
    Return column names relevant to `query`, restricted to `candidate_tables`.

    Selection strategy:
      - If cfg.top_k_columns > 0: keep the top-k columns per query phrase.
      - Otherwise: keep every column whose best phrase score exceeds the threshold.
    """
    if index.col_embeddings is None:
        return []

    # Build a boolean mask for columns that belong to candidate tables
    candidate_table_set = set(candidate_tables)
    col_mask = torch.tensor(
        [t in candidate_table_set for t in index.col_table_map],
        dtype=torch.bool,
    )

    if not col_mask.any():
        return []

    # Filter embeddings and metadata to candidate tables only
    filtered_emb = index.col_embeddings[col_mask]            # (n_filtered, d)
    filtered_cols = [c for c, m in zip(index.columns, col_mask.tolist()) if m]

    phrases = _extract_phrases(query, cfg.max_ngram)
    if not phrases:
        return []

    phrase_emb = embed_model.encode(phrases, convert_to_tensor=True)
    scores = util.cos_sim(phrase_emb, filtered_emb)  # (n_phrases, n_filtered)

    detected: set[str] = set()

    if cfg.top_k_columns > 0:
        # Top-k strategy: for each phrase, pick the k best columns
        for i in range(len(phrases)):
            for col_idx in _top_k_indices(scores[i], cfg.top_k_columns):
                detected.add(filtered_cols[col_idx])
    else:
        # Threshold strategy (original behaviour)
        for i in range(len(phrases)):
            best_score = torch.max(scores[i]).item()
            if best_score > cfg.semantic_similarity_threshold:
                detected.add(filtered_cols[torch.argmax(scores[i]).item()])

    logger.debug("Stage 2 — detected columns: %s", list(detected))
    return list(detected)


# ---------------------------------------------------------------------------
# Public entry point — two-stage schema linking
# ---------------------------------------------------------------------------

def semantic_schema_linking(
    graph: nx.Graph,
    db_name: str,
    user_query: str,
    all_columns: list[str],
    col_embeddings: torch.Tensor,
    embed_model: SentenceTransformer,
    cfg: PipelineConfig,
    index: Optional[SchemaIndex] = None,
) -> list[str]:
    """
    Two-stage schema linking:
      Stage 1 — retrieve the top-k most relevant *tables*.
      Stage 2 — retrieve the top-k most relevant *columns* within those tables.

    Falls back gracefully:
      - If top_k_tables == 0, Stage 1 is skipped (all tables considered).
      - If top_k_columns == 0, Stage 2 uses the similarity threshold instead.
      - If `index` is None, a SchemaIndex is built on-the-fly from the legacy
        `all_columns` / `col_embeddings` arguments (backward compatible).
    """
    if index is None:
        # Reconstruct a minimal SchemaIndex from the legacy arguments
        # so the rest of the function works uniformly.
        col_nodes = [
            (n, d)
            for n, d in graph.nodes(data=True)
            if d.get("database") == db_name and d.get("type") == "column"
        ]
        tables = list(dict.fromkeys(d["table"] for _, d in col_nodes))
        col_table_map = [d["table"] for _, d in col_nodes]

        table_embeddings = embed_model.encode(tables, convert_to_tensor=True) if tables else None

        index = SchemaIndex(
            db_name=db_name,
            tables=tables,
            table_embeddings=table_embeddings,
            columns=all_columns,
            col_node_ids=[n for n, _ in col_nodes],
            col_table_map=col_table_map,
            col_embeddings=col_embeddings,
        )

    # Stage 1
    candidate_tables = retrieve_candidate_tables(user_query, index, embed_model, cfg)

    # Stage 2
    return retrieve_candidate_columns(user_query, index, candidate_tables, embed_model, cfg)


def trace_schema_paths(
    graph: nx.Graph,
    db_name: str,
    detected_columns: list[str],
) -> tuple[list[str], list[list[str]], list[dict]]:
    """
    For every pair of detected column nodes, find the shortest path in the graph.

    Returns:
        column_nodes : node IDs for the detected columns
        paths        : list of node-ID paths
        relations    : list of {from, to, relation} dicts
    """
    column_nodes = [
        n
        for n, d in graph.nodes(data=True)
        if d.get("database") == db_name and d.get("column") in detected_columns
    ]

    paths, relations = [], []
    for i in range(len(column_nodes)):
        for j in range(i + 1, len(column_nodes)):
            try:
                path = nx.shortest_path(graph, column_nodes[i], column_nodes[j])
                paths.append(path)
                for k in range(len(path) - 1):
                    edge_data = graph.get_edge_data(path[k], path[k + 1])
                    relations.append(
                        {
                            "from": path[k],
                            "to": path[k + 1],
                            "relation": edge_data.get("relation") if edge_data else None,
                        }
                    )
            except nx.NetworkXNoPath:
                continue

    return column_nodes, paths, relations


# ---------------------------------------------------------------------------
# Path pruning (Option B)
# ---------------------------------------------------------------------------

def prune_path_nodes(
    graph: nx.Graph,
    detected_column_names: set[str],
    all_path_nodes: list[str],
) -> list[str]:
    """
    Drop intermediate path nodes that carry no semantic signal.

    The shortest-path expansion between two detected columns often traverses
    "bridge" nodes — columns that exist only to connect two tables via FK/PK
    chains. Including every bridge node inflates the schema context with
    irrelevant columns and hurts precision.

    Pruning rule — keep a node if ANY of:
      1. Its column name was semantically detected from the query (direct hit).
      2. It is a PRIMARY KEY.
      3. It is a FOREIGN KEY (has at least one foreign_key edge).

    Drop if: it is only an intermediate traversal node with no PK/FK role.

    This preserves the join skeleton (PK/FK columns are always kept so the
    LLM can still write correct JOINs) while dropping pure noise columns.
    """
    kept: list[str] = []
    for node in all_path_nodes:
        d = graph.nodes.get(node, {})
        if not d:
            continue

        # Rule 1: direct semantic hit
        if d.get("column", "").lower() in detected_column_names:
            kept.append(node)
            continue

        # Rule 2: primary key
        if d.get("is_pk", False):
            kept.append(node)
            continue

        # Rule 3: foreign key (has at least one foreign_key edge)
        has_fk = any(
            graph.get_edge_data(node, nb, {}).get("relation") == "foreign_key"
            for nb in graph.neighbors(node)
        )
        if has_fk:
            kept.append(node)
            continue

        # Rule 4 (implicit): intermediate-only node — drop it

    return kept


# ---------------------------------------------------------------------------
# Context builder
# ---------------------------------------------------------------------------

def build_schema_context(graph: nx.Graph, column_nodes: list[str]) -> str:
    """
    Render a CREATE TABLE block for every table touched by `column_nodes`.
    Includes FK references where applicable.
    """
    # Group nodes by table
    tables: dict[str, list[dict]] = {}
    for node in column_nodes:
        d = graph.nodes[node]
        table = d["table"]
        tables.setdefault(table, []).append(
            {
                "col": d["column"],
                "col_type": d["column_type"],
                "is_pk": d["is_pk"],
                "node_id": node,
            }
        )

    lines = []
    for table, cols in tables.items():
        lines.append(f"CREATE TABLE {table} (")

        col_lines, fk_lines = [], []
        for col_data in cols:
            col, col_type, is_pk = col_data["col"], col_data["col_type"], col_data["is_pk"]
            col_lines.append(
                f"    {col} {col_type} PRIMARY KEY" if is_pk else f"    {col} {col_type}"
            )

            for neighbor in graph.neighbors(col_data["node_id"]):
                edge = graph.get_edge_data(col_data["node_id"], neighbor)
                if edge and edge.get("relation") == "foreign_key":
                    n_data = graph.nodes[neighbor]
                    if n_data["is_pk"] and n_data["table"] != table:
                        fk_lines.append(
                            f"    FOREIGN KEY ({col}) REFERENCES {n_data['table']}({n_data['column']})"
                        )

        lines.append(",\n".join(col_lines + fk_lines))
        lines.append(");\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Schema evaluation (Recall / Precision vs gold SQL)
# ---------------------------------------------------------------------------

# Pure SQL syntax keywords — never valid schema names.
# NOTE: "id" is intentionally NOT here. It is one of the most common FK/PK
# column names in Spider. Filtering it out would systematically undercount
# recall on join queries. Instead, _parse_gold_elements validates every token
# against the actual schema, so spurious "id" matches are caught structurally.
_SQL_STOPWORDS = frozenset({
    "select", "from", "where", "join", "on", "as", "and", "or", "not",
    "in", "is", "null", "like", "between", "exists", "case", "when",
    "then", "else", "end", "order", "by", "group", "having", "limit",
    "offset", "union", "intersect", "except", "distinct", "all", "asc",
    "desc", "count", "sum", "avg", "min", "max", "inner", "left", "right",
    "outer", "full", "cross", "natural", "using", "with", "create", "table",
    "insert", "update", "delete", "set", "values", "primary", "key",
    "foreign", "references", "index", "view", "drop", "alter", "add",
    # Alias placeholders — never real schema names
    "t1", "t2", "t3", "t4", "t5",
})


def _parse_gold_elements(gold_sql: str, graph: nx.Graph, db_name: str) -> set[str]:
    """
    Extract the set of table/column names that the gold SQL actually references,
    using structural matching against the known schema rather than raw token overlap.

    Strategy:
      1. Tokenise the SQL (split on whitespace + common delimiters).
      2. Filter out SQL keywords and stopwords.
      3. Accept a token only if it exactly matches a known table or column name
         for this database — this eliminates false positives from common words
         like 'id', 'name', or 'type' that happen to appear in the SQL as
         literals or aliases rather than schema references.
    """
    # Collect all known schema names for this database
    known_tables: set[str] = set()
    known_columns: set[str] = set()
    for _, d in graph.nodes(data=True):
        if d.get("database") != db_name:
            continue
        known_tables.add(d["table"].lower())
        col = d["column"].lower()
        if col != "*":
            known_columns.add(col)

    # Tokenise: split on whitespace and punctuation, lowercase
    raw_tokens = re.split(r"[\s\(\),;=<>!\"']+", gold_sql.lower())
    # Strip any remaining leading dots (e.g. ".column_name" after alias.col)
    tokens = {t.lstrip(".") for t in raw_tokens if t and t not in _SQL_STOPWORDS}

    gold_elements: set[str] = set()
    for token in tokens:
        if token in known_tables:
            gold_elements.add(token)
        if token in known_columns:
            gold_elements.add(token)

    return gold_elements


def evaluate_schema_linking(
    gold_sql: str,
    pruned_nodes: list[str],
    graph: nx.Graph,
    db_name: str,
) -> tuple[float, float]:
    """
    Compute recall and precision of retrieved schema elements vs gold SQL.

    Uses schema-aware matching (see _parse_gold_elements) instead of raw token
    overlap to avoid inflated recall from common short tokens like 'id'.

    Returns:
        recall    : fraction of gold elements that were retrieved
        precision : fraction of retrieved elements that are in the gold set
    """
    gold_elements = _parse_gold_elements(gold_sql, graph, db_name)

    if not gold_elements:
        return 1.0, 1.0  # Nothing to retrieve — perfect by convention

    retrieved: set[str] = set()
    for node in pruned_nodes:
        d = graph.nodes[node]
        retrieved.add(d["table"].lower())
        col = d["column"].lower()
        if col != "*":
            retrieved.add(col)

    hits = gold_elements & retrieved
    recall = len(hits) / len(gold_elements)
    precision = len(hits) / len(retrieved) if retrieved else 0.0
    return recall, precision
