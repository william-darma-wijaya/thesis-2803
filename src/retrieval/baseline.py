"""
Baseline comparison module: Table-level Graph vs Column-level Graph (GraphRAG).

This module implements the *table/node* baseline from the notebook experiment
(graph-rag-text-to-sql-baseline-vs-graph.ipynb) and integrates it cleanly into
the existing pipeline architecture.

Architecture difference:
  - GraphRAG  (column/node) : each node = one column. Semantic linking matches
                               query phrases to individual column embeddings.
  - Baseline  (table/node)  : each node = one table + all its columns. Semantic
                               linking matches the whole query to table-level
                               embeddings. Coarser granularity → higher recall,
                               lower precision.

Everything after retrieval is identical for both arms, INCLUDING few-shot:
the baseline gets the same cfg.few_shot_k examples as GraphRAG, so the
comparison isolates the retrieval strategy (IMPLEMENTATION_DECISIONS.md poin 28).

Usage:
    python src/retrieval/baseline.py                           # default 20% dev-set sample
    python src/retrieval/baseline.py --sample 0.5             # 50% dev set
    python src/retrieval/baseline.py --sample 1.0             # full dev set
    python src/retrieval/baseline.py --skip-sweep             # skip top-k sweep

For a side-by-side GraphRAG-vs-Baseline comparison with per-query ESM/EX/CM and
data/raw_logs/*.json, use `python src/experiments/pipeline.py --baseline`
instead — this module only runs the baseline arm on its own.

The script can also be imported and called from pipeline.py via run_baseline().

Output files:
    outputs/predictions/baseline_predictions.txt    — SQL predictions (Spider format)
    outputs/logs/baseline_log.txt                   — per-sample detail log
    outputs/tables/baseline_results.csv             — recall / precision per sample
"""

import argparse
import csv
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.config import PipelineConfig
from src.generation.few_shot import FewShotIndex, build_few_shot_block, build_few_shot_index
from src.generation.generation import build_prompt, generate_sql_with_token_count, load_model_and_tokenizer
from src.retrieval.retrieval import k_from_pct
from src.core.schema import load_spider_schema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TableSchemaIndex:
    """Precomputed table-level embeddings for one database (baseline variant)."""
    db_name: str
    table_node_ids: list[str]          # e.g. ["concert_singer.singer", ...]
    table_embeddings: torch.Tensor     # shape (n_tables, d)


@dataclass
class BaselineResult:
    index: int
    db_id: str
    question: str
    gold_sql: str
    pred_sql: str
    recall: float
    precision: float


# ---------------------------------------------------------------------------
# Step 1 — Build TABLE-level graph
# ---------------------------------------------------------------------------

def build_table_graph(schema_df) -> nx.Graph:
    """
    Build a graph where each node represents ONE TABLE (not one column).

    Node attributes:
        database   : str  — Spider db_id
        table      : str  — table name
        columns    : list[dict]  — [{"Column": ..., "Type": ..., "is_pk": ...}]
        type       : "table"

    Edge attributes:
        relation   : "foreign_key"
        from_col   : source column name
        to_col     : target column name

    This is the baseline described in the notebook: coarser granularity means
    the schema context given to the LLM is always at the full-table level,
    whereas GraphRAG prunes down to individual columns.
    """
    G = nx.Graph()
    df_clean = schema_df[schema_df["Column"] != "*"].copy()

    # One node per (database, table) pair
    for (db, table), group in df_clean.groupby(["Database", "Table"]):
        node_id = f"{db}.{table}"
        columns = (
            group[["Column", "Type", "PK"]]
            .rename(columns={"PK": "is_pk"})
            .to_dict("records")
        )
        G.add_node(
            node_id,
            database=db,
            table=table,
            columns=columns,
            type="table",
        )

    # Edges via FK relations (table → table)
    fk_rows = df_clean[df_clean["FK_Relation"] != "-"]
    for _, row in fk_rows.iterrows():
        src = f"{row['Database']}.{row['Table']}"
        parts = row["FK_Relation"].split(".")
        if len(parts) != 2:
            continue
        tgt_table, tgt_col = parts
        tgt = f"{row['Database']}.{tgt_table}"
        if G.has_node(src) and G.has_node(tgt) and not G.has_edge(src, tgt):
            G.add_edge(
                src, tgt,
                relation="foreign_key",
                from_col=row["Column"],
                to_col=tgt_col,
            )

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    logger.info("Table graph built: %d table-nodes, %d FK-edges", n_nodes, n_edges)
    return G


# ---------------------------------------------------------------------------
# Step 2 — Precompute table-level embeddings
# ---------------------------------------------------------------------------

def build_table_index(
    graph: nx.Graph,
    db_name: str,
    embed_model: SentenceTransformer,
) -> TableSchemaIndex:
    """
    Encode each table as a text string that includes the table name AND all
    its column names. This gives the embedder more signal than just the table
    name alone, and mirrors the approach from the notebook.

    Text format: "table <name> columns <col1> <col2> ..."
    """
    nodes = [
        (n, d)
        for n, d in graph.nodes(data=True)
        if d.get("database") == db_name and d.get("type") == "table"
    ]

    if not nodes:
        return TableSchemaIndex(db_name, [], torch.zeros(0))

    table_node_ids = []
    texts = []
    for node_id, d in nodes:
        col_names = " ".join(c["Column"] for c in d["columns"])
        texts.append(f"table {d['table']} columns {col_names}")
        table_node_ids.append(node_id)

    embeddings = embed_model.encode(texts, convert_to_tensor=True)
    return TableSchemaIndex(
        db_name=db_name,
        table_node_ids=table_node_ids,
        table_embeddings=embeddings,
    )


# ---------------------------------------------------------------------------
# Step 3 — Semantic schema linking (table level)
# ---------------------------------------------------------------------------

def semantic_linking_table_level(
    index: TableSchemaIndex,
    user_query: str,
    embed_model: SentenceTransformer,
    top_k: int = 3,
) -> list[str]:
    """
    Match the user query against table embeddings and return the top-k most
    similar table node IDs.

    Unlike the column-level approach (which decomposes the query into n-grams
    and matches each phrase separately), this does a SINGLE query-level
    embedding match — reflecting the notebook's table-level baseline design.

    Returns:
        List of table node IDs (e.g. ["concert_singer.singer", ...])
    """
    if not index.table_node_ids:
        return []

    query_emb = embed_model.encode(
        [f"query {user_query}"], convert_to_tensor=True
    )
    scores = util.cos_sim(query_emb, index.table_embeddings)[0]
    k = min(top_k, len(index.table_node_ids))
    top_indices = scores.topk(k).indices.tolist()
    return [index.table_node_ids[i] for i in top_indices]


# ---------------------------------------------------------------------------
# Step 4 — Path tracing (table level)
# ---------------------------------------------------------------------------

def trace_table_paths(
    graph: nx.Graph,
    detected_table_nodes: list[str],
) -> list[str]:
    """
    Expand the set of detected tables by including any intermediate tables
    that lie on the shortest FK-path between detected tables.

    Example: if we detect [Singer, Concert] and there is a FK path
    Singer → Singer_in_concert → Concert, the intermediate table
    Singer_in_concert is added to ensure the JOIN is possible.

    Returns:
        Deduplicated list of table node IDs, detected tables first, in a
        deterministic order (a set's order changes between Python processes
        and would reorder the CREATE TABLE blocks in the prompt).
    """
    all_nodes = dict.fromkeys(detected_table_nodes)

    for i in range(len(detected_table_nodes)):
        for j in range(i + 1, len(detected_table_nodes)):
            try:
                path = nx.shortest_path(
                    graph, detected_table_nodes[i], detected_table_nodes[j]
                )
                all_nodes.update(dict.fromkeys(path))
            except nx.NetworkXNoPath:
                pass

    return list(all_nodes)


# ---------------------------------------------------------------------------
# Step 5 — Build schema context string (table level)
# ---------------------------------------------------------------------------

def build_table_schema_context(
    graph: nx.Graph,
    table_nodes: list[str],
) -> str:
    """
    Render a CREATE TABLE DDL block for each selected table node.

    Because each node already carries ALL column info (unlike the column-level
    graph), we include every column in the retrieved tables — no pruning.
    FK annotations are derived from the graph edges.

    Returns:
        Multi-line string suitable for injection into the LLM prompt.
    """
    lines = []
    for node in table_nodes:
        if node not in graph.nodes:
            continue
        d = graph.nodes[node]
        table = d["table"]
        lines.append(f"CREATE TABLE {table} (")

        col_lines = []
        for col in d["columns"]:
            line = f"    {col['Column']} {col['Type']}"
            if col.get("is_pk") == "Yes":
                line += " PRIMARY KEY"
            col_lines.append(line)

        fk_lines = []
        this_cols = {col["Column"] for col in d["columns"]}
        for _, neighbor, ed in graph.edges(node, data=True):
            # Only annotate if this node is the FK source (from_col lives here,
            # not on the referenced/parent side of the edge).
            if ed.get("relation") == "foreign_key" and ed.get("from_col") in this_cols:
                neighbor_table = graph.nodes[neighbor]["table"]
                fk_lines.append(
                    f"    FOREIGN KEY ({ed['from_col']}) "
                    f"REFERENCES {neighbor_table}({ed['to_col']})"
                )

        lines.append(",\n".join(col_lines + fk_lines))
        lines.append(");\n")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Step 6 — Evaluate schema linking (table level)
# ---------------------------------------------------------------------------

def evaluate_table_linking(
    gold_sql: str,
    retrieved_table_nodes: list[str],
    graph: nx.Graph,
    db_name: str,
) -> tuple[float, float]:
    """
    Compute recall and precision of the table-level schema linking step.

    ⚠️ INTERNAL PROGRESS PROXY — NOT the reportable Schema Linking Accuracy.
    Same limitation as evaluate_schema_linking() in retrieval.py: unqualified
    table + column name strings in one mixed set, no table.column qualification.
    Fine as a relative tuning signal; do NOT report as SLA. The reportable
    metric is src/metrics/sla.py (parser-based, table/column split, validated).
    See context/IMPLEMENTATION_DECISIONS.md poin 19.

    Uses schema-aware matching: tokens are validated against known schema names
    and SQL stopwords are filtered out, preventing spurious matches on tokens
    like 'id', 'name', or 'type' that appear in SQL as literals or aliases.

    Returns:
        (recall, precision) — floats in [0, 1]
    """
    from src.retrieval.retrieval import _SQL_STOPWORDS

    # Build known schema names from the table-level graph for this database
    known_tables: set[str] = set()
    known_columns: set[str] = set()
    for _, d in graph.nodes(data=True):
        if d.get("database") != db_name:
            continue
        known_tables.add(d["table"].lower())
        for col in d.get("columns", []):
            c = col["Column"].lower()
            if c != "*":
                known_columns.add(c)

    # Schema-aware gold element parsing (mirrors _parse_gold_elements)
    raw_tokens = re.split(r'[\s\(\),;=<>!"\']+', gold_sql.lower())
    tokens = {t.lstrip(".") for t in raw_tokens if t and t not in _SQL_STOPWORDS}

    gold_elements: set[str] = set()
    for token in tokens:
        if token in known_tables:
            gold_elements.add(token)
        if token in known_columns:
            gold_elements.add(token)

    if not gold_elements:
        return 1.0, 1.0

    # Build retrieved element set from selected table nodes
    retrieved_elements: set[str] = set()
    for node in retrieved_table_nodes:
        if node not in graph.nodes:
            continue
        d = graph.nodes[node]
        retrieved_elements.add(d["table"].lower())
        for col in d.get("columns", []):
            retrieved_elements.add(col["Column"].lower())

    if not retrieved_elements:
        return 0.0, 0.0

    hit = gold_elements & retrieved_elements
    recall = len(hit) / len(gold_elements)
    precision = len(hit) / len(retrieved_elements)
    return recall, precision


# ---------------------------------------------------------------------------
# Core per-sample step
# ---------------------------------------------------------------------------

def _all_table_nodes(graph: nx.Graph, db_id: str) -> list[str]:
    """All table node ids for `db_id` — used whenever retrieval is bypassed
    or disabled (full-schema bypass, baseline_top_k_tables_pct=0, or no tables
    detected)."""
    return [
        n for n, d in graph.nodes(data=True)
        if d.get("database") == db_id and d.get("type") == "table"
    ]


def retrieve_baseline_tables(
    graph: nx.Graph,
    db_id: str,
    question: str,
    table_index: TableSchemaIndex,
    embed_model: SentenceTransformer,
    cfg: PipelineConfig,
) -> list[str]:
    """
    The whole Baseline retrieval step for one question, shared by
    run_single_baseline(), ablation.py, sweep.py and the notebook:
    top-k tables (k = k_from_pct(cfg.baseline_top_k_tables_pct, tables in DB))
    -> FK path tracing. baseline_top_k_tables_pct=0 or full-schema bypass ->
    every table, same semantics as GraphRAG's top_k_tables_pct=0.
    """
    if cfg.use_full_schema_bypass or cfg.baseline_top_k_tables_pct <= 0:
        return _all_table_nodes(graph, db_id)

    k = k_from_pct(cfg.baseline_top_k_tables_pct, len(table_index.table_node_ids))
    detected = semantic_linking_table_level(table_index, question, embed_model, top_k=k)
    if not detected:
        return _all_table_nodes(graph, db_id)
    return trace_table_paths(graph, detected)


def table_nodes_to_schema(graph: nx.Graph, table_nodes: list[str]) -> list[str]:
    """
    Table node ids -> sorted ["table.column", ...] (raw_logs predicted_schema /
    SLA input). Baseline sends ALL columns of the selected tables, so every
    column of each table node is listed, not just the table names.
    """
    out = set()
    for node in table_nodes:
        d = graph.nodes[node]
        table = d.get("table")
        if not table:
            continue
        for col_info in d.get("columns", []):
            column = col_info.get("Column")
            if column and column != "*":
                out.add(f"{table.lower()}.{column.lower()}")
    return sorted(out)


def run_single_baseline(
    question: str,
    gold_sql: str,
    db_id: str,
    graph: nx.Graph,
    embed_model: SentenceTransformer,
    model,
    tokenizer,
    cfg: PipelineConfig,
    table_index: TableSchemaIndex,
    few_shot_index: FewShotIndex | None = None,
) -> tuple[str, float, float, list, int, int]:
    """
    Run the full BASELINE pipeline for one question:
        embed → link tables → trace paths → build context → few-shot → generate SQL.

    few_shot_index : same index GraphRAG uses; cfg.few_shot_k examples are
                     added exactly like run_single() does. None -> zero-shot.

    Returns:
        (pred_sql, recall, precision, table_nodes, n_in, n_out)
        table_nodes : selected table node ids (used for raw_logs predicted_schema —
                      baseline sends ALL columns of these tables, no pruning)
        n_in, n_out : prompt/output token counts (T_in/T_out), same
                      generate_sql_with_token_count() pattern as pipeline.py's
                      run_single() and ablation.py's _run_k()
    """
    table_nodes = retrieve_baseline_tables(graph, db_id, question, table_index, embed_model, cfg)
    if not table_nodes:
        return "SELECT 1", 0.0, 0.0, [], 0, 0

    recall, precision = evaluate_table_linking(gold_sql, table_nodes, graph, db_id)
    schema_context = build_table_schema_context(graph, table_nodes)
    few_shot_block = build_few_shot_block(question, few_shot_index, embed_model, cfg)

    extracted_values = {
        "strings": re.findall(r"'([^']*)'", question),
        "numbers": re.findall(r"\d+", question),
    }
    prompt = build_prompt(question, schema_context, extracted_values, few_shot_block)
    pred_sql, n_out = generate_sql_with_token_count(prompt, model, tokenizer, cfg)
    n_in = len(tokenizer.encode(prompt))

    return pred_sql, recall, precision, table_nodes, n_in, n_out


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def _save_predictions(results: list[BaselineResult], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(r.pred_sql.strip() + "\n")
    logger.info("Predictions saved → %s", path)


def _save_log(results: list[BaselineResult], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"[{r.index}] DB: {r.db_id}\n")
            f.write(f"  Q      : {r.question}\n")
            f.write(f"  GOLD   : {r.gold_sql}\n")
            f.write(f"  PRED   : {r.pred_sql}\n")
            f.write(f"  Recall : {r.recall:.4f}  Precision: {r.precision:.4f}\n")
            f.write("-" * 60 + "\n")
    logger.info("Log saved → %s", path)


def _save_csv(results: list[BaselineResult], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["index", "db_id", "recall", "precision"]
        )
        writer.writeheader()
        for r in results:
            writer.writerow({
                "index": r.index,
                "db_id": r.db_id,
                "recall": round(r.recall, 4),
                "precision": round(r.precision, 4),
            })
    logger.info("CSV saved → %s", path)


def _print_summary(results: list[BaselineResult], label: str = "Baseline (table/node)") -> None:
    valid = [r for r in results if r.recall >= 0]
    if not valid:
        return
    avg_r = sum(r.recall for r in valid) / len(valid) * 100
    avg_p = sum(r.precision for r in valid) / len(valid) * 100
    W = 60
    print("\n" + "=" * W)
    print(f"  SCHEMA LINKING SUMMARY — {label}")
    print("=" * W)
    print(f"  Recall    : {avg_r:.2f}%  (gold elements retrieved)")
    print(f"  Precision : {avg_p:.2f}%  (retrieved elements that were relevant)")
    print(f"  Samples   : {len(valid)}")
    print("=" * W)


# ---------------------------------------------------------------------------
# Official Spider evaluation
# ---------------------------------------------------------------------------

def _run_spider_eval(pred_file: Path, cfg: PipelineConfig) -> None:
    evaluator = Path("external/spider_eval/evaluation.py")
    if not evaluator.exists():
        logger.warning(
            "external/spider_eval/evaluation.py not found — skipping Spider eval.\n"
            "  mkdir -p external/spider_eval && cd external/spider_eval\n"
            "  wget https://raw.githubusercontent.com/taoyds/spider/master/evaluation.py\n"
            "  wget https://raw.githubusercontent.com/taoyds/spider/master/process_sql.py"
        )
        return

    base_args = [
        "--gold", str(cfg.gold_sql),
        "--pred", str(pred_file),
        "--db",   str(cfg.db_dir),
        "--table", str(cfg.tables_json),
    ]
    print("\n" + "=" * 60)
    print("  SPIDER EVALUATION (Exact Match)")
    print("=" * 60)
    subprocess.run(["python", "external/spider_eval/evaluation.py"] + base_args + ["--etype", "match"], check=False)

    print("\n" + "=" * 60)
    print("  SPIDER EVALUATION (Execution Accuracy)")
    print("=" * 60)
    subprocess.run(["python", "external/spider_eval/evaluation.py"] + base_args + ["--etype", "exec"], check=False)


# ---------------------------------------------------------------------------
# Public API — called from pipeline.py or standalone
# ---------------------------------------------------------------------------

def run_baseline(
    cfg: PipelineConfig,
    sample_ratio: float = 1.0,
    pred_path: Path = Path("outputs/predictions/baseline_predictions.txt"),
    log_path:  Path = Path("outputs/logs/baseline_log.txt"),
    csv_path:  Path = Path("outputs/tables/baseline_results.csv"),
    run_eval:  bool = True,
) -> list[BaselineResult]:
    """
    Run the full table-level baseline pipeline.

    Args:
        cfg          : PipelineConfig (paths, model names, etc.)
        sample_ratio : fraction of dev set to evaluate (0 < ratio ≤ 1.0)
        pred_path    : where to write Spider-format predictions
        log_path     : where to write detailed per-sample log
        csv_path     : where to write per-sample metrics CSV
        run_eval     : whether to call Spider evaluation.py at the end

    Returns:
        List of BaselineResult (one per dev sample evaluated).
    """
    # 1. Schema
    logger.info("Loading schema and building TABLE-LEVEL graph …")
    schema_df = load_spider_schema(cfg.tables_json)
    table_graph = build_table_graph(schema_df)

    # 2. Dev data
    logger.info("Loading dev set …")
    with open(cfg.dev_json, "r", encoding="utf-8") as f:
        dev_data = json.load(f)

    if sample_ratio < 1.0:
        n = max(1, int(len(dev_data) * sample_ratio))
        rng = np.random.default_rng(cfg.seed)
        indices = rng.choice(len(dev_data), size=n, replace=False)
        dev_data = [dev_data[i] for i in sorted(indices)]
        logger.info("Evaluating %d / total samples (%.0f%%)", len(dev_data), sample_ratio * 100)

    # 3. Embedding model
    logger.info("Loading embedding model: %s", cfg.embedding_model)
    embed_model = SentenceTransformer(cfg.embedding_model)

    # 4. LLM
    logger.info("Loading LLM: %s (4-bit)", cfg.llm_model)
    llm, tokenizer = load_model_and_tokenizer(cfg)

    # 5. Precompute table-level indices for all DB IDs in the sample
    unique_db_ids = list(dict.fromkeys(item["db_id"] for item in dev_data))
    logger.info("Building table indices for %d databases …", len(unique_db_ids))
    table_cache: dict[str, TableSchemaIndex] = {
        db_id: build_table_index(table_graph, db_id, embed_model)
        for db_id in tqdm(unique_db_ids, desc="Indexing tables")
    }

    # 6. Few-shot index — same k as GraphRAG (cfg.few_shot_k)
    few_shot_idx: FewShotIndex | None = None
    if cfg.few_shot_k > 0:
        logger.info("Building few-shot index (k=%d) …", cfg.few_shot_k)
        few_shot_idx = build_few_shot_index(cfg.train_json, embed_model)

    # 7. Main loop
    results: list[BaselineResult] = []

    with open(pred_path, "w", encoding="utf-8") as pf:
        for i, item in enumerate(tqdm(dev_data, desc="Baseline — generating SQL")):
            db_id    = item["db_id"]
            question = item["question"]
            gold_sql = item["query"]

            try:
                pred_sql, recall, precision, _table_nodes, _n_in, _n_out = run_single_baseline(
                    question, gold_sql, db_id,
                    table_graph, embed_model, llm, tokenizer, cfg,
                    table_index=table_cache[db_id],
                    few_shot_index=few_shot_idx,
                )
            except Exception:
                logger.exception("Error on sample %d (db=%s)", i, db_id)
                pred_sql, recall, precision = "SELECT 1", 0.0, 0.0

            result = BaselineResult(
                index=i + 1,
                db_id=db_id,
                question=question,
                gold_sql=gold_sql,
                pred_sql=pred_sql,
                recall=recall,
                precision=precision,
            )
            results.append(result)
            pf.write(pred_sql.strip() + "\n")

            # Progress print every 20 samples
            if (i + 1) % 20 == 0 or i == 0 or (i + 1) == len(dev_data):
                logger.info(
                    "[%d/%d] DB=%-20s recall=%.2f  precision=%.2f",
                    i + 1, len(dev_data), db_id, recall, precision,
                )

    # 8. Save artefacts
    _save_log(results, log_path)
    _save_csv(results, csv_path)
    _print_summary(results, label="Baseline (table/node)")

    # 9. Spider eval
    if run_eval:
        _run_spider_eval(pred_path, cfg)

    return results


# ---------------------------------------------------------------------------
# NOTE (2026-09-06): the old `run_comparison()` here was removed. It ran an
# older GraphRAG-vs-Baseline path that produced ONLY recall/precision +
# subprocess Spider eval — no per-query ESM/EX/CM, no data/raw_logs/*.json.
# `src/experiments/pipeline.py`'s `run_comparison()` fully supersedes it
# (per-query ESM/EX/CM, raw_logs, confound diagnostics). Use:
#     python src/experiments/pipeline.py --baseline
# This module stays standalone-runnable for the baseline arm on its own via
# `run_baseline()` below.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Baseline (table/node) pipeline for GraphRAG comparison."
    )
    parser.add_argument(
        "--sample", type=float, default=0.2,
        help="Fraction of dev set to evaluate (default: 0.2 = 20%%).",
    )
    parser.add_argument(
        "--skip-sweep", action="store_true",
        help="Skip the top-k precision sweep (use config.py values as-is).",
    )
    parser.add_argument(
        "--full-schema", action="store_true",
        help="Bypass table linking — feed the entire DB schema to the LLM.",
    )
    args = parser.parse_args()

    cfg = PipelineConfig(use_full_schema_bypass=args.full_schema)

    # For a side-by-side GraphRAG-vs-Baseline run with per-query ESM/EX/CM and
    # data/raw_logs/*.json, use `python src/experiments/pipeline.py --baseline`.
    run_baseline(cfg, sample_ratio=args.sample)
