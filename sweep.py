"""
Precision sweep: top_k_tables × top_k_columns

Runs schema-linking ONLY (no LLM) on a subset of the dev set across
all combinations of top_k_tables and top_k_columns. Fast because no
GPU inference is involved — just embedding lookups.

Usage:
    python sweep.py                      # default: 20% dev set
    python sweep.py --sample 0.5        # 50% dev set
    python sweep.py --sample 1.0        # full dev set

Output:
    sweep_results.csv   — raw per-combination metrics
    sweep_summary.txt   — human-readable ranked table
"""

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import PipelineConfig
from retrieval import (
    SchemaIndex,
    build_schema_index,
    evaluate_schema_linking,
    prune_path_nodes,
    semantic_schema_linking,
    trace_schema_paths,
)
from schema import build_schema_graph, load_spider_schema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Grid to sweep
# ---------------------------------------------------------------------------

# tables=1 → recall ~83% (below 90% threshold, never valid)
# tables=2 → recall ~97.5% (valid but pruning makes higher recall better)
# Start from tables=3 (recall ~99%) — pruning handles precision from here.
TOP_K_TABLES_VALUES  = [3, 4, 5]
# Extended column grid — pruning will strip irrelevant columns anyway,
# so higher values are safe to test.
TOP_K_COLUMNS_VALUES = [2, 3, 5, 7, 10]

# Minimum recall required before we consider a config valid.
# Text-to-SQL needs high recall — a missing table = guaranteed wrong SQL.
RECALL_THRESHOLD = 0.90


# ---------------------------------------------------------------------------
# Single combination evaluation
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    top_k_tables:        int
    top_k_columns:       int
    avg_recall:          float
    avg_precision:       float
    f1:                  float  # standard harmonic mean
    recall_weighted_f1:  float  # recall-weighted: only maximise precision
                                #   among configs that meet RECALL_THRESHOLD
    meets_recall_target: bool   # avg_recall >= RECALL_THRESHOLD
    n_samples:           int


def _run_combination(
    top_k_tables: int,
    top_k_columns: int,
    dev_subset: list[dict],
    graph,
    schema_cache: dict[str, SchemaIndex],
    embed_model: SentenceTransformer,
    base_cfg: PipelineConfig,
) -> SweepResult:
    """Evaluate schema-linking recall + precision for one (tables, cols) pair."""
    cfg = PipelineConfig(
        data_path=base_cfg.data_path,
        top_k_tables=top_k_tables,
        top_k_columns=top_k_columns,
        few_shot_k=0,           # irrelevant for linking-only sweep
        use_full_schema_bypass=False,
    )

    recalls, precisions = [], []

    for item in dev_subset:
        db_id    = item["db_id"]
        question = item["question"]
        gold_sql = item["query"]
        index    = schema_cache[db_id]

        detected_cols = semantic_schema_linking(
            graph, db_id, question,
            all_columns=index.columns,
            col_embeddings=index.col_embeddings,
            embed_model=embed_model,
            cfg=cfg,
            index=index,
        )

        if not detected_cols:
            # fallback = full schema (same as pipeline)
            column_nodes = [
                n for n, d in graph.nodes(data=True)
                if d.get("database") == db_id and d.get("type") == "column"
            ]
        else:
            c_nodes, paths, _ = trace_schema_paths(graph, db_id, detected_cols)
            all_path_nodes = list(set(c_nodes + [node for path in paths for node in path]))
            # Apply same pruning as pipeline so sweep metrics match LLM input
            detected_col_names = {c.lower() for c in detected_cols}
            column_nodes = prune_path_nodes(graph, detected_col_names, all_path_nodes)
            if not column_nodes:
                column_nodes = all_path_nodes

        if not column_nodes:
            recalls.append(0.0)
            precisions.append(0.0)
            continue

        r, p = evaluate_schema_linking(gold_sql, column_nodes, graph, db_id)
        recalls.append(r)
        precisions.append(p)

    avg_r = float(np.mean(recalls))
    avg_p = float(np.mean(precisions))
    f1    = (2 * avg_r * avg_p / (avg_r + avg_p)) if (avg_r + avg_p) > 0 else 0.0

    meets_target = avg_r >= RECALL_THRESHOLD
    # Recall-weighted F1: configs below the recall threshold are penalised
    # by setting their score to recall only (no precision credit).
    # Among configs that meet the threshold, rank purely by precision
    # since recall is already "good enough".
    if meets_target:
        rw_f1 = avg_p          # maximise precision once recall target is met
    else:
        rw_f1 = avg_r * 0.5   # penalise: half credit, recall-only

    return SweepResult(
        top_k_tables=top_k_tables,
        top_k_columns=top_k_columns,
        avg_recall=avg_r,
        avg_precision=avg_p,
        f1=f1,
        recall_weighted_f1=rw_f1,
        meets_recall_target=meets_target,
        n_samples=len(dev_subset),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_table(results: list[SweepResult]) -> None:
    """Print two ranked tables: standard F1 and recall-weighted ranking."""
    W = 74

    # --- Table 1: ranked by standard F1 (for reference) ---
    header = f"{'tables':>8} {'cols':>6} {'recall':>9} {'precision':>11} {'F1':>8} {'meets90%':>9}  note"
    print("\n" + "=" * W)
    print("SWEEP RESULTS — ranked by standard F1  (reference only)")
    print("=" * W)
    print(header)
    print("-" * W)
    for i, r in enumerate(sorted(results, key=lambda r: r.f1, reverse=True)):
        target_str = "YES" if r.meets_recall_target else "NO "
        note = "<-- best std-F1" if i == 0 else ""
        print(
            f"{r.top_k_tables:>8} {r.top_k_columns:>6} "
            f"{r.avg_recall*100:>8.2f}% {r.avg_precision*100:>10.2f}% "
            f"{r.f1*100:>7.2f}% {target_str:>9}  {note}"
        )
    print("=" * W)

    # --- Table 2: ranked by recall-weighted score (RECOMMENDED) ---
    print("\n" + "=" * W)
    print(
        f"SWEEP RESULTS — ranked by recall-weighted score  "
        f"(recall threshold: {RECALL_THRESHOLD*100:.0f}%)"
    )
    print("  Logic: configs below recall threshold are penalised.")
    print("         among configs that meet threshold, rank by precision.")
    print("=" * W)
    print(header)
    print("-" * W)
    for i, r in enumerate(sorted(results, key=lambda r: r.recall_weighted_f1, reverse=True)):
        target_str = "YES" if r.meets_recall_target else "NO "
        note = ""
        if i == 0:
            note = "<-- RECOMMENDED"
        elif not r.meets_recall_target:
            note = f"recall too low (<{RECALL_THRESHOLD*100:.0f}%)"
        print(
            f"{r.top_k_tables:>8} {r.top_k_columns:>6} "
            f"{r.avg_recall*100:>8.2f}% {r.avg_precision*100:>10.2f}% "
            f"{r.f1*100:>7.2f}% {target_str:>9}  {note}"
        )
    print("=" * W)


def _save_csv(results: list[SweepResult], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["top_k_tables", "top_k_columns", "avg_recall",
                        "avg_precision", "f1", "recall_weighted_f1",
                        "meets_recall_target", "n_samples"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow({
                "top_k_tables":        r.top_k_tables,
                "top_k_columns":       r.top_k_columns,
                "avg_recall":          round(r.avg_recall, 4),
                "avg_precision":       round(r.avg_precision, 4),
                "f1":                  round(r.f1, 4),
                "recall_weighted_f1":  round(r.recall_weighted_f1, 4),
                "meets_recall_target": r.meets_recall_target,
                "n_samples":           r.n_samples,
            })
    logger.info("CSV saved to %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(sample_ratio: float) -> None:
    cfg = PipelineConfig()

    logger.info("Loading schema and building graph …")
    schema_df = load_spider_schema(cfg.tables_json)
    graph     = build_schema_graph(schema_df)

    logger.info("Loading dev set …")
    with open(cfg.dev_json, "r", encoding="utf-8") as f:
        dev_data = json.load(f)

    # Deterministic subsample
    n = max(1, int(len(dev_data) * sample_ratio))
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dev_data), size=n, replace=False)
    dev_subset = [dev_data[i] for i in sorted(indices)]
    logger.info("Sweep subset: %d / %d questions (%.0f%%)", n, len(dev_data), sample_ratio * 100)

    logger.info("Loading embedding model: %s", cfg.embedding_model)
    embed_model = SentenceTransformer(cfg.embedding_model)

    # Pre-build schema indices for all DBs in subset
    unique_db_ids = list(dict.fromkeys(item["db_id"] for item in dev_subset))
    logger.info("Building schema indices for %d databases …", len(unique_db_ids))
    schema_cache: dict[str, SchemaIndex] = {
        db_id: build_schema_index(graph, db_id, embed_model)
        for db_id in tqdm(unique_db_ids, desc="Indexing")
    }

    # Run grid
    combos = list(product(TOP_K_TABLES_VALUES, TOP_K_COLUMNS_VALUES))
    logger.info("Running %d combinations …", len(combos))

    results: list[SweepResult] = []
    for top_k_tables, top_k_columns in tqdm(combos, desc="Sweep"):
        result = _run_combination(
            top_k_tables, top_k_columns,
            dev_subset, graph, schema_cache, embed_model, cfg,
        )
        results.append(result)
        logger.info(
            "  tables=%d cols=%d → recall=%.2f%% precision=%.2f%% F1=%.2f%%",
            top_k_tables, top_k_columns,
            result.avg_recall * 100, result.avg_precision * 100, result.f1 * 100,
        )

    _print_table(results)
    _save_csv(results, Path("sweep_results.csv"))

    # Recommend best config
    best = max(results, key=lambda r: r.recall_weighted_f1)
    print(
        f"\nRecommended config (recall-weighted): "
        f"top_k_tables={best.top_k_tables}, top_k_columns={best.top_k_columns}\n"
        f"  recall={best.avg_recall*100:.2f}%  "
        f"precision={best.avg_precision*100:.2f}%  "
        f"meets_90%_target={'YES' if best.meets_recall_target else 'NO'}\n"
        f"  (standard F1={best.f1*100:.2f}%, "
        f"recall-weighted score={best.recall_weighted_f1*100:.2f}%)"
    )
    if not best.meets_recall_target:
        print(
            f"  WARNING: best config still below {RECALL_THRESHOLD*100:.0f}% recall.\n"
            "  Consider expanding TOP_K_TABLES_VALUES in sweep.py."
        )
    print("→ Update these two values in config.py before running the full pipeline.\n")



# ---------------------------------------------------------------------------
# Public API — called by pipeline.py
# ---------------------------------------------------------------------------

def run_sweep_and_get_best(
    sample_ratio: float,
    cfg,
) -> tuple[int, int]:
    """
    Run the full precision sweep and return (best_top_k_tables, best_top_k_columns).

    Called automatically by pipeline.py at startup unless --skip-sweep is passed.
    The sweep uses embedding lookups only — no LLM inference — so it is fast.
    Results are also saved to sweep_results.csv for your records.
    """
    import json
    from sentence_transformers import SentenceTransformer
    from schema import build_schema_graph, load_spider_schema

    logger.info("Loading schema for sweep …")
    schema_df = load_spider_schema(cfg.tables_json)
    graph     = build_schema_graph(schema_df)

    with open(cfg.dev_json, "r", encoding="utf-8") as f:
        dev_data = json.load(f)

    n = max(1, int(len(dev_data) * sample_ratio))
    rng = np.random.default_rng(cfg.seed)
    indices = rng.choice(len(dev_data), size=n, replace=False)
    dev_subset = [dev_data[i] for i in sorted(indices)]
    logger.info("Sweep subset: %d / %d questions", n, len(dev_data))

    logger.info("Loading embedding model for sweep: %s", cfg.embedding_model)
    embed_model = SentenceTransformer(cfg.embedding_model)

    unique_db_ids = list(dict.fromkeys(item["db_id"] for item in dev_subset))
    schema_cache = {
        db_id: build_schema_index(graph, db_id, embed_model)
        for db_id in tqdm(unique_db_ids, desc="Sweep — indexing")
    }

    combos = list(product(TOP_K_TABLES_VALUES, TOP_K_COLUMNS_VALUES))
    results: list[SweepResult] = []
    for top_k_tables, top_k_columns in tqdm(combos, desc="Sweep — grid"):
        result = _run_combination(
            top_k_tables, top_k_columns,
            dev_subset, graph, schema_cache, embed_model, cfg,
        )
        results.append(result)
        logger.info(
            "  tables=%d cols=%d → recall=%.2f%% precision=%.2f%% F1=%.2f%%",
            top_k_tables, top_k_columns,
            result.avg_recall * 100, result.avg_precision * 100, result.f1 * 100,
        )

    _print_table(results)
    _save_csv(results, Path("sweep_results.csv"))

    best = max(results, key=lambda r: r.recall_weighted_f1)
    logger.info(
        "Best config (recall-weighted): top_k_tables=%d top_k_columns=%d "
        "recall=%.2f%% precision=%.2f%%",
        best.top_k_tables, best.top_k_columns,
        best.avg_recall * 100, best.avg_precision * 100,
    )
    return best.top_k_tables, best.top_k_columns


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Schema-linking precision sweep")
    parser.add_argument(
        "--sample", type=float, default=0.2,
        help="Fraction of dev set to use (default: 0.2 = 20%%)",
    )
    args, unknown = parser.parse_known_args()
    main(args.sample)
