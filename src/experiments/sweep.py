"""
Precision sweep: top_k_tables × top_k_columns × semantic_similarity_threshold

Runs schema-linking ONLY (no LLM) on a subset of the dev set across
all combinations of the three parameters. Fast because no GPU inference
is involved — just embedding lookups. Also reports selectivity
(columns sent vs. columns available in the candidate tables).

Usage:
    python src/experiments/sweep.py                      # default: 20% dev set
    python src/experiments/sweep.py --sample 0.5        # 50% dev set
    python src/experiments/sweep.py --sample 1.0        # full dev set

Output:
    outputs/tables/sweep_results.csv   — raw per-combination metrics
    sweep_summary.txt   — human-readable ranked table
"""

import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.config import PipelineConfig
from src.retrieval.retrieval import (
    SchemaIndex,
    build_schema_index,
    evaluate_schema_linking,
    retrieve_candidate_columns,
    retrieve_candidate_tables,
    trace_schema_paths,
)
from src.core.schema import build_schema_graph, load_spider_schema

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
# tables=2 → recall ~97.5% (valid but higher recall is better)
# Start from tables=3 (recall ~99%).
TOP_K_TABLES_VALUES  = [3, 4, 5]
# Extended column grid (path pruning was removed — see IMPLEMENTATION_DECISIONS.md
# poin 25 — so precision is controlled by top_k_columns alone).
TOP_K_COLUMNS_VALUES = [2, 3, 5, 7, 10]
# Minimum cosine similarity for a column to be kept (0.0 = no threshold, top-k cap only).
# BGE-M3 scores are not calibrated, so these are guesses to be read off the sweep.
# NOTE: F6 weights recall 36x over precision, so ranking by F6 tends to favour the
# loosest setting; read the precision / cols-sent columns before trusting "best".
SIM_THRESHOLD_VALUES = [0.0, 0.3, 0.4, 0.5]

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
    sim_threshold:       float
    avg_recall:          float
    avg_precision:       float
    f1:                  float  # standard F1 (β=1), shown for reference
    f6:                  float  # F6 (β=6): peak correlation with EX (arxiv 2501.17174)
    meets_recall_target: bool   # avg_recall >= RECALL_THRESHOLD (informational)
    n_samples:           int
    # Selectivity diagnostic: columns actually sent to the prompt vs. all columns
    # of the Stage-1 candidate tables. sent/available near 1.0 = no column pruning.
    avg_cols_sent:       float
    avg_cols_available:  float


def _run_combination(
    top_k_tables: int,
    top_k_columns: int,
    sim_threshold: float,
    dev_subset: list[dict],
    graph,
    schema_cache: dict[str, SchemaIndex],
    embed_model: SentenceTransformer,
    base_cfg: PipelineConfig,
) -> SweepResult:
    """Evaluate schema-linking recall + precision for one (tables, cols, threshold) triple."""
    cfg = PipelineConfig(
        data_path=base_cfg.data_path,
        top_k_tables=top_k_tables,
        top_k_columns=top_k_columns,
        semantic_similarity_threshold=sim_threshold,
        few_shot_k=0,           # irrelevant for linking-only sweep
        use_full_schema_bypass=False,
    )

    recalls, precisions = [], []
    cols_sent, cols_available = [], []

    for item in dev_subset:
        db_id    = item["db_id"]
        question = item["question"]
        gold_sql = item["query"]
        index    = schema_cache[db_id]

        # Same two stages as semantic_schema_linking(), split so the candidate
        # tables are available for the selectivity diagnostic.
        candidate_tables = retrieve_candidate_tables(question, index, embed_model, cfg)
        detected_cols = retrieve_candidate_columns(
            question, index, candidate_tables, embed_model, cfg,
        )
        candidate_set = set(candidate_tables)
        cols_available.append(sum(1 for t in index.col_table_map if t in candidate_set))

        if not detected_cols:
            # fallback = full schema (same as pipeline)
            column_nodes = [
                n for n, d in graph.nodes(data=True)
                if d.get("database") == db_id and d.get("type") == "column"
            ]
        else:
            c_nodes, paths, _ = trace_schema_paths(graph, db_id, detected_cols)
            column_nodes = list(set(c_nodes + [node for path in paths for node in path]))

        cols_sent.append(len(column_nodes))
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
    # F6 (β=6): weights recall 36× more than precision.
    # Chosen because β=6 yields peak correlation with Execution Accuracy (EX)
    # among all F-beta variants — see arxiv 2501.17174.
    denom = 36 * avg_p + avg_r
    f6 = 37 * avg_r * avg_p / denom if denom > 0 else 0.0

    return SweepResult(
        top_k_tables=top_k_tables,
        top_k_columns=top_k_columns,
        sim_threshold=sim_threshold,
        avg_recall=avg_r,
        avg_precision=avg_p,
        f1=f1,
        f6=f6,
        meets_recall_target=meets_target,
        n_samples=len(dev_subset),
        avg_cols_sent=float(np.mean(cols_sent)),
        avg_cols_available=float(np.mean(cols_available)),
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _selectivity(r: SweepResult) -> float:
    """sent/avail: ~1.0 means the column stage barely pruned anything."""
    return r.avg_cols_sent / r.avg_cols_available if r.avg_cols_available else 0.0


def _format_row(r: SweepResult, note: str) -> str:
    target_str = "YES" if r.meets_recall_target else "NO "
    return (
        f"{r.top_k_tables:>8} {r.top_k_columns:>6} {r.sim_threshold:>5.2f} "
        f"{r.avg_recall*100:>8.2f}% {r.avg_precision*100:>10.2f}% "
        f"{r.f1*100:>7.2f}% {r.avg_cols_sent:>6.1f} {r.avg_cols_available:>6.1f} "
        f"{_selectivity(r):>10.2f} {target_str:>9}  {note}"
    )


def _print_table(results: list[SweepResult]) -> None:
    """Print two ranked tables: standard F1 and recall-weighted ranking."""
    W = 104

    # --- Table 1: ranked by standard F1 (for reference) ---
    header = (
        f"{'tables':>8} {'cols':>6} {'thr':>5} {'recall':>9} {'precision':>11} "
        f"{'F1':>8} {'sent':>6} {'avail':>6} {'sent/avail':>10} {'meets90%':>9}  note"
    )
    print("\n" + "=" * W)
    print("SWEEP RESULTS — ranked by standard F1  (reference only)")
    print("=" * W)
    print(header)
    print("-" * W)
    for i, r in enumerate(sorted(results, key=lambda r: r.f1, reverse=True)):
        note = "<-- best std-F1" if i == 0 else ""
        print(_format_row(r, note))
    print("=" * W)

    # --- Table 2: ranked by F6 (RECOMMENDED) ---
    print("\n" + "=" * W)
    print("SWEEP RESULTS — ranked by F6 score (β=6)  [RECOMMENDED]")
    print("  F6 = 37×P×R / (36P+R)  |  recall weighted 36× more than precision")
    print(f"  Peak EX correlation among F-beta variants (arxiv 2501.17174)  |  90% recall line shown for reference")
    print("=" * W)
    print(header)
    print("-" * W)
    for i, r in enumerate(sorted(results, key=lambda r: r.f6, reverse=True)):
        note = ""
        if i == 0:
            note = "<-- RECOMMENDED (best F6)"
        elif not r.meets_recall_target:
            note = f"recall < {RECALL_THRESHOLD*100:.0f}%"
        print(_format_row(r, note))
    print("=" * W)


def _save_csv(results: list[SweepResult], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["top_k_tables", "top_k_columns", "sim_threshold",
                        "avg_recall", "avg_precision", "f1", "f6",
                        "avg_cols_sent", "avg_cols_available", "sent_over_avail",
                        "meets_recall_target", "n_samples"],
        )
        writer.writeheader()
        for r in results:
            writer.writerow({
                "top_k_tables":        r.top_k_tables,
                "top_k_columns":       r.top_k_columns,
                "sim_threshold":       r.sim_threshold,
                "avg_recall":          round(r.avg_recall, 4),
                "avg_precision":       round(r.avg_precision, 4),
                "f1":                  round(r.f1, 4),
                "f6":                  round(r.f6, 4),
                "avg_cols_sent":       round(r.avg_cols_sent, 2),
                "avg_cols_available":  round(r.avg_cols_available, 2),
                "sent_over_avail":     round(_selectivity(r), 4),
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

    results = _run_grid(dev_subset, graph, schema_cache, embed_model, cfg)

    _print_table(results)
    _save_csv(results, Path("outputs/tables/sweep_results.csv"))

    # Recommend best config
    best = max(results, key=lambda r: r.f6)
    print(
        f"\nRecommended config (best F6): "
        f"top_k_tables={best.top_k_tables}, top_k_columns={best.top_k_columns}, "
        f"semantic_similarity_threshold={best.sim_threshold}\n"
        f"  recall={best.avg_recall*100:.2f}%  "
        f"precision={best.avg_precision*100:.2f}%  "
        f"meets_90%_target={'YES' if best.meets_recall_target else 'NO'}\n"
        f"  (F1={best.f1*100:.2f}%  F6={best.f6*100:.2f}%  "
        f"cols sent/available={_selectivity(best):.2f})"
    )
    if not best.meets_recall_target:
        print(
            f"  WARNING: best config still below {RECALL_THRESHOLD*100:.0f}% recall.\n"
            "  Consider expanding TOP_K_TABLES_VALUES in sweep.py."
        )
    print("→ Update these three values in config.py before running the full pipeline.\n")


def _run_grid(dev_subset, graph, schema_cache, embed_model, cfg) -> list[SweepResult]:
    """Run every (top_k_tables, top_k_columns, threshold) combination."""
    combos = list(product(TOP_K_TABLES_VALUES, TOP_K_COLUMNS_VALUES, SIM_THRESHOLD_VALUES))
    logger.info("Running %d combinations …", len(combos))

    results: list[SweepResult] = []
    for top_k_tables, top_k_columns, sim_threshold in tqdm(combos, desc="Sweep"):
        result = _run_combination(
            top_k_tables, top_k_columns, sim_threshold,
            dev_subset, graph, schema_cache, embed_model, cfg,
        )
        results.append(result)
        logger.info(
            "  tables=%d cols=%d thr=%.2f → recall=%.2f%% precision=%.2f%% "
            "F1=%.2f%% sent/avail=%.2f",
            top_k_tables, top_k_columns, sim_threshold,
            result.avg_recall * 100, result.avg_precision * 100, result.f1 * 100,
            _selectivity(result),
        )
    return results


# ---------------------------------------------------------------------------
# Public API — called by pipeline.py
# ---------------------------------------------------------------------------

def run_sweep_and_get_best(
    sample_ratio: float,
    cfg,
) -> tuple[int, int, float]:
    """
    Run the full precision sweep and return
    (best_top_k_tables, best_top_k_columns, best_similarity_threshold).

    Called automatically by pipeline.py at startup unless --skip-sweep is passed.
    The sweep uses embedding lookups only — no LLM inference — so it is fast.
    Results are also saved to sweep_results.csv for your records.
    """
    import json
    from sentence_transformers import SentenceTransformer
    from src.core.schema import build_schema_graph, load_spider_schema

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

    results = _run_grid(dev_subset, graph, schema_cache, embed_model, cfg)

    _print_table(results)
    _save_csv(results, Path("outputs/tables/sweep_results.csv"))

    best = max(results, key=lambda r: r.f6)
    logger.info(
        "Best config (recall-weighted): top_k_tables=%d top_k_columns=%d thr=%.2f "
        "recall=%.2f%% precision=%.2f%% sent/avail=%.2f",
        best.top_k_tables, best.top_k_columns, best.sim_threshold,
        best.avg_recall * 100, best.avg_precision * 100, _selectivity(best),
    )
    return best.top_k_tables, best.top_k_columns, best.sim_threshold


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Schema-linking precision sweep")
    parser.add_argument(
        "--sample", type=float, default=0.2,
        help="Fraction of dev set to use (default: 0.2 = 20%%)",
    )
    args, unknown = parser.parse_known_args()
    main(args.sample)
