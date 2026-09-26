"""
Ablation stage 1 — top-k fraction sweep, ranked by SLA (no LLM).

Grid (both from config.py's top_k_frame, see IMPLEMENTATION_DECISIONS.md poin 28):
    GraphRAG : top_k_tables_pct x top_k_columns_pct   (cross product, 16 configs)
    Baseline : baseline_top_k_tables_pct              (no column stage, 4 configs)
semantic_similarity_threshold stays at the config value unless --thresholds is
given (GraphRAG only; the baseline has no threshold).

Every config is scored with the reportable Schema Linking Accuracy
(src/metrics/sla.py, NOT the name-matching proxy of retrieval.py): the exact
"table.column" set the prompt would contain vs the gold schema from the
official Spider parser, precision/recall/F1 at table AND column level,
macro-averaged per query. Retrieval runs through the same
retrieve_graphrag_schema() / retrieve_baseline_tables() as the pipeline.

Winner per mode = highest --rank-by (default col_f1); ties go to the config
that sends fewer columns. The few-shot k (stage 2, ablation.py) and the model
(stage 3, pipeline.py --models) are chosen later, by EX.

Usage:
    python src/experiments/sweep.py                        # 20% dev set
    python src/experiments/sweep.py --sample 1.0
    python src/experiments/sweep.py --rank-by col_recall
    python src/experiments/sweep.py --thresholds 0.0 0.3 0.35 0.5

Output:
    outputs/tables/sweep_results.csv
"""

import argparse
import csv
import json
import logging
import sys
from dataclasses import asdict, dataclass, replace
from itertools import product
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.config import PipelineConfig
from src.core.schema import build_schema_graph, load_spider_schema
from src.metrics.sla import aggregate_sla, compute_sla, extract_ground_truth_schema
from src.retrieval.baseline import (
    _all_table_nodes,
    build_table_graph,
    build_table_index,
    retrieve_baseline_tables,
    table_nodes_to_schema,
)
from src.retrieval.retrieval import (
    build_schema_index,
    column_nodes_to_schema,
    retrieve_graphrag_schema,
)
from src.utils.schema_utils import load_db_schema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RANK_METRICS = ("col_f1", "col_recall", "col_precision", "table_f1", "table_recall", "table_precision")
SWEEP_CSV = Path("outputs/tables/sweep_results.csv")


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class SweepResult:
    mode:               str            # "graphrag" | "baseline"
    tables_pct:         float
    columns_pct:        float | None   # None for baseline (no column stage)
    sim_threshold:      float | None   # None for baseline
    table_precision:    float
    table_recall:       float
    table_f1:           float
    col_precision:      float
    col_recall:         float
    col_f1:             float
    avg_tables_sent:    float
    avg_cols_sent:      float          # columns in the prompt ("table.column" count)
    avg_cols_candidate: float | None   # GraphRAG: columns of the Stage-1 tables
    avg_cols_db:        float          # columns in the whole database
    n_samples:          int

    @property
    def label(self) -> str:
        if self.mode == "baseline":
            return f"tables={self.tables_pct:.2f}"
        return f"tables={self.tables_pct:.2f} cols={self.columns_pct:.2f} thr={self.sim_threshold:.2f}"


def _score(
    mode: str,
    tables_pct: float,
    columns_pct: float | None,
    sim_threshold: float | None,
    gold_pred: list[tuple[set, list]],
    tables_sent: list[int],
    cols_candidate: list[int] | None,
    cols_db: list[int],
) -> SweepResult:
    t = aggregate_sla([compute_sla(g, set(p), level="table") for g, p in gold_pred])
    c = aggregate_sla([compute_sla(g, set(p), level="column") for g, p in gold_pred])
    return SweepResult(
        mode=mode, tables_pct=tables_pct, columns_pct=columns_pct, sim_threshold=sim_threshold,
        table_precision=t.precision, table_recall=t.recall, table_f1=t.f1,
        col_precision=c.precision, col_recall=c.recall, col_f1=c.f1,
        avg_tables_sent=float(np.mean(tables_sent)),
        avg_cols_sent=float(np.mean([len(p) for _, p in gold_pred])),
        avg_cols_candidate=float(np.mean(cols_candidate)) if cols_candidate is not None else None,
        avg_cols_db=float(np.mean(cols_db)),
        n_samples=len(gold_pred),
    )


# ---------------------------------------------------------------------------
# One configuration per mode
# ---------------------------------------------------------------------------

def _run_graphrag(tables_pct, columns_pct, sim_threshold, samples, graph, cache,
                  n_cols_db, embed_model, base_cfg) -> SweepResult:
    cfg = replace(
        base_cfg,
        top_k_tables_pct=tables_pct,
        top_k_columns_pct=columns_pct,
        semantic_similarity_threshold=sim_threshold,
        use_full_schema_bypass=False,
    )
    gold_pred, tables_sent, cols_candidate, cols_db = [], [], [], []
    for item, gold in samples:
        db_id = item["db_id"]
        index = cache[db_id]
        column_nodes, candidate_tables = retrieve_graphrag_schema(
            graph, db_id, item["question"], index, embed_model, cfg,
        )
        predicted = column_nodes_to_schema(graph, column_nodes)
        gold_pred.append((gold, predicted))
        tables_sent.append(len({p.split(".", 1)[0] for p in predicted}))
        candidate = set(candidate_tables)
        cols_candidate.append(sum(1 for t in index.col_table_map if t in candidate))
        cols_db.append(n_cols_db[db_id])
    return _score("graphrag", tables_pct, columns_pct, sim_threshold,
                  gold_pred, tables_sent, cols_candidate, cols_db)


def _run_baseline(tables_pct, samples, tbl_graph, tbl_cache, n_cols_db,
                  embed_model, base_cfg) -> SweepResult:
    cfg = replace(base_cfg, baseline_top_k_tables_pct=tables_pct, use_full_schema_bypass=False)
    gold_pred, tables_sent, cols_db = [], [], []
    for item, gold in samples:
        db_id = item["db_id"]
        table_nodes = retrieve_baseline_tables(
            tbl_graph, db_id, item["question"], tbl_cache[db_id], embed_model, cfg,
        )
        gold_pred.append((gold, table_nodes_to_schema(tbl_graph, table_nodes)))
        tables_sent.append(len(table_nodes))
        cols_db.append(n_cols_db[db_id])
    return _score("baseline", tables_pct, None, None, gold_pred, tables_sent, None, cols_db)


# ---------------------------------------------------------------------------
# Selection & reporting
# ---------------------------------------------------------------------------

def pick_best(results: list[SweepResult], mode: str, rank_by: str) -> SweepResult:
    """Highest rank_by for `mode`; ties -> fewer columns sent (cheaper prompt)."""
    candidates = [r for r in results if r.mode == mode]
    return max(candidates, key=lambda r: (round(getattr(r, rank_by), 6), -r.avg_cols_sent))


def best_config(results: list[SweepResult], rank_by: str) -> dict:
    """Winners as {PipelineConfig field: value}, ready for setattr(cfg, ...)."""
    g = pick_best(results, "graphrag", rank_by)
    b = pick_best(results, "baseline", rank_by)
    return {
        "top_k_tables_pct": g.tables_pct,
        "top_k_columns_pct": g.columns_pct,
        "semantic_similarity_threshold": g.sim_threshold,
        "baseline_top_k_tables_pct": b.tables_pct,
    }


def _print_table(results: list[SweepResult], rank_by: str) -> None:
    W = 118
    for mode in ("graphrag", "baseline"):
        rows = sorted((r for r in results if r.mode == mode),
                      key=lambda r: (round(getattr(r, rank_by), 6), -r.avg_cols_sent), reverse=True)
        print("\n" + "=" * W)
        print(f"SWEEP — {mode.upper()}  ranked by SLA {rank_by}  (macro-average per query, src/metrics/sla.py)")
        print("=" * W)
        print(f"{'config':<34} {'tbl P':>7} {'tbl R':>7} {'tbl F1':>7} {'col P':>7} {'col R':>7} "
              f"{'col F1':>7} {'tbl':>5} {'cols':>6} {'cand':>6} {'db':>6}")
        print("-" * W)
        for i, r in enumerate(rows):
            cand = f"{r.avg_cols_candidate:>6.1f}" if r.avg_cols_candidate is not None else f"{'-':>6}"
            print(
                f"{r.label:<34} {r.table_precision*100:>6.1f}% {r.table_recall*100:>6.1f}% "
                f"{r.table_f1*100:>6.1f}% {r.col_precision*100:>6.1f}% {r.col_recall*100:>6.1f}% "
                f"{r.col_f1*100:>6.1f}% {r.avg_tables_sent:>5.1f} {r.avg_cols_sent:>6.1f} "
                f"{cand} {r.avg_cols_db:>6.1f}" + ("  <-- BEST" if i == 0 else "")
            )
        print("=" * W)
    print("tbl/cols = tables/columns in the prompt, cand = columns of the Stage-1 tables, db = columns in the DB")


def _save_csv(results: list[SweepResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(r) for r in results]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
    logger.info("CSV saved to %s", path)


# ---------------------------------------------------------------------------
# Public API — used by the CLI below, pipeline.py and eval_pipeline.ipynb
# ---------------------------------------------------------------------------

def run_sweep(
    sample_ratio: float,
    cfg: PipelineConfig,
    rank_by: str = "col_f1",
    thresholds: list[float] | None = None,
) -> tuple[list[SweepResult], dict]:
    """
    Run the whole stage-1 grid and return (all results, best_config()).
    Retrieval + SLA only — no LLM, so it is cheap (embedding lookups).
    """
    if rank_by not in RANK_METRICS:
        raise ValueError(f"rank_by must be one of {RANK_METRICS}, got {rank_by!r}")
    thresholds = thresholds or [cfg.semantic_similarity_threshold]

    logger.info("Loading schema and building both graphs …")
    schema_df = load_spider_schema(cfg.tables_json)
    col_graph = build_schema_graph(schema_df)
    tbl_graph = build_table_graph(schema_df)

    with open(cfg.dev_json, "r", encoding="utf-8") as f:
        dev_data = json.load(f)
    n = max(1, int(len(dev_data) * sample_ratio))
    rng = np.random.default_rng(cfg.seed)
    indices = rng.choice(len(dev_data), size=n, replace=False)
    dev_subset = [dev_data[i] for i in sorted(indices)]
    logger.info("Sweep subset: %d / %d questions", n, len(dev_data))

    # Gold schema once per question (official Spider parser, see sla.py).
    db_schema = load_db_schema(str(cfg.tables_json))
    samples, n_failed = [], 0
    for item in dev_subset:
        try:
            samples.append((item, extract_ground_truth_schema(item["query"], item["db_id"], db_schema)))
        except Exception:
            n_failed += 1
    if n_failed:
        logger.warning("%d gold SQL could not be parsed -> excluded from the sweep", n_failed)

    logger.info("Loading embedding model: %s", cfg.embedding_model)
    embed_model = SentenceTransformer(cfg.embedding_model)

    unique_db_ids = list(dict.fromkeys(item["db_id"] for item, _ in samples))
    col_cache = {db_id: build_schema_index(col_graph, db_id, embed_model)
                 for db_id in tqdm(unique_db_ids, desc="Indexing columns")}
    tbl_cache = {db_id: build_table_index(tbl_graph, db_id, embed_model)
                 for db_id in tqdm(unique_db_ids, desc="Indexing tables")}
    n_cols_db = {db_id: len(table_nodes_to_schema(tbl_graph, _all_table_nodes(tbl_graph, db_id)))
                 for db_id in unique_db_ids}

    frame = cfg.top_k_frame
    g_combos = list(product(frame, frame, thresholds))
    logger.info("Running %d GraphRAG + %d Baseline configs …", len(g_combos), len(frame))

    results: list[SweepResult] = []
    for tables_pct, columns_pct, thr in tqdm(g_combos, desc="Sweep GraphRAG"):
        r = _run_graphrag(tables_pct, columns_pct, thr, samples, col_graph, col_cache,
                          n_cols_db, embed_model, cfg)
        results.append(r)
        logger.info("  graphrag %s -> col F1=%.2f%% R=%.2f%% | cols sent=%.1f",
                    r.label, r.col_f1 * 100, r.col_recall * 100, r.avg_cols_sent)
    for tables_pct in tqdm(frame, desc="Sweep Baseline"):
        r = _run_baseline(tables_pct, samples, tbl_graph, tbl_cache, n_cols_db, embed_model, cfg)
        results.append(r)
        logger.info("  baseline %s -> col F1=%.2f%% R=%.2f%% | cols sent=%.1f",
                    r.label, r.col_f1 * 100, r.col_recall * 100, r.avg_cols_sent)

    _print_table(results, rank_by)
    _save_csv(results, SWEEP_CSV)

    best = best_config(results, rank_by)
    print(f"\nBest config (SLA {rank_by}):")
    for attr, value in best.items():
        print(f"  {attr} = {value}")
    print("-> Put these in config.py (or pass --tables-pct/--columns-pct/--baseline-tables-pct/"
          "--threshold) before running ablation.py (stage 2).\n")
    return results, best


def run_sweep_and_get_best(
    sample_ratio: float,
    cfg: PipelineConfig,
    rank_by: str = "col_f1",
    thresholds: list[float] | None = None,
) -> dict:
    """
    Stage 1 winners as {PipelineConfig field: value}. Called by pipeline.py at
    startup unless --skip-sweep is passed, and by eval_pipeline.ipynb.
    """
    return run_sweep(sample_ratio, cfg, rank_by, thresholds)[1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation stage 1: top-k fraction sweep ranked by SLA")
    parser.add_argument("--sample", type=float, default=0.2,
                        help="Fraction of dev set to use (default: 0.2 = 20%%)")
    parser.add_argument("--rank-by", choices=RANK_METRICS, default="col_f1",
                        help="SLA metric that picks the winner (default: col_f1).")
    parser.add_argument("--thresholds", type=float, nargs="+", default=None,
                        help="Also sweep semantic_similarity_threshold (GraphRAG). "
                             "Default: only the config.py value.")
    args, unknown = parser.parse_known_args()
    run_sweep(args.sample, PipelineConfig(), args.rank_by, args.thresholds)
