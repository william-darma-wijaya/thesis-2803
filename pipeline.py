"""
Main entry point for the GraphRAG Text-to-SQL pipeline.

Usage:
    python pipeline.py [--full-schema]

Flags:
    --full-schema   Bypass GraphRAG retrieval and feed the entire DB schema to the LLM.
                    Useful for ablation studies.
"""

import argparse
import json
import logging
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import PipelineConfig
from sweep import run_sweep_and_get_best
from few_shot import FewShotIndex, build_few_shot_index, format_few_shot_block, retrieve_few_shot_examples
from generation import build_prompt, generate_sql, load_model_and_tokenizer
from retrieval import (
    SchemaIndex,
    build_schema_context,
    build_schema_index,
    evaluate_schema_linking,
    precompute_column_embeddings,
    prune_path_nodes,
    semantic_schema_linking,
    trace_schema_paths,
)
from schema import build_schema_graph, load_spider_schema

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data class for a single pipeline result
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    index: int
    db_id: str
    question: str
    gold_sql: str
    pred_sql: str
    recall: float
    precision: float
    retrieved_schema: str = ''   # CREATE TABLE DDL fed to the LLM
    gold_schema: str = ''        # DDL reconstructed from gold SQL elements



# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def _bar(value: float, width: int = 20) -> str:
    """Render a compact ASCII progress bar for a 0–1 float."""
    filled = round(value * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


# ---------------------------------------------------------------------------
# Core pipeline step
# ---------------------------------------------------------------------------

def run_single(
    question: str,
    gold_sql: str,
    db_id: str,
    graph,
    embed_model: SentenceTransformer,
    model,
    tokenizer,
    cfg: PipelineConfig,
    schema_index: SchemaIndex = None,
    few_shot_index: FewShotIndex = None,
) -> tuple[str, float, float, str, list]:
    """
    Run the full retrieval + generation pipeline for one question.

    Args:
        schema_index    : pre-built SchemaIndex from build_schema_index()
        few_shot_index  : pre-built FewShotIndex from build_few_shot_index()
                          Pass None to run zero-shot.
    Returns:
        pred_sql  : generated SQL string
        recall    : schema-linking recall vs gold SQL
        precision : schema-linking precision vs gold SQL
    """
    # --- Build schema index if not supplied (fallback, not the hot path) ---
    if schema_index is None:
        schema_index = build_schema_index(graph, db_id, embed_model)

    # --- Schema retrieval ---
    if cfg.use_full_schema_bypass:
        column_nodes = [
            n
            for n, d in graph.nodes(data=True)
            if d.get("database") == db_id and d.get("type") == "column"
        ]
    else:
        detected_cols = semantic_schema_linking(
            graph, db_id, question,
            all_columns=schema_index.columns,
            col_embeddings=schema_index.col_embeddings,
            embed_model=embed_model,
            cfg=cfg,
            index=schema_index,
        )

        if not detected_cols:
            column_nodes = [
                n
                for n, d in graph.nodes(data=True)
                if d.get("database") == db_id and d.get("type") == "column"
            ]
        else:
            c_nodes, paths, _ = trace_schema_paths(graph, db_id, detected_cols)
            # Collect all path nodes, then prune intermediate-only nodes
            all_path_nodes = list(set(c_nodes + [node for path in paths for node in path]))
            detected_col_names = {c.lower() for c in detected_cols}
            column_nodes = prune_path_nodes(graph, detected_col_names, all_path_nodes)
            # Fallback: if pruning removed everything, keep original expansion
            if not column_nodes:
                column_nodes = all_path_nodes

    if not column_nodes:
        return "SELECT 1", 0.0, 0.0, "", []

    # --- Schema linking evaluation ---
    recall, precision = evaluate_schema_linking(gold_sql, column_nodes, graph, db_id)

    # --- Few-shot example retrieval ---
    few_shot_block = ""
    if few_shot_index is not None and cfg.few_shot_k > 0:
        examples = retrieve_few_shot_examples(
            question, db_id, few_shot_index, embed_model, cfg
        )
        few_shot_block = format_few_shot_block(examples)

    # --- Prompt construction & SQL generation ---
    schema_context = build_schema_context(graph, column_nodes)
    extracted_values = {
        "strings": re.findall(r"'([^']*)'", question),
        "numbers": re.findall(r"\d+", question),
    }
    prompt = build_prompt(question, schema_context, extracted_values, few_shot_block)
    pred_sql = generate_sql(prompt, model, tokenizer, cfg)

    return pred_sql, recall, precision, schema_context, column_nodes


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def run_official_evaluation(cfg: PipelineConfig) -> None:
    """Run the Spider official evaluation script for EM and EX metrics."""
    evaluator = Path("evaluation.py")
    if not evaluator.exists():
        logger.warning(
            "evaluation.py not found — skipping official Spider evaluation.\n"
            "Download it with:\n"
            "  wget https://raw.githubusercontent.com/taoyds/spider/master/evaluation.py\n"
            "  wget https://raw.githubusercontent.com/taoyds/spider/master/process_sql.py"
        )
        return

    common_args = [
        "--gold", str(cfg.gold_sql),
        "--pred", str(cfg.predictions_file),
        "--db", str(cfg.db_dir),
        "--table", str(cfg.tables_json),
    ]
    print("\n" + "=" * 60)
    print("🎯 OFFICIAL SPIDER EVALUATION (Exact Match)")
    print("=" * 60)
    subprocess.run(["python", "evaluation.py"] + common_args + ["--etype", "match"], check=False)

    print("\n" + "=" * 60)
    print("🎯 OFFICIAL SPIDER EVALUATION (Execution Accuracy)")
    print("=" * 60)
    subprocess.run(["python", "evaluation.py"] + common_args + ["--etype", "exec"], check=False)


def print_schema_linking_summary(results: list[PipelineResult]) -> None:
    valid = [r for r in results if r.recall >= 0]
    if not valid:
        return
    avg_recall = sum(r.recall for r in valid) / len(valid) * 100
    avg_precision = sum(r.precision for r in valid) / len(valid) * 100

    print("\n" + "=" * 60)
    print("📈 SCHEMA LINKING SUMMARY (GraphRAG)")
    print("=" * 60)
    print(f"  Recall    : {avg_recall:.2f}%  — how many gold elements were retrieved")
    print(f"  Precision : {avg_precision:.2f}%  — fraction of retrieved elements that were relevant")
    print(f"  Samples   : {len(valid)}")




# ---------------------------------------------------------------------------
# Gold schema helper
# ---------------------------------------------------------------------------

def build_gold_schema_context(
    gold_sql: str,
    graph,
    db_id: str,
) -> str:
    """
    Reconstruct a CREATE TABLE DDL for the tables/columns that the gold SQL
    actually references. Used purely for display — lets you compare what the
    model received vs what the gold answer required.
    """
    from retrieval import _parse_gold_elements, build_schema_context

    gold_elements = _parse_gold_elements(gold_sql, graph, db_id)

    # Find nodes whose table or column name appears in gold elements
    gold_nodes = [
        n for n, d in graph.nodes(data=True)
        if d.get("database") == db_id
        and d.get("type") == "column"
        and (d["table"].lower() in gold_elements or d["column"].lower() in gold_elements)
    ]

    if not gold_nodes:
        return "(could not parse gold schema)"

    return build_schema_context(graph, gold_nodes)

# ---------------------------------------------------------------------------
# Seed & model setup
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(cfg: PipelineConfig) -> None:
    set_seed(cfg.seed)

    # --- 1. Load schema & build graph ---
    logger.info("Loading schema and building graph …")
    schema_df = load_spider_schema(cfg.tables_json)
    graph = build_schema_graph(schema_df)

    # --- 2. Load dev questions ---
    logger.info("Loading Spider dev set …")
    with open(cfg.dev_json, "r", encoding="utf-8") as f:
        dev_data = json.load(f)
    logger.info("Dev set size: %d questions", len(dev_data))

    # --- 3. Load models ---
    logger.info("Loading embedding model: %s", cfg.embedding_model)
    embed_model = SentenceTransformer(cfg.embedding_model)

    logger.info("Loading LLM: %s (4-bit quantised)", cfg.llm_model)
    llm, tokenizer = load_model_and_tokenizer(cfg)

    # --- 4. Pre-build schema indices for every database in the dev set ---
    # Done ONCE at startup so embeddings are never recomputed mid-loop.
    mode_label = "FULL SCHEMA (bypass)" if cfg.use_full_schema_bypass else "GraphRAG"
    logger.info("Starting pipeline — mode: %s", mode_label)

    unique_db_ids = list(dict.fromkeys(item["db_id"] for item in dev_data))
    logger.info("Pre-building schema indices for %d databases …", len(unique_db_ids))
    schema_cache: dict[str, SchemaIndex] = {
        db_id: build_schema_index(graph, db_id, embed_model)
        for db_id in tqdm(unique_db_ids, desc="Indexing schemas")
    }

    # --- 5. Build few-shot index from training set (if enabled) ---
    few_shot_idx: FewShotIndex | None = None
    if cfg.few_shot_k > 0:
        logger.info("Building few-shot index from training set …")
        few_shot_idx = build_few_shot_index(cfg.train_json, embed_model)
    else:
        logger.info("Few-shot disabled (few_shot_k=0), running zero-shot.")

    results: list[PipelineResult] = []

    with open(cfg.predictions_file, "w", encoding="utf-8") as pred_file:
        for i, item in enumerate(tqdm(dev_data, desc="Generating SQL")):
            db_id = item["db_id"]
            question = item["question"]
            gold_sql = item["query"]

            try:
                pred_sql, recall, precision, retrieved_schema, column_nodes = run_single(
                    question, gold_sql, db_id, graph, embed_model, llm, tokenizer, cfg,
                    schema_index=schema_cache[db_id],
                    few_shot_index=few_shot_idx,
                )
                gold_schema = build_gold_schema_context(gold_sql, graph, db_id)
            except Exception:
                logger.exception("Error on sample %d (db=%s)", i, db_id)
                pred_sql, recall, precision = "SELECT 1", 0.0, 0.0
                retrieved_schema, gold_schema, column_nodes = "", "", []

            result = PipelineResult(
                index=i + 1,
                db_id=db_id,
                question=question,
                gold_sql=gold_sql,
                pred_sql=pred_sql,
                recall=recall,
                precision=precision,
                retrieved_schema=retrieved_schema,
                gold_schema=gold_schema,
            )
            results.append(result)
            pred_file.write(f"{pred_sql}\n")

            # ── Rich progress block ──────────────────────────────────────
            W = 72
            em_mark = "MATCH" if gold_sql.strip().lower() == pred_sql.strip().lower() else "DIFF"

            print("\n" + "=" * W)
            print(f"  [{i+1:>4}/{len(dev_data)}]  DB: {db_id:<24} {em_mark}")
            print("=" * W)
            print(f"  Q        : {question}")
            print("-" * W)
            print(f"  Recall   : {_bar(recall)}  {recall*100:5.1f}%")
            print(f"  Precision: {_bar(precision)}  {precision*100:5.1f}%")
            print("-" * W)
            print(f"  GOLD SQL : {gold_sql}")
            print(f"  PRED SQL : {pred_sql}")
            print("-" * W)
            print("  RETRIEVED SCHEMA:")
            for line in retrieved_schema.strip().splitlines():
                print(f"    {line}")
            print("-" * W)
            print("  GOLD SCHEMA (inferred from gold SQL):")
            for line in gold_schema.strip().splitlines():
                print(f"    {line}")
            print("=" * W)

    logger.info("Predictions saved to %s", cfg.predictions_file)

    # --- 6. Print schema linking summary ---
    print_schema_linking_summary(results)

    # --- 7. Official Spider evaluation ---
    run_official_evaluation(cfg)



# ---------------------------------------------------------------------------
# Comparison: GraphRAG vs Baseline side-by-side
# ---------------------------------------------------------------------------

def run_comparison(cfg: PipelineConfig, sample_ratio: float) -> None:
    """
    Run GraphRAG (column/node) AND Baseline (table/node) on the same
    dev-set sample, then print a side-by-side schema-linking + prediction
    comparison report.

    Predictions are saved to:
        predictions.txt          — GraphRAG
        baseline_predictions.txt — Baseline
        comparison_report.txt    — side-by-side summary
    """
    import numpy as np
    from baseline import (
        BaselineResult,
        build_table_graph,
        build_table_index,
        run_single_baseline,
        _save_predictions as bl_save_pred,
        _save_log        as bl_save_log,
        _save_csv        as bl_save_csv,
        _print_summary   as bl_print_summary,
        _run_spider_eval as bl_spider_eval,
    )

    set_seed(cfg.seed)

    # ── Shared setup ──────────────────────────────────────────────────────
    logger.info("Loading schema …")
    schema_df = load_spider_schema(cfg.tables_json)

    logger.info("Loading dev set …")
    with open(cfg.dev_json, "r", encoding="utf-8") as f:
        dev_data = json.load(f)

    # Deterministic subsample (same for both modes)
    if sample_ratio < 1.0:
        n = max(1, int(len(dev_data) * sample_ratio))
        rng = np.random.default_rng(cfg.seed)
        indices = rng.choice(len(dev_data), size=n, replace=False)
        dev_data = [dev_data[i] for i in sorted(indices)]
        logger.info(
            "Comparison sample: %d / total questions (%.0f%%)",
            len(dev_data), sample_ratio * 100,
        )

    logger.info("Loading embedding model: %s", cfg.embedding_model)
    embed_model = SentenceTransformer(cfg.embedding_model)

    logger.info("Loading LLM: %s (4-bit)", cfg.llm_model)
    llm, tokenizer = load_model_and_tokenizer(cfg)

    unique_db_ids = list(dict.fromkeys(item["db_id"] for item in dev_data))

    # ── Build both graphs & indices ───────────────────────────────────────
    logger.info("Building column graph (GraphRAG) …")
    col_graph = build_schema_graph(schema_df)
    col_cache: dict[str, "SchemaIndex"] = {
        db_id: build_schema_index(col_graph, db_id, embed_model)
        for db_id in tqdm(unique_db_ids, desc="Indexing columns")
    }

    logger.info("Building table graph (Baseline) …")
    tbl_graph = build_table_graph(schema_df)
    tbl_cache = {
        db_id: build_table_index(tbl_graph, db_id, embed_model)
        for db_id in tqdm(unique_db_ids, desc="Indexing tables")
    }

    # Few-shot index (GraphRAG only)
    few_shot_idx = None
    if cfg.few_shot_k > 0:
        from few_shot import build_few_shot_index
        logger.info("Building few-shot index …")
        few_shot_idx = build_few_shot_index(cfg.train_json, embed_model)

    # ── Main loop — run both pipelines on the same sample ─────────────────
    graphrag_results: list[PipelineResult] = []
    baseline_results: list[BaselineResult] = []

    g_pred_path = Path("predictions.txt")
    b_pred_path = Path("baseline_predictions.txt")

    W = 72
    with open(g_pred_path, "w", encoding="utf-8") as gf, \
         open(b_pred_path, "w", encoding="utf-8") as bf:

        for i, item in enumerate(tqdm(dev_data, desc="Running both pipelines")):
            db_id    = item["db_id"]
            question = item["question"]
            gold_sql = item["query"]

            # GraphRAG
            try:
                g_pred, g_recall, g_prec, g_schema, _ = run_single(
                    question, gold_sql, db_id,
                    col_graph, embed_model, llm, tokenizer, cfg,
                    schema_index=col_cache[db_id],
                    few_shot_index=few_shot_idx,
                )
            except Exception:
                logger.exception("GraphRAG error on sample %d", i)
                g_pred, g_recall, g_prec, g_schema = "SELECT 1", 0.0, 0.0, ""

            # Baseline
            try:
                b_pred, b_recall, b_prec = run_single_baseline(
                    question, gold_sql, db_id,
                    tbl_graph, embed_model, llm, tokenizer, cfg,
                    table_index=tbl_cache[db_id],
                )
            except Exception:
                logger.exception("Baseline error on sample %d", i)
                b_pred, b_recall, b_prec = "SELECT 1", 0.0, 0.0

            graphrag_results.append(PipelineResult(
                index=i + 1, db_id=db_id, question=question,
                gold_sql=gold_sql, pred_sql=g_pred,
                recall=g_recall, precision=g_prec,
            ))
            baseline_results.append(BaselineResult(
                index=i + 1, db_id=db_id, question=question,
                gold_sql=gold_sql, pred_sql=b_pred,
                recall=b_recall, precision=b_prec,
            ))

            gf.write(g_pred.strip() + "\n")
            bf.write(b_pred.strip() + "\n")

            # Per-sample progress print
            if (i + 1) % 20 == 0 or i == 0 or (i + 1) == len(dev_data):
                print("\n" + "=" * W)
                print(f"  [{i+1:>4}/{len(dev_data)}]  DB: {db_id}")
                print("=" * W)
                print(f"  Q              : {question}")
                print(f"  GOLD           : {gold_sql}")
                print("-" * W)
                print(f"  [GraphRAG] pred: {g_pred}")
                print(f"             R={g_recall*100:.1f}%  P={g_prec*100:.1f}%")
                print(f"  [Baseline] pred: {b_pred}")
                print(f"             R={b_recall*100:.1f}%  P={b_prec*100:.1f}%")
                print("=" * W)

    # ── Save baseline artefacts ────────────────────────────────────────────
    bl_save_log(baseline_results, Path("baseline_log.txt"))
    bl_save_csv(baseline_results, Path("baseline_results.csv"))

    # ── Schema linking summaries ───────────────────────────────────────────
    print_schema_linking_summary(graphrag_results)
    bl_print_summary(baseline_results, label="Baseline (table/node)")

    # ── Side-by-side comparison report ────────────────────────────────────
    g_r = np.mean([r.recall    for r in graphrag_results]) * 100
    g_p = np.mean([r.precision for r in graphrag_results]) * 100
    b_r = np.mean([r.recall    for r in baseline_results]) * 100
    b_p = np.mean([r.precision for r in baseline_results]) * 100

    report_lines = [
        "",
        "=" * 70,
        "  COMPARISON REPORT — GraphRAG (column/node) vs Baseline (table/node)",
        "=" * 70,
        f"  {'Metric':<22} {'GraphRAG':>12}  {'Baseline':>12}  {'Delta (G-B)':>12}",
        "-" * 70,
        f"  {'Schema Recall':<22} {g_r:>11.2f}%  {b_r:>11.2f}%  {g_r-b_r:>+11.2f}%",
        f"  {'Schema Precision':<22} {g_p:>11.2f}%  {b_p:>11.2f}%  {g_p-b_p:>+11.2f}%",
        "-" * 70,
        "  Node granularity      column/node      table/node",
        "  Schema context        pruned cols      all cols in table",
        "  Linking strategy      n-gram match     single query embed",
        "=" * 70,
        "",
    ]
    report = "\n".join(report_lines)
    print(report)
    Path("comparison_report.txt").write_text(report, encoding="utf-8")
    logger.info("Comparison report → comparison_report.txt")

    # ── Official Spider eval for both ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SPIDER EVAL — GraphRAG (column/node)")
    print("=" * 60)
    run_official_evaluation(cfg)

    print("\n" + "=" * 60)
    print("  SPIDER EVAL — Baseline (table/node)")
    print("=" * 60)
    bl_spider_eval(b_pred_path, cfg)


# ---------------------------------------------------------------------------
# CLI — GANTI if __name__ == "__main__" yang lama dengan ini
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphRAG Text-to-SQL Pipeline")
    parser.add_argument(
        "--full-schema", action="store_true",
        help="Bypass GraphRAG and feed the full DB schema to the LLM (ablation mode).",
    )
    parser.add_argument(
        "--skip-sweep", action="store_true",
        help="Skip the precision sweep and use top_k values from config.py as-is.",
    )
    parser.add_argument(
        "--sweep-sample", type=float, default=0.2,
        help="Fraction of dev set used for the precision sweep (default: 0.2).",
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="Also run the table/node baseline and print a comparison report.",
    )
    parser.add_argument(
        "--sample", type=float, default=1.0,
        help="Fraction of dev set to evaluate when --baseline is used (default: 1.0).",
    )
    args, unknown = parser.parse_known_args()

    config = PipelineConfig(use_full_schema_bypass=args.full_schema)

    # Sweep (skipped automatically when --baseline is used or --full-schema)
    if not args.skip_sweep and not config.use_full_schema_bypass and not args.baseline:
        logger.info("=" * 60)
        logger.info("STEP 0: Precision sweep (pass --skip-sweep to skip)")
        logger.info("=" * 60)
        best_tables, best_cols = run_sweep_and_get_best(
            sample_ratio=args.sweep_sample,
            cfg=config,
        )
        logger.info(
            "Sweep done — updating config: top_k_tables=%d, top_k_columns=%d",
            best_tables, best_cols,
        )
        config.top_k_tables  = best_tables
        config.top_k_columns = best_cols
    else:
        logger.info(
            "Skipping sweep — using config: top_k_tables=%d, top_k_columns=%d",
            config.top_k_tables, config.top_k_columns,
        )

    if args.baseline:
        # Import baseline imports here to keep them lazy
        from baseline import (
            BaselineResult,
            build_table_graph,
            build_table_index,
            run_single_baseline,
            _save_predictions,
            _save_log,
            _save_csv,
            _print_summary,
            _run_spider_eval,
        )
        run_comparison(config, sample_ratio=args.sample)
    else:
        main(config)
