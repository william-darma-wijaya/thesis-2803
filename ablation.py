"""
Ablation study: few-shot k=0 vs k=1 vs k=3 vs k=5

Runs the FULL pipeline (LLM inference included) for each k value on a
configurable subset of the dev set, then prints a comparison table and
saves predictions for each k so you can run the official Spider evaluator
on each one independently.

Usage:
    python ablation.py --mode graphrag --k-values 0 1 3 5 --sample 1.0
    python ablation.py --mode baseline --k-values 0 1 3 5 --sample 1.0
    python ablation.py                        # default: graphrag, 20% dev set, k in [0,1,3,5]

Outputs (one set per mode × k):
    ablation_graphrag_predictions_k{k}.txt   — GraphRAG predictions
    ablation_baseline_predictions_k{k}.txt   — Baseline predictions
    ablation_results.csv                     — recall/precision per mode × k
"""

import argparse
import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import PipelineConfig
from few_shot import FewShotIndex, build_few_shot_index, format_few_shot_block, retrieve_few_shot_examples
from generation import build_prompt, generate_sql, load_model_and_tokenizer
from retrieval import (
    build_schema_context,
    build_schema_index,
    evaluate_schema_linking,
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

DEFAULT_K_VALUES = [0, 1, 3, 5]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    k: int
    mode: str
    avg_recall:            float
    avg_precision:         float
    avg_prompt_tokens:     float   # T_in  — input (prompt) tokens
    avg_output_tokens:     float   # T_out — output (generated SQL) tokens
    avg_token_consumption: float   # T = T_in + α × T_out  (the TEP metric)
    n_samples:             int
    predictions_file:      Path
    prompts_file:          Path


# ---------------------------------------------------------------------------
# Single k run
# ---------------------------------------------------------------------------

def _run_k(
    k: int,
    dev_subset: list[dict],
    graph,
    cache: dict,
    few_shot_index: FewShotIndex | None,
    embed_model: SentenceTransformer,
    llm,
    tokenizer,
    base_cfg: PipelineConfig,
    output_dir: Path,
    mode: str = "graphrag",
) -> AblationResult:
    """
    Run full pipeline for one k value and save predictions.

    Args:
        cache : SchemaIndex dict for mode="graphrag",
                TableSchemaIndex dict for mode="baseline".
        mode  : "graphrag" (column-level retrieval) or
                "baseline" (table-level retrieval).
    """
    from baseline import (
        semantic_linking_table_level,
        trace_table_paths,
        build_table_schema_context,
        evaluate_table_linking,
    )

    cfg = PipelineConfig(
        data_path=base_cfg.data_path,
        top_k_tables=base_cfg.top_k_tables,
        top_k_columns=base_cfg.top_k_columns,
        few_shot_k=k,
        few_shot_same_db_first=base_cfg.few_shot_same_db_first,
        use_full_schema_bypass=False,
    )

    pred_path   = output_dir / f"ablation_{mode}_predictions_k{k}.txt"
    prompt_path = output_dir / f"ablation_{mode}_prompts_k{k}.jsonl"
    recalls, precisions = [], []
    prompt_tokens_list, output_tokens_list, consumption_list = [], [], []

    alpha = base_cfg.token_output_weight  # T = T_in + α × T_out

    with open(pred_path, "w", encoding="utf-8") as pred_file, \
         open(prompt_path, "w", encoding="utf-8") as prompt_file:
        for idx, item in enumerate(tqdm(dev_subset, desc=f"{mode} k={k}", leave=False)):
            db_id    = item["db_id"]
            question = item["question"]
            gold_sql = item["query"]

            try:
                if mode == "graphrag":
                    index = cache[db_id]
                    detected_cols = semantic_schema_linking(
                        graph, db_id, question,
                        all_columns=index.columns,
                        col_embeddings=index.col_embeddings,
                        embed_model=embed_model,
                        cfg=cfg,
                        index=index,
                    )
                    if not detected_cols:
                        column_nodes = [
                            n for n, d in graph.nodes(data=True)
                            if d.get("database") == db_id and d.get("type") == "column"
                        ]
                    else:
                        c_nodes, paths, _ = trace_schema_paths(graph, db_id, detected_cols)
                        column_nodes = list(
                            set(c_nodes + [node for path in paths for node in path])
                        )
                    r, p = evaluate_schema_linking(gold_sql, column_nodes, graph, db_id)
                    schema_context = build_schema_context(graph, column_nodes)

                else:  # baseline
                    table_index = cache[db_id]
                    detected = semantic_linking_table_level(
                        table_index, question, embed_model,
                        top_k=cfg.top_k_tables if cfg.top_k_tables > 0 else 3,
                    )
                    if not detected:
                        table_nodes = [
                            n for n, d in graph.nodes(data=True)
                            if d.get("database") == db_id and d.get("type") == "table"
                        ]
                    else:
                        table_nodes = trace_table_paths(graph, detected)
                    r, p = evaluate_table_linking(gold_sql, table_nodes, graph, db_id)
                    schema_context = build_table_schema_context(graph, table_nodes)

                recalls.append(r)
                precisions.append(p)

                # --- Few-shot block ---
                few_shot_block = ""
                if few_shot_index is not None and k > 0:
                    examples = retrieve_few_shot_examples(
                        question, db_id, few_shot_index, embed_model, cfg
                    )
                    few_shot_block = format_few_shot_block(examples)

                # --- Prompt, generate, count tokens ---
                extracted_values = {
                    "strings": re.findall(r"'([^']*)'", question),
                    "numbers": re.findall(r"\d+", question),
                }
                prompt   = build_prompt(question, schema_context, extracted_values, few_shot_block)
                pred_sql = generate_sql(prompt, llm, tokenizer, cfg)

                n_in  = len(tokenizer.encode(prompt))
                n_out = len(tokenizer.encode(pred_sql))
                n_t   = int(n_in + alpha * n_out)

                prompt_tokens_list.append(n_in)
                output_tokens_list.append(n_out)
                consumption_list.append(n_t)

                prompt_file.write(
                    json.dumps({
                        "i": idx + 1, "db_id": db_id, "question": question,
                        "tokens_in": n_in, "tokens_out": n_out,
                        "token_consumption": n_t,
                        "prompt": prompt, "pred_sql": pred_sql,
                    }, ensure_ascii=False) + "\n"
                )
                logger.info(
                    "  [%d/%d] db=%-20s T=%4d (in=%d out=%d) | R=%.1f%% P=%.1f%%",
                    idx + 1, len(dev_subset), db_id, n_t, n_in, n_out,
                    recalls[-1] * 100, precisions[-1] * 100,
                )

            except Exception:
                logger.exception("Error on question '%s'", question[:60])
                pred_sql = "SELECT 1"
                recalls.append(0.0)
                precisions.append(0.0)
                prompt_tokens_list.append(0)
                output_tokens_list.append(0)
                consumption_list.append(0)

            pred_file.write(f"{pred_sql}\n")

    def _safe_mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    return AblationResult(
        k=k,
        mode=mode,
        avg_recall=float(np.mean(recalls)),
        avg_precision=float(np.mean(precisions)),
        avg_prompt_tokens=_safe_mean(prompt_tokens_list),
        avg_output_tokens=_safe_mean(output_tokens_list),
        avg_token_consumption=_safe_mean(consumption_list),
        n_samples=len(dev_subset),
        predictions_file=pred_path,
        prompts_file=prompt_path,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_table(results: list[AblationResult], cfg: "PipelineConfig | None" = None) -> None:
    alpha = cfg.token_output_weight if cfg else 1.0
    W = 100
    print("\n" + "=" * W)
    print("FEW-SHOT ABLATION RESULTS")
    print(f"  Token Consumption T = T_in + {alpha}×T_out")
    print("=" * W)
    print(f"{'mode':<12} {'k':>5} {'recall':>9} {'precision':>11} {'T_consume':>11} {'T_in':>8} {'T_out':>7}  predictions file")
    print("-" * W)
    for r in sorted(results, key=lambda x: (x.mode, x.k)):
        print(
            f"{r.mode:<12} {r.k:>5} {r.avg_recall*100:>8.2f}% "
            f"{r.avg_precision*100:>10.2f}% {r.avg_token_consumption:>11.1f} "
            f"{r.avg_prompt_tokens:>8.1f} {r.avg_output_tokens:>7.1f}  {r.predictions_file.name}"
        )
    print("=" * W)

    gold   = str(cfg.gold_sql)   if cfg else "<gold_path>"
    db_dir = str(cfg.db_dir)     if cfg else "<db_dir>"
    tables = str(cfg.tables_json) if cfg else "<tables_json>"

    modes = list(dict.fromkeys(r.mode for r in results))
    print("\nRun Spider official evaluation for each k / mode:")
    print("  (or pass --run-eval to ablation.py to run automatically)\n")
    for mode in modes:
        for r in sorted((x for x in results if x.mode == mode), key=lambda x: x.k):
            for etype in ["match", "exec"]:
                print(
                    f"  python evaluation.py --gold {gold} "
                    f"--pred {r.predictions_file} "
                    f"--db {db_dir} --table {tables} --etype {etype}"
                )


def _run_ablation_evals(results: list[AblationResult], cfg: PipelineConfig) -> None:
    """Run Spider official evaluation (EM + EX) for every ablation prediction file."""
    import subprocess
    evaluator = Path("evaluation.py")
    if not evaluator.exists():
        logger.warning(
            "evaluation.py not found — cannot auto-run Spider eval.\n"
            "Download it first:\n"
            "  wget https://raw.githubusercontent.com/taoyds/spider/master/evaluation.py\n"
            "  wget https://raw.githubusercontent.com/taoyds/spider/master/process_sql.py"
        )
        return

    base_args = [
        "--gold",  str(cfg.gold_sql),
        "--db",    str(cfg.db_dir),
        "--table", str(cfg.tables_json),
    ]
    for r in sorted(results, key=lambda x: (x.mode, x.k)):
        for etype in ["match", "exec"]:
            label = "Exact Match" if etype == "match" else "Execution Accuracy"
            print("\n" + "=" * 70)
            print(f"  SPIDER EVAL — mode={r.mode}  k={r.k}  {label}")
            print("=" * 70)
            subprocess.run(
                ["python", "evaluation.py"] + base_args +
                ["--pred", str(r.predictions_file), "--etype", etype],
                check=False,
            )


def _save_csv(results: list[AblationResult], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=[
                "mode", "k", "avg_recall", "avg_precision",
                "avg_prompt_tokens", "avg_output_tokens", "avg_token_consumption",
                "n_samples", "predictions_file", "prompts_file",
            ]
        )
        writer.writeheader()
        for r in results:
            writer.writerow({
                "mode":                  r.mode,
                "k":                     r.k,
                "avg_recall":            round(r.avg_recall, 4),
                "avg_precision":         round(r.avg_precision, 4),
                "avg_prompt_tokens":     round(r.avg_prompt_tokens, 1),
                "avg_output_tokens":     round(r.avg_output_tokens, 1),
                "avg_token_consumption": round(r.avg_token_consumption, 1),
                "n_samples":             r.n_samples,
                "predictions_file":      str(r.predictions_file),
                "prompts_file":          str(r.prompts_file),
            })
    logger.info("CSV saved to %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(sample_ratio: float, k_values: list[int], mode: str = "graphrag", run_eval: bool = False) -> None:
    cfg = PipelineConfig()
    output_dir = Path(".")

    # Warn if config still has default top_k values (sweep not yet run)
    if cfg.top_k_tables == 3 and cfg.top_k_columns == 5:
        logger.warning(
            "top_k_tables=3 top_k_columns=5 (config.py defaults). "
            "Run sweep.py first and update config.py if sweep results differ."
        )

    logger.info("Loading schema …")
    schema_df = load_spider_schema(cfg.tables_json)

    logger.info("Loading dev set …")
    with open(cfg.dev_json, "r", encoding="utf-8") as f:
        dev_data = json.load(f)

    n = max(1, int(len(dev_data) * sample_ratio))
    rng = np.random.default_rng(42)
    indices = rng.choice(len(dev_data), size=n, replace=False)
    dev_subset = [dev_data[i] for i in sorted(indices)]
    logger.info("Ablation subset: %d / %d questions (%.0f%%)", n, len(dev_data), sample_ratio * 100)

    logger.info("Loading embedding model: %s", cfg.embedding_model)
    embed_model = SentenceTransformer(cfg.embedding_model)

    logger.info("Loading LLM: %s …", cfg.llm_model)
    llm, tokenizer = load_model_and_tokenizer(cfg)

    unique_db_ids = list(dict.fromkeys(item["db_id"] for item in dev_subset))

    if mode == "graphrag":
        logger.info("Building column-level graph (GraphRAG) …")
        graph = build_schema_graph(schema_df)
        logger.info("Building schema indices for %d databases …", len(unique_db_ids))
        cache = {
            db_id: build_schema_index(graph, db_id, embed_model)
            for db_id in tqdm(unique_db_ids, desc="Indexing columns")
        }
    else:  # baseline
        from baseline import build_table_graph, build_table_index
        logger.info("Building table-level graph (Baseline) …")
        graph = build_table_graph(schema_df)
        logger.info("Building table indices for %d databases …", len(unique_db_ids))
        cache = {
            db_id: build_table_index(graph, db_id, embed_model)
            for db_id in tqdm(unique_db_ids, desc="Indexing tables")
        }

    # Few-shot index (needed for any k > 0)
    few_shot_index: FewShotIndex | None = None
    if any(k > 0 for k in k_values):
        logger.info("Building few-shot index from training set …")
        few_shot_index = build_few_shot_index(cfg.train_json, embed_model)

    # Run each k
    results: list[AblationResult] = []
    for k in k_values:
        logger.info("=" * 50)
        logger.info("Running ablation: mode=%s k=%d", mode, k)
        result = _run_k(
            k, dev_subset, graph, cache,
            few_shot_index if k > 0 else None,
            embed_model, llm, tokenizer, cfg, output_dir,
            mode=mode,
        )
        results.append(result)
        logger.info(
            "mode=%s k=%d done → recall=%.2f%% precision=%.2f%%",
            mode, k, result.avg_recall * 100, result.avg_precision * 100,
        )

    _print_table(results, cfg)
    _save_csv(results, Path("ablation_results.csv"))

    if run_eval:
        logger.info("Auto-running Spider evaluation for all prediction files …")
        _run_ablation_evals(results, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Few-shot k ablation study")
    parser.add_argument(
        "--mode", choices=["graphrag", "baseline"], default="graphrag",
        help="Retrieval mode to ablate (default: graphrag).",
    )
    parser.add_argument(
        "--sample", type=float, default=0.2,
        help="Fraction of dev set to use (default: 0.2)",
    )
    parser.add_argument(
        "--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES,
        help="List of k values to test (default: 0 1 3 5)",
    )
    parser.add_argument(
        "--run-eval", action="store_true",
        help="Auto-run Spider official evaluation (EM + EX) for every prediction file after inference.",
    )
    args, unknown = parser.parse_known_args()
    main(args.sample, args.k_values, mode=args.mode, run_eval=args.run_eval)
