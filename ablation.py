"""
Ablation study: few-shot k=0 vs k=1 vs k=3 vs k=5

Runs the FULL pipeline (LLM inference included) for each k value on a
configurable subset of the dev set, then prints a comparison table and
saves predictions for each k so you can run the official Spider evaluator
on each one independently.

Usage:
    python ablation.py                        # default: 20% dev set, k in [0,1,3,5]
    python ablation.py --sample 0.5           # 50% dev set
    python ablation.py --k-values 0 1 3       # custom k list
    python ablation.py --sample 1.0           # full dev set (slow)

Outputs (one set per k value):
    ablation_predictions_k{k}.txt   — predictions in Spider format
    ablation_results.csv            — recall/precision per k
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
    SchemaIndex,
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
    avg_recall:    float
    avg_precision: float
    n_samples:     int
    predictions_file: Path


# ---------------------------------------------------------------------------
# Single k run
# ---------------------------------------------------------------------------

def _run_k(
    k: int,
    dev_subset: list[dict],
    graph,
    schema_cache: dict[str, SchemaIndex],
    few_shot_index: FewShotIndex | None,
    embed_model: SentenceTransformer,
    llm,
    tokenizer,
    base_cfg: PipelineConfig,
    output_dir: Path,
) -> AblationResult:
    """Run full pipeline for one k value and save predictions."""
    cfg = PipelineConfig(
        data_path=base_cfg.data_path,
        top_k_tables=base_cfg.top_k_tables,
        top_k_columns=base_cfg.top_k_columns,
        few_shot_k=k,
        few_shot_same_db_first=base_cfg.few_shot_same_db_first,
        use_full_schema_bypass=False,
    )

    pred_path = output_dir / f"ablation_predictions_k{k}.txt"
    recalls, precisions = [], []

    with open(pred_path, "w", encoding="utf-8") as pred_file:
        for item in tqdm(dev_subset, desc=f"k={k}", leave=False):
            db_id    = item["db_id"]
            question = item["question"]
            gold_sql = item["query"]
            index    = schema_cache[db_id]

            try:
                # --- Schema retrieval ---
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
                recalls.append(r)
                precisions.append(p)

                # --- Few-shot block ---
                few_shot_block = ""
                if few_shot_index is not None and k > 0:
                    examples = retrieve_few_shot_examples(
                        question, db_id, few_shot_index, embed_model, cfg
                    )
                    few_shot_block = format_few_shot_block(examples)

                # --- Prompt & generate ---
                schema_context = build_schema_context(graph, column_nodes)
                extracted_values = {
                    "strings": re.findall(r"'([^']*)'", question),
                    "numbers": re.findall(r"\d+", question),
                }
                prompt    = build_prompt(question, schema_context, extracted_values, few_shot_block)
                pred_sql  = generate_sql(prompt, llm, tokenizer, cfg)

            except Exception:
                logger.exception("Error on question '%s'", question[:60])
                pred_sql = "SELECT 1"
                recalls.append(0.0)
                precisions.append(0.0)

            pred_file.write(f"{pred_sql}\n")

    return AblationResult(
        k=k,
        avg_recall=float(np.mean(recalls)),
        avg_precision=float(np.mean(precisions)),
        n_samples=len(dev_subset),
        predictions_file=pred_path,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_table(results: list[AblationResult]) -> None:
    print("\n" + "=" * 60)
    print("FEW-SHOT ABLATION RESULTS")
    print("=" * 60)
    print(f"{'k':>5} {'recall':>9} {'precision':>11}  predictions file")
    print("-" * 60)
    for r in sorted(results, key=lambda x: x.k):
        print(
            f"{r.k:>5} {r.avg_recall*100:>8.2f}% "
            f"{r.avg_precision*100:>10.2f}%  {r.predictions_file.name}"
        )
    print("=" * 60)
    print(
        "\nRun Spider official evaluation for each k:\n"
        "  python evaluation.py --gold <gold_path> "
        "--pred ablation_predictions_k0.txt --db <db_dir> "
        "--table <tables_json> --etype exec\n"
        "  (repeat for k1, k3, k5)"
    )


def _save_csv(results: list[AblationResult], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["k", "avg_recall", "avg_precision", "n_samples", "predictions_file"]
        )
        writer.writeheader()
        for r in results:
            writer.writerow({
                "k":                r.k,
                "avg_recall":       round(r.avg_recall, 4),
                "avg_precision":    round(r.avg_precision, 4),
                "n_samples":        r.n_samples,
                "predictions_file": str(r.predictions_file),
            })
    logger.info("CSV saved to %s", path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(sample_ratio: float, k_values: list[int]) -> None:
    cfg = PipelineConfig()
    output_dir = Path(".")

    logger.info("Loading schema and building graph …")
    schema_df = load_spider_schema(cfg.tables_json)
    graph     = build_schema_graph(schema_df)

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

    # Schema index cache
    unique_db_ids = list(dict.fromkeys(item["db_id"] for item in dev_subset))
    logger.info("Building schema indices for %d databases …", len(unique_db_ids))
    schema_cache: dict[str, SchemaIndex] = {
        db_id: build_schema_index(graph, db_id, embed_model)
        for db_id in tqdm(unique_db_ids, desc="Indexing")
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
        logger.info("Running ablation: k=%d", k)
        result = _run_k(
            k, dev_subset, graph, schema_cache,
            few_shot_index if k > 0 else None,
            embed_model, llm, tokenizer, cfg, output_dir,
        )
        results.append(result)
        logger.info(
            "k=%d done → recall=%.2f%% precision=%.2f%%",
            k, result.avg_recall * 100, result.avg_precision * 100,
        )

    _print_table(results)
    _save_csv(results, Path("ablation_results.csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Few-shot k ablation study")
    parser.add_argument(
        "--sample", type=float, default=0.2,
        help="Fraction of dev set to use (default: 0.2)",
    )
    parser.add_argument(
        "--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES,
        help="List of k values to test (default: 0 1 3 5)",
    )
    args, unknown = parser.parse_known_args()
    main(args.sample, args.k_values)
