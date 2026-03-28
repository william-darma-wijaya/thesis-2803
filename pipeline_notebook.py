# =============================================================================
# GraphRAG Text-to-SQL Pipeline — Notebook Version
#
# Run each cell in order. Heavy cells (LLM loading, main loop) only need to
# be re-run when you change something relevant. Everything else can be
# re-run cheaply to tweak config or inspect results.
# =============================================================================


# ── Cell 1 ── Imports & logging ───────────────────────────────────────────────
import json
import logging
import random
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import PipelineConfig
from few_shot import (
    FewShotIndex,
    build_few_shot_index,
    format_few_shot_block,
    retrieve_few_shot_examples,
)
from generation import build_prompt, generate_sql, load_model_and_tokenizer
from retrieval import (
    SchemaIndex,
    build_schema_context,
    build_schema_index,
    evaluate_schema_linking,
    prune_path_nodes,
    semantic_schema_linking,
    trace_schema_paths,
    _parse_gold_elements,
)
from schema import build_schema_graph, load_spider_schema
from sweep import run_sweep_and_get_best

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
print("✓ imports done")


# ── Cell 2 ── Config ──────────────────────────────────────────────────────────
# Tweak any settings here before running the rest.
# Re-run this cell whenever you want to change a config value — no need to
# reload models.

config = PipelineConfig()

# Override specific values if needed, e.g.:
# config.few_shot_k        = 3
# config.top_k_tables      = 2
# config.top_k_columns     = 2
# config.use_full_schema_bypass = False   # set True for full-schema ablation

print(f"  data_path        : {config.data_path}")
print(f"  embedding_model  : {config.embedding_model}")
print(f"  llm_model        : {config.llm_model}")
print(f"  few_shot_k       : {config.few_shot_k}")
print(f"  top_k_tables     : {config.top_k_tables}")
print(f"  top_k_columns    : {config.top_k_columns}")
print(f"  full_schema_bypass: {config.use_full_schema_bypass}")


# ── Cell 3 ── Seed ────────────────────────────────────────────────────────────
# Sets all random seeds for reproducibility.
# Re-run if you change config.seed.

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(config.seed)
print(f"✓ seed set to {config.seed}")


# ── Cell 4 ── Load schema & build graph ───────────────────────────────────────
# Parses tables.json into a NetworkX graph where nodes = columns,
# edges = PK/FK relationships.
# Only needs re-running if tables.json changes (it won't during a run).

schema_df = load_spider_schema(config.tables_json)
graph = build_schema_graph(schema_df)
print(f"✓ graph built: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")


# ── Cell 5 ── Load dev set ────────────────────────────────────────────────────
# Loads the 1034 dev questions from dev.json.

with open(config.dev_json, "r", encoding="utf-8") as f:
    dev_data = json.load(f)

print(f"✓ dev set loaded: {len(dev_data)} questions")


# ── Cell 6 ── Load embedding model ────────────────────────────────────────────
# BGE-M3 — used for schema linking AND few-shot retrieval.
# Loads once and is reused everywhere. No GPU required for this model.
# Re-running this cell reloads the model from disk (~1–2 min).

embed_model = SentenceTransformer(config.embedding_model)
print(f"✓ embedding model loaded: {config.embedding_model}")


# ── Cell 7 ── Precision sweep ─────────────────────────────────────────────────
# Runs schema-linking ONLY (no LLM) across a grid of top_k_tables × top_k_columns
# on 20% of the dev set. Finds the best config by recall-weighted score and
# updates config in-place.
#
# Skip this cell if you already know your best config — just set config values
# in Cell 2 and move on.
#
# Runtime: ~5–10 min (embedding only, no GPU needed)

if not config.use_full_schema_bypass:
    best_tables, best_cols = run_sweep_and_get_best(
        sample_ratio=0.2,
        cfg=config,
    )
    config.top_k_tables  = best_tables
    config.top_k_columns = best_cols
    print(f"\n✓ config updated → top_k_tables={config.top_k_tables}, top_k_columns={config.top_k_columns}")
else:
    print("⚡ full-schema bypass mode — sweep skipped")


# ── Cell 8 ── Build schema index cache ────────────────────────────────────────
# Pre-computes table + column embeddings for every database in the dev set.
# Stored in schema_cache dict — reused for every question in the main loop.
# Must be re-run if embed_model or config.top_k_* changes.
#
# Runtime: ~3–5 min

unique_db_ids = list(dict.fromkeys(item["db_id"] for item in dev_data))
schema_cache: dict[str, SchemaIndex] = {
    db_id: build_schema_index(graph, db_id, embed_model)
    for db_id in tqdm(unique_db_ids, desc="Indexing schemas")
}
print(f"✓ schema index built for {len(schema_cache)} databases")


# ── Cell 9 ── Load LLM ────────────────────────────────────────────────────────
# Loads Qwen2.5-Coder-7B-Instruct with 4-bit NF4 quantization.
# This is the heaviest cell — takes 5–10 min and uses most of your GPU memory.
# Only re-run this if you want to switch models.
#
# Runtime: ~5–10 min | VRAM: ~6–8 GB

llm, tokenizer = load_model_and_tokenizer(config)
print(f"✓ LLM loaded: {config.llm_model}")


# ── Cell 10 ── Build few-shot index ───────────────────────────────────────────
# Encodes all ~8,659 training questions with BGE-M3.
# Used at inference time to find the top-k most similar training examples.
# Set config.few_shot_k = 0 to skip this cell and run zero-shot.
#
# Runtime: ~3–5 min

if config.few_shot_k > 0:
    few_shot_idx = build_few_shot_index(config.train_json, embed_model)
    print(f"✓ few-shot index built ({len(few_shot_idx.examples)} training examples)")
else:
    few_shot_idx = None
    print("⚡ few-shot disabled (few_shot_k=0) — running zero-shot")


# ── Cell 11 ── Helper functions ───────────────────────────────────────────────
# Display helpers and gold schema builder.
# Re-run if you want to change the print format.

def _bar(value: float, width: int = 20) -> str:
    """ASCII progress bar for a 0–1 float."""
    filled = round(value * width)
    return "[" + "#" * filled + "." * (width - filled) + "]"


def build_gold_schema_context(gold_sql: str, graph, db_id: str) -> str:
    """
    Reconstruct a CREATE TABLE DDL from the tables/columns the gold SQL references.
    Used purely for display — lets you compare retrieved vs required schema.
    """
    gold_elements = _parse_gold_elements(gold_sql, graph, db_id)
    gold_nodes = [
        n for n, d in graph.nodes(data=True)
        if d.get("database") == db_id
        and d.get("type") == "column"
        and (d["table"].lower() in gold_elements or d["column"].lower() in gold_elements)
    ]
    if not gold_nodes:
        return "(could not parse gold schema)"
    return build_schema_context(graph, gold_nodes)


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
    """Run the full retrieval + generation pipeline for one question."""
    if schema_index is None:
        schema_index = build_schema_index(graph, db_id, embed_model)

    # --- Schema retrieval ---
    if cfg.use_full_schema_bypass:
        column_nodes = [
            n for n, d in graph.nodes(data=True)
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
                n for n, d in graph.nodes(data=True)
                if d.get("database") == db_id and d.get("type") == "column"
            ]
        else:
            c_nodes, paths, _ = trace_schema_paths(graph, db_id, detected_cols)
            all_path_nodes = list(set(c_nodes + [node for path in paths for node in path]))
            detected_col_names = {c.lower() for c in detected_cols}
            column_nodes = prune_path_nodes(graph, detected_col_names, all_path_nodes)
            if not column_nodes:
                column_nodes = all_path_nodes

    if not column_nodes:
        return "SELECT 1", 0.0, 0.0, "", []

    recall, precision = evaluate_schema_linking(gold_sql, column_nodes, graph, db_id)

    few_shot_block = ""
    if few_shot_index is not None and cfg.few_shot_k > 0:
        examples = retrieve_few_shot_examples(question, db_id, few_shot_index, embed_model, cfg)
        few_shot_block = format_few_shot_block(examples)

    schema_context = build_schema_context(graph, column_nodes)
    extracted_values = {
        "strings": re.findall(r"'([^']*)'", question),
        "numbers": re.findall(r"\d+", question),
    }
    prompt = build_prompt(question, schema_context, extracted_values, few_shot_block)
    pred_sql = generate_sql(prompt, model, tokenizer, cfg)

    return pred_sql, recall, precision, schema_context, column_nodes


print("✓ helper functions defined")


# ── Cell 12 ── Main loop ──────────────────────────────────────────────────────
# Runs all 1034 questions through the pipeline.
# Saves predictions to predictions.txt (one SQL per line, Spider format).
# Prints a rich progress block per question.
#
# Runtime: ~2–4 hours on T4 GPU
# TIP: If you need to resume after a crash, see Cell 13 for partial-resume logic.

@dataclass
class PipelineResult:
    index: int
    db_id: str
    question: str
    gold_sql: str
    pred_sql: str
    recall: float
    precision: float
    retrieved_schema: str = ""
    gold_schema: str = ""


results: list[PipelineResult] = []
W = 72

with open(config.predictions_file, "w", encoding="utf-8") as pred_file:
    for i, item in enumerate(tqdm(dev_data, desc="Generating SQL")):
        db_id    = item["db_id"]
        question = item["question"]
        gold_sql = item["query"]

        try:
            pred_sql, recall, precision, retrieved_schema, column_nodes = run_single(
                question, gold_sql, db_id, graph, embed_model, llm, tokenizer, config,
                schema_index=schema_cache[db_id],
                few_shot_index=few_shot_idx,
            )
            gold_schema = build_gold_schema_context(gold_sql, graph, db_id)
        except Exception as e:
            logger.exception("Error on sample %d (db=%s): %s", i, db_id, e)
            pred_sql, recall, precision = "SELECT 1", 0.0, 0.0
            retrieved_schema, gold_schema, column_nodes = "", "", []

        results.append(PipelineResult(
            index=i + 1,
            db_id=db_id,
            question=question,
            gold_sql=gold_sql,
            pred_sql=pred_sql,
            recall=recall,
            precision=precision,
            retrieved_schema=retrieved_schema,
            gold_schema=gold_schema,
        ))
        pred_file.write(f"{pred_sql}\n")

        # Rich per-question print
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

print(f"\n✓ predictions saved to {config.predictions_file}")


# ── Cell 13 ── Schema linking summary ────────────────────────────────────────
# Prints average recall and precision across all questions.
# Run after Cell 12 finishes (or after a partial run to see progress).

valid = [r for r in results if r.recall >= 0]
avg_recall    = sum(r.recall    for r in valid) / len(valid) * 100
avg_precision = sum(r.precision for r in valid) / len(valid) * 100

print("\n" + "=" * 60)
print("📈 SCHEMA LINKING SUMMARY (GraphRAG)")
print("=" * 60)
print(f"  Recall    : {avg_recall:.2f}%  — how many gold elements were retrieved")
print(f"  Precision : {avg_precision:.2f}%  — fraction of retrieved elements that were relevant")
print(f"  Samples   : {len(valid)}")


# ── Cell 14 ── Official Spider evaluation (EM + EX) ───────────────────────────
# Calls evaluation.py to compute official Exact Match and Execution Accuracy.
# Requires evaluation.py and process_sql.py in the working directory.
#
# If missing, download with:
#   !wget https://raw.githubusercontent.com/taoyds/spider/master/evaluation.py
#   !wget https://raw.githubusercontent.com/taoyds/spider/master/process_sql.py

evaluator = Path("evaluation.py")
if not evaluator.exists():
    print(
        "⚠️  evaluation.py not found.\n"
        "Download it with:\n"
        "  !wget https://raw.githubusercontent.com/taoyds/spider/master/evaluation.py\n"
        "  !wget https://raw.githubusercontent.com/taoyds/spider/master/process_sql.py"
    )
else:
    common_args = [
        "--gold",  str(config.gold_sql),
        "--pred",  str(config.predictions_file),
        "--db",    str(config.db_dir),
        "--table", str(config.tables_json),
    ]

    print("\n" + "=" * 60)
    print("🎯 OFFICIAL SPIDER EVALUATION (Exact Match)")
    print("=" * 60)
    subprocess.run(["python", "evaluation.py"] + common_args + ["--etype", "match"], check=False)

    print("\n" + "=" * 60)
    print("🎯 OFFICIAL SPIDER EVALUATION (Execution Accuracy)")
    print("=" * 60)
    subprocess.run(["python", "evaluation.py"] + common_args + ["--etype", "exec"], check=False)
