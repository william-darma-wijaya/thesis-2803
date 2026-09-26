"""
Main entry point for the GraphRAG Text-to-SQL pipeline.

Usage:
    python src/experiments/pipeline.py [--full-schema]

Flags:
                    
    "--full-schema", action="store_true",
        help="Bypass GraphRAG and feed the full DB schema to the LLM (ablation mode).",
    
    "--skip-sweep", action="store_true",
        help="Skip the precision sweep and use top_k values from config.py as-is.",
    
    "--sweep-sample", type=float, default=0.2,
        help="Fraction of dev set used for the precision sweep (default: 0.2).",
    
    "--baseline", action="store_true",
        help="Also run the table/node baseline and print a comparison report.",
    
    "--sample", type=float, default=1.0,
        help="Fraction of dev set to evaluate when --baseline is used (default: 1.0).",

    "--no-dimensions", action="store_true",
        help="With --baseline: skip the 6-dimension analysis afterwards.",

    "--ex-per-k", type=str,
        help="EX per few-shot k for Dimension 6 (0-100), e.g. 0=45.2,1=52.1,3=54.0,5=53.8.",

    "--always-dim5", action="store_true",
        help="Run Dimension 5 even without the Dimension 2 trigger.",

    "--models", nargs="+", metavar="MODEL",
        help="With --baseline: ablation stage 3 -- run the comparison once per
              model ('all' = config.py llm_model_frame), outputs per model.",

    "--tables-pct / --columns-pct / --baseline-tables-pct / --threshold /
     --few-shot-k / --llm-model",
        help="Override the config.py value (e.g. stage 1/2 winners on Kaggle).",
"""

import argparse
import csv
import gc
import json
import logging
import random
import re
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.config import PipelineConfig, add_config_override_args, apply_config_overrides
from src.experiments.sweep import run_sweep_and_get_best
from src.generation.few_shot import FewShotIndex, build_few_shot_block, build_few_shot_index
from src.generation.generation import build_prompt, generate_sql_with_token_count, load_model_and_tokenizer
from src.retrieval.baseline import table_nodes_to_schema
from src.retrieval.retrieval import (
    SchemaIndex,
    build_schema_context,
    build_schema_index,
    column_nodes_to_schema,
    evaluate_schema_linking,
    retrieve_graphrag_schema,
)
from src.core.schema import build_schema_graph, load_spider_schema
from src.metrics import esm_ex_cm
from src.utils.sql_normalize import normalize_sql_file_for_parsing
from src.metrics.sla import extract_ground_truth_schema
from src.utils.schema_utils import load_db_schema
from src.utils.sql_execution import compare_execution_results, execute_sql

# Read-only diagnostics for two documented evaluation confounds
# (context/IMPLEMENTATION_DECISIONS.md poin 5 = EX order-sensitivity, poin 11 =
# JOIN-keyword parser gap). These regexes are used ONLY for counting — they
# never rewrite any SQL that is scored.
_OUTER_JOIN_RE = re.compile(
    r"\b(?:(?:LEFT|RIGHT|FULL)(?:\s+OUTER)?|OUTER)\s+JOIN\b", re.IGNORECASE
)  # same shape as sql_normalize._EXEC_UNSAFE_JOIN_RE, kept in sync deliberately
_ORDER_BY_RE = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)

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
) -> tuple[str, float, float, str, list, int, int]:
    """
    Run the full retrieval + generation pipeline for one question.

    Args:
        schema_index    : pre-built SchemaIndex from build_schema_index()
        few_shot_index  : pre-built FewShotIndex from build_few_shot_index()
                          Pass None to run zero-shot.
    Returns:
        pred_sql     : generated SQL string
        recall       : schema-linking recall vs gold SQL
        precision    : schema-linking precision vs gold SQL
        schema_context : CREATE TABLE DDL fed to the LLM
        column_nodes : retrieved column node ids (used for raw_logs predicted_schema)
        n_in         : prompt token count (T_in) -- added for raw_logs / token
                       consumption logging, same generate_sql_with_token_count()
                       already used by ablation.py, just wired into this path too
        n_out        : generated SQL token count (T_out), raw output before
                       post-processing (see guide 1.5 poin (e))
    """
    # --- Build schema index if not supplied (fallback, not the hot path) ---
    if schema_index is None:
        schema_index = build_schema_index(graph, db_id, embed_model)

    # --- Schema retrieval (Stage 1 tables -> Stage 2 columns -> traversal) ---
    column_nodes, _ = retrieve_graphrag_schema(graph, db_id, question, schema_index, embed_model, cfg)

    if not column_nodes:
        return "SELECT 1", 0.0, 0.0, "", [], 0, 0

    # --- Schema linking evaluation ---
    recall, precision = evaluate_schema_linking(gold_sql, column_nodes, graph, db_id)

    # --- Few-shot example retrieval (same helper + k as the baseline) ---
    few_shot_block = build_few_shot_block(question, few_shot_index, embed_model, cfg)

    # --- Prompt construction & SQL generation ---
    schema_context = build_schema_context(graph, column_nodes)
    extracted_values = {
        "strings": re.findall(r"'([^']*)'", question),
        "numbers": re.findall(r"\d+", question),
    }
    prompt = build_prompt(question, schema_context, extracted_values, few_shot_block)
    pred_sql, n_out = generate_sql_with_token_count(prompt, model, tokenizer, cfg)
    n_in = len(tokenizer.encode(prompt))  # T_in -- same pattern as ablation.py's _run_k()

    return pred_sql, recall, precision, schema_context, column_nodes, n_in, n_out


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def run_official_evaluation(cfg: PipelineConfig) -> None:
    """Run the Spider official evaluation script for EM and EX metrics."""
    evaluator = Path("external/spider_eval/evaluation.py")
    if not evaluator.exists():
        logger.warning(
            "external/spider_eval/evaluation.py not found — skipping official Spider evaluation.\n"
            "Download it with:\n"
            "  mkdir -p external/spider_eval && cd external/spider_eval\n"
            "  wget https://raw.githubusercontent.com/taoyds/spider/master/evaluation.py\n"
            "  wget https://raw.githubusercontent.com/taoyds/spider/master/process_sql.py"
        )
        return

    # process_sql.py resmi SPIDER cuma mengenali join-type keyword 'join'
    # polos (JOIN_KEYWORDS di process_sql.py baris 32) -- predicted/gold SQL
    # yang pakai INNER/CROSS/LEFT/RIGHT/FULL JOIN bikin get_sql() crash di
    # dalam evaluation.py CLI juga (jalur subprocess ini pakai parser yang
    # sama dengan jalur in-process esm_ex_cm.py). Normalize dulu sebelum
    # dioper ke CLI -- lihat src/utils/sql_normalize.py dan
    # context/IMPLEMENTATION_DECISIONS.md poin 11.
    #
    # --etype match TIDAK PERNAH mengeksekusi SQL (dikonfirmasi evaluation.py
    # baris ~546: eval_exec_match cuma dipanggil kalau etype in
    # ["all","exec"]), jadi aman normalisasi PENUH (execution_safe_only=False).
    # --etype exec BENAR-BENAR mengeksekusi string dari file ini (evaluate()
    # tidak punya pemisahan raw-vs-normalized seperti jalur in-process) --
    # WAJIB execution_safe_only=True (cuma INNER/CROSS JOIN, yang 100% setara
    # hasil eksekusi), supaya LEFT/RIGHT/FULL JOIN tidak diam-diam berubah
    # semantik saat benar-benar dieksekusi. Predicted/gold LEFT/RIGHT/FULL
    # JOIN karena itu TETAP gagal parse di jalur --etype exec ini -- known
    # residual limitation, lihat IMPLEMENTATION_DECISIONS.md poin 11.
    eval_dir = cfg.predictions_file.parent

    match_gold = normalize_sql_file_for_parsing(
        cfg.gold_sql, eval_dir / "_normalized_match_gold.txt",
        has_db_id_suffix=True, execution_safe_only=False,
    )
    match_pred = normalize_sql_file_for_parsing(
        cfg.predictions_file, eval_dir / "_normalized_match_pred.txt",
        has_db_id_suffix=False, execution_safe_only=False,
    )
    exec_gold = normalize_sql_file_for_parsing(
        cfg.gold_sql, eval_dir / "_normalized_exec_gold.txt",
        has_db_id_suffix=True, execution_safe_only=True,
    )
    exec_pred = normalize_sql_file_for_parsing(
        cfg.predictions_file, eval_dir / "_normalized_exec_pred.txt",
        has_db_id_suffix=False, execution_safe_only=True,
    )

    common_args = ["--db", str(cfg.db_dir), "--table", str(cfg.tables_json)]

    print("\n" + "=" * 60)
    print("🎯 OFFICIAL SPIDER EVALUATION (Exact Match)")
    print("=" * 60)
    subprocess.run(
        ["python", "external/spider_eval/evaluation.py",
         "--gold", str(match_gold), "--pred", str(match_pred)] + common_args + ["--etype", "match"],
        check=False,
    )

    print("\n" + "=" * 60)
    print("🎯 OFFICIAL SPIDER EVALUATION (Execution Accuracy)")
    print("=" * 60)
    subprocess.run(
        ["python", "external/spider_eval/evaluation.py",
         "--gold", str(exec_gold), "--pred", str(exec_pred)] + common_args + ["--etype", "exec"],
        check=False,
    )


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
    from src.retrieval.retrieval import _parse_gold_elements, build_schema_context

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
# predicted_schema helpers for data/raw_logs/*.json (see run_comparison())
# ---------------------------------------------------------------------------

# Moved next to their retrieval code so sweep.py can score SLA on exactly the
# same "table.column" sets (sweep.py cannot import pipeline.py: circular).
# Old names kept because eval_pipeline.ipynb imports them from here.
_column_nodes_to_predicted_schema = column_nodes_to_schema
_table_nodes_to_predicted_schema = table_nodes_to_schema


# ---------------------------------------------------------------------------
# Confound diagnostics (read-only — never touch ESM/EX/CM/raw_logs values)
# ---------------------------------------------------------------------------

def _new_confound_diag() -> dict:
    return {
        "n_samples": 0,
        # Outer-join usage. Since IMPLEMENTATION_DECISIONS.md poin 11, the
        # in-process evaluator normalizes LEFT/RIGHT/FULL JOIN -> JOIN *for
        # parsing only* (raw string still executed for EX). So the old "silent
        # ESM/CM=0 sink" is gone — but a predicted outer join is now compared
        # STRUCTURALLY as if it were an inner join. Where that yields ESM=1 but
        # EX=0, the normalization may be masking a real semantic difference.
        "outer_join_gold": 0,
        "outer_join_pred_graphrag": 0,
        "outer_join_pred_baseline": 0,
        "outer_join_norm_masked_graphrag": 0,  # pred has outer join, ESM=1, EX=0
        "outer_join_norm_masked_baseline": 0,
        # EX order-sensitivity (IMPLEMENTATION_DECISIONS.md poin 5, NOT reversed).
        # Count queries where official (order-sensitive) EX = 0 but an
        # order-insensitive multiset comparison of the SAME rows matches gold.
        # Split out the subset where gold has an explicit ORDER BY (there, row
        # order legitimately matters, so those are NOT false negatives).
        "gold_unexecutable": 0,
        "ex0_orderinsensitive_match_graphrag": 0,
        "ex0_orderinsensitive_match_baseline": 0,
        "ex0_orderinsensitive_match_graphrag_gold_orderby": 0,
        "ex0_orderinsensitive_match_baseline_gold_orderby": 0,
    }


def _update_confound_diag(
    diag: dict,
    db_path: str,
    gold_sql: str,
    g_pred: str,
    g_eval,
    b_pred: str,
    b_eval,
) -> None:
    """
    Mutate `diag` with one sample's contribution. Never raises — a diagnostic
    failure must not disturb raw-log production. Executes SQL only for samples
    where at least one arm already scored EX = 0 (the only ones that can be an
    order-sensitivity artifact), so the added cost is near-zero on the majority.
    """
    try:
        diag["n_samples"] += 1

        g_outer = bool(_OUTER_JOIN_RE.search(g_pred or ""))
        b_outer = bool(_OUTER_JOIN_RE.search(b_pred or ""))
        if _OUTER_JOIN_RE.search(gold_sql or ""):
            diag["outer_join_gold"] += 1
        if g_outer:
            diag["outer_join_pred_graphrag"] += 1
            if getattr(g_eval, "esm", 0) == 1 and getattr(g_eval, "ex", 0) == 0:
                diag["outer_join_norm_masked_graphrag"] += 1
        if b_outer:
            diag["outer_join_pred_baseline"] += 1
            if getattr(b_eval, "esm", 0) == 1 and getattr(b_eval, "ex", 0) == 0:
                diag["outer_join_norm_masked_baseline"] += 1

        g_ex = getattr(g_eval, "ex", 0)
        b_ex = getattr(b_eval, "ex", 0)
        if g_ex != 0 and b_ex != 0:
            return
        gold_rows = execute_sql(db_path, gold_sql)
        if gold_rows is None:
            diag["gold_unexecutable"] += 1
            return
        gold_has_order_by = bool(_ORDER_BY_RE.search(gold_sql or ""))
        for arm, pred, ex in (("graphrag", g_pred, g_ex), ("baseline", b_pred, b_ex)):
            if ex != 0:
                continue
            pred_rows = execute_sql(db_path, pred)
            if pred_rows is None:
                continue  # genuinely broken SQL, not an order artifact
            if compare_execution_results(pred_rows, gold_rows, order_sensitive=False) == 1:
                diag[f"ex0_orderinsensitive_match_{arm}"] += 1
                if gold_has_order_by:
                    diag[f"ex0_orderinsensitive_match_{arm}_gold_orderby"] += 1
    except Exception:
        logger.exception("confound diagnostic failed (non-fatal)")


def _format_confound_diag(diag: dict) -> str:
    n = diag["n_samples"] or 1
    g_fn = diag["ex0_orderinsensitive_match_graphrag"] - diag["ex0_orderinsensitive_match_graphrag_gold_orderby"]
    b_fn = diag["ex0_orderinsensitive_match_baseline"] - diag["ex0_orderinsensitive_match_baseline_gold_orderby"]
    return "\n".join([
        "",
        "=" * 70,
        "  CONFOUND DIAGNOSTICS  (read-only — do NOT affect ESM/EX/CM/raw_logs)",
        "=" * 70,
        f"  Samples: {diag['n_samples']}",
        "",
        "  Outer JOIN usage  (LEFT/RIGHT/FULL — normalized to JOIN for parsing, poin 11)",
        f"    gold queries with outer JOIN       : {diag['outer_join_gold']:>4}  ({diag['outer_join_gold']/n*100:.1f}%)",
        f"    GraphRAG predictions               : {diag['outer_join_pred_graphrag']:>4}  "
        f"(ESM=1 & EX=0, norm may be masking a real diff: {diag['outer_join_norm_masked_graphrag']})",
        f"    Baseline predictions               : {diag['outer_join_pred_baseline']:>4}  "
        f"(ESM=1 & EX=0: {diag['outer_join_norm_masked_baseline']})",
        "",
        "  EX order-sensitivity  (official EX=0 but order-insensitive row multiset matches gold)",
        f"    GraphRAG : {diag['ex0_orderinsensitive_match_graphrag']:>4}   "
        f"(gold has ORDER BY: {diag['ex0_orderinsensitive_match_graphrag_gold_orderby']}, "
        f"likely false-negative: {g_fn})",
        f"    Baseline : {diag['ex0_orderinsensitive_match_baseline']:>4}   "
        f"(gold has ORDER BY: {diag['ex0_orderinsensitive_match_baseline_gold_orderby']}, "
        f"likely false-negative: {b_fn})",
        f"    asymmetry (GraphRAG - Baseline) in likely false-negatives : {g_fn - b_fn:+d}",
        f"    gold SQL unexecutable (excluded)   : {diag['gold_unexecutable']}",
        "=" * 70,
        "",
    ])


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
                pred_sql, recall, precision, retrieved_schema, column_nodes, _n_in, _n_out = run_single(
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
# Dimension analysis (metrics -> 6 dimensions), fed by data/raw_logs/*.json
# ---------------------------------------------------------------------------

def run_dimension_analysis(
    raw_logs_dir: Path,
    out_dir: Path,
    dev_json: Path,
    full_dev: bool,
    ex_per_k: dict | None = None,
    always_dim5: bool = False,
    qvt_dir: Path = Path("data/qvt_variations"),
) -> dict:
    """
    Run the thesis's 6 analysis dimensions over raw_logs_dir and write
    dimensions_results.json + dimensions_report.txt to out_dir.

    This is the single integration point between the generation pipeline and
    the evaluation layer (src/metrics/ + src/dimensions/); both run_comparison()
    and eval_pipeline.ipynb call it, so they cannot drift apart. It only
    orchestrates -- all logic lives in build_qvt_variations.py and
    run_all_dimensions.py (see IMPLEMENTATION_DECISIONS.md poin 21, 23, 24).

    full_dev : True only when raw_logs cover the WHOLE dev set (sample_ratio
               1.0). QVT (Dimension 4) needs it: query_id is the row position
               in dev.json, so QVT data is (re)built and Dimension 4 runs only
               then. On a subsample Dimension 4 is left out on purpose -- an
               older data/qvt_variations/ from a previous full run must not be
               silently paired with different raw_logs.
    ex_per_k : {k: EX 0-100} for Dimension 6 (not stored machine-readably by
               ablation.py, see open item D); None -> Dimension 6 is skipped
               with an explanation.
    """
    from src.experiments.build_qvt_variations import build_qvt_files
    from src.experiments.run_all_dimensions import run_all, save_dimension_outputs

    only = [1, 2, 3, 4, 5, 6] if full_dev else [1, 2, 3, 5, 6]
    if full_dev:
        with open(dev_json, "r", encoding="utf-8") as f:
            dev_data = json.load(f)
        build_qvt_files(dev_data, raw_logs_dir, qvt_dir)
    else:
        logger.info("Subsampled run -> QVT/Dimension 4 skipped (needs the full dev set).")

    results, report = run_all(
        raw_logs_dir=raw_logs_dir, qvt_dir=qvt_dir, only=only,
        ex_per_k=ex_per_k, always_dim5=always_dim5,
    )
    print(report)
    save_dimension_outputs(results, report, out_dir)
    return results


# ---------------------------------------------------------------------------
# Comparison: GraphRAG vs Baseline side-by-side
# ---------------------------------------------------------------------------

def model_run_dirs(llm_model: str, out_root: Path = Path("outputs")) -> tuple[Path, Path, Path]:
    """
    (out_dir, raw_logs_dir, qvt_dir) for one model of the stage-3 model
    comparison, e.g. Qwen/Qwen2.5-Coder-3B-Instruct ->
    outputs/models/Qwen2.5-Coder-3B-Instruct/, data/raw_logs/Qwen2.5-Coder-3B-Instruct/,
    data/qvt_variations/Qwen2.5-Coder-3B-Instruct/. Keeps models from overwriting
    each other and from the default (single-model) run's paths.
    """
    slug = llm_model.rstrip("/").split("/")[-1]
    return (
        out_root / "models" / slug,
        Path("data/raw_logs") / slug,
        Path("data/qvt_variations") / slug,
    )


def summarize_raw_logs(raw_logs: list[dict], alpha: float) -> dict:
    """EX/ESM (0-100) and mean token counts of one condition's raw_logs."""
    if not raw_logs:
        return {"n": 0}
    t_in = np.mean([r["token_input"] for r in raw_logs])
    t_out = np.mean([r["token_output"] for r in raw_logs])
    return {
        "n": len(raw_logs),
        "ex": float(np.mean([r["ex_result"] for r in raw_logs]) * 100),
        "esm": float(np.mean([r["esm_result"] for r in raw_logs]) * 100),
        "t_in": float(t_in),
        "t_out": float(t_out),
        "t": float(t_in + alpha * t_out),
    }


def run_comparison(
    cfg: PipelineConfig,
    sample_ratio: float,
    run_dimensions: bool = True,
    ex_per_k: dict | None = None,
    always_dim5: bool = False,
    out_dir: Path = Path("outputs"),
    raw_logs_dir: Path = Path("data/raw_logs"),
    qvt_dir: Path = Path("data/qvt_variations"),
) -> dict:
    """
    Run GraphRAG (column/node) AND Baseline (table/node) on the same
    dev-set sample, then print a side-by-side schema-linking + prediction
    comparison report. Both arms use the same LLM and the same few-shot k
    (cfg.few_shot_k); only the retrieval differs.

    Saved (paths relative to out_dir / raw_logs_dir; the defaults are the
    single-model run, model_run_dirs() gives the per-model ones):
        predictions/predictions.txt          — GraphRAG
        predictions/baseline_predictions.txt — Baseline
        tables/comparison_report.txt         — side-by-side summary
        raw_logs_dir/*.json                  — per-query raw logs (input of the 6 dimensions)
        tables/dimensions_{results.json,report.txt}
                                             — 6-dimension analysis (unless run_dimensions=False)

    Returns {"graphrag": summarize_raw_logs(...), "baseline": ...}.
    """
    from src.retrieval.baseline import (
        BaselineResult,
        build_table_graph,
        build_table_index,
        run_single_baseline,
        _save_log        as bl_save_log,
        _save_csv        as bl_save_csv,
        _print_summary   as bl_print_summary,
        _run_spider_eval as bl_spider_eval,
    )

    set_seed(cfg.seed)
    for sub in ("predictions", "tables", "logs"):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    # run_official_evaluation() reads cfg.predictions_file (and writes its
    # normalized copies next to it), so point it at this run's GraphRAG file.
    cfg = replace(cfg, predictions_file=out_dir / "predictions" / "predictions.txt")

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

    # Few-shot index — shared by BOTH arms, same k (cfg.few_shot_k)
    few_shot_idx = None
    if cfg.few_shot_k > 0:
        logger.info("Building few-shot index (k=%d, GraphRAG + Baseline) …", cfg.few_shot_k)
        few_shot_idx = build_few_shot_index(cfg.train_json, embed_model)

    # ── Raw-log evaluation setup (SLA/ESM/EX/CM per query -> data/raw_logs/) ──
    # This is what feeds src/dimensions/ later (see context/EVALUATION_ANALYSIS_GUIDE.md
    # Bagian 2 for the exact raw_log schema). db_schema and kmaps are each built
    # ONCE for the whole run (not per query, not per db_id) -- see the performance
    # notes on load_db_schema()/build_kmaps() docstrings for why that matters.
    # spider_schema_for_db (the parser Schema object, needs a live sqlite handle)
    # is cached lazily per db_id as we encounter it in the loop below.
    logger.info("Loading db_schema + building foreign-key maps for raw_logs evaluation …")
    db_schema = load_db_schema(str(cfg.tables_json))
    kmaps = esm_ex_cm.build_kmaps(str(cfg.tables_json))
    spider_schema_cache: dict[str, "esm_ex_cm.Schema"] = {}
    graphrag_raw_logs: list[dict] = []
    baseline_raw_logs: list[dict] = []
    confound_diag = _new_confound_diag()

    # ── Main loop — run both pipelines on the same sample ─────────────────
    graphrag_results: list[PipelineResult] = []
    baseline_results: list[BaselineResult] = []

    g_pred_path = cfg.predictions_file
    b_pred_path = out_dir / "predictions" / "baseline_predictions.txt"

    W = 72
    with open(g_pred_path, "w", encoding="utf-8") as gf, \
         open(b_pred_path, "w", encoding="utf-8") as bf:

        for i, item in enumerate(tqdm(dev_data, desc="Running both pipelines")):
            db_id    = item["db_id"]
            question = item["question"]
            gold_sql = item["query"]

            # GraphRAG
            try:
                g_pred, g_recall, g_prec, g_schema, g_column_nodes, g_n_in, g_n_out = run_single(
                    question, gold_sql, db_id,
                    col_graph, embed_model, llm, tokenizer, cfg,
                    schema_index=col_cache[db_id],
                    few_shot_index=few_shot_idx,
                )
            except Exception:
                logger.exception("GraphRAG error on sample %d", i)
                g_pred, g_recall, g_prec, g_schema = "SELECT 1", 0.0, 0.0, ""
                g_column_nodes, g_n_in, g_n_out = [], 0, 0

            # Baseline
            try:
                b_pred, b_recall, b_prec, b_table_nodes, b_n_in, b_n_out = run_single_baseline(
                    question, gold_sql, db_id,
                    tbl_graph, embed_model, llm, tokenizer, cfg,
                    table_index=tbl_cache[db_id],
                    few_shot_index=few_shot_idx,
                )
            except Exception:
                logger.exception("Baseline error on sample %d", i)
                b_pred, b_recall, b_prec = "SELECT 1", 0.0, 0.0
                b_table_nodes, b_n_in, b_n_out = [], 0, 0

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

            # ── Raw-log entry (SLA + ESM/EX/CM) for both conditions ──────────
            # Wrapped in its own try/except SEPARATE from the G/B generation
            # try/excepts above -- a failure here (parser/evaluator edge case)
            # must not throw away the predictions we already generated, it
            # should just skip logging that one query's evaluation metrics.
            try:
                db_path = str(cfg.db_dir / db_id / f"{db_id}.sqlite")
                if db_id not in spider_schema_cache:
                    spider_schema_cache[db_id] = esm_ex_cm.build_schema_for_db(db_path)
                spider_schema = spider_schema_cache[db_id]
                kmap = kmaps[db_id]

                gold_schema = sorted(extract_ground_truth_schema(gold_sql, db_id, db_schema))

                g_eval = esm_ex_cm.evaluate_single_query(g_pred, gold_sql, db_id, db_path, spider_schema, kmap)
                b_eval = esm_ex_cm.evaluate_single_query(b_pred, gold_sql, db_id, db_path, spider_schema, kmap)

                # Read-only confound instrumentation (never mutates the values
                # written to raw_logs below). Self-contained try/except inside.
                _update_confound_diag(
                    confound_diag, db_path, gold_sql, g_pred, g_eval, b_pred, b_eval,
                )

                # difficulty is a property of gold_sql alone, so g_eval and
                # b_eval always agree on it -- either works, just pick one.
                difficulty = g_eval.difficulty

                query_id = f"dev_{i + 1:04d}"
                graphrag_raw_logs.append({
                    "query_id": query_id, "db_id": db_id, "difficulty": difficulty,
                    "gold_sql": gold_sql, "predicted_sql": g_pred,
                    "gold_schema": gold_schema,
                    "predicted_schema": _column_nodes_to_predicted_schema(col_graph, g_column_nodes),
                    "token_input": g_n_in, "token_output": g_n_out,
                    "esm_result": g_eval.esm, "ex_result": g_eval.ex,
                    "cm_per_clause": g_eval.cm_per_clause,
                })
                baseline_raw_logs.append({
                    "query_id": query_id, "db_id": db_id, "difficulty": difficulty,
                    "gold_sql": gold_sql, "predicted_sql": b_pred,
                    "gold_schema": gold_schema,
                    "predicted_schema": _table_nodes_to_predicted_schema(tbl_graph, b_table_nodes),
                    "token_input": b_n_in, "token_output": b_n_out,
                    "esm_result": b_eval.esm, "ex_result": b_eval.ex,
                    "cm_per_clause": b_eval.cm_per_clause,
                })
            except Exception:
                logger.exception("Raw-log evaluation error on sample %d (db=%s)", i, db_id)

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
    bl_save_log(baseline_results, out_dir / "logs" / "baseline_log.txt")
    bl_save_csv(baseline_results, out_dir / "tables" / "baseline_results.csv")

    # ── Save raw_logs (input for src/dimensions/, see EVALUATION_ANALYSIS_GUIDE.md
    # Bagian 2 for the schema) ──────────────────────────────────────────────
    raw_logs_dir.mkdir(parents=True, exist_ok=True)
    with open(raw_logs_dir / "graphrag_log.json", "w", encoding="utf-8") as f:
        json.dump(graphrag_raw_logs, f, ensure_ascii=False, indent=2)
    with open(raw_logs_dir / "baseline_log.json", "w", encoding="utf-8") as f:
        json.dump(baseline_raw_logs, f, ensure_ascii=False, indent=2)
    logger.info(
        "Raw logs saved → %s (%d entries), %s (%d entries)",
        raw_logs_dir / "graphrag_log.json", len(graphrag_raw_logs),
        raw_logs_dir / "baseline_log.json", len(baseline_raw_logs),
    )

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
        f"  Top-k tables (pct)    {cfg.top_k_tables_pct:<16} {cfg.baseline_top_k_tables_pct}",
        f"  Top-k columns (pct)   {cfg.top_k_columns_pct:<16} -",
        f"  Few-shot k            {cfg.few_shot_k:<16} {cfg.few_shot_k}",
        f"  LLM                   {cfg.llm_model}",
        "=" * 70,
        "",
    ]
    report = "\n".join(report_lines) + _format_confound_diag(confound_diag)
    print(report)
    report_path = out_dir / "tables" / "comparison_report.txt"
    report_path.write_text(report, encoding="utf-8")
    logger.info("Comparison report → %s", report_path)
    logger.info(
        "Confound diag — outer JOIN gold=%d graphrag=%d baseline=%d | "
        "EX order-sensitivity ex0-but-set-match graphrag=%d baseline=%d",
        confound_diag["outer_join_gold"],
        confound_diag["outer_join_pred_graphrag"],
        confound_diag["outer_join_pred_baseline"],
        confound_diag["ex0_orderinsensitive_match_graphrag"],
        confound_diag["ex0_orderinsensitive_match_baseline"],
    )

    # ── Official Spider eval for both ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  SPIDER EVAL — GraphRAG (column/node)")
    print("=" * 60)
    run_official_evaluation(cfg)

    print("\n" + "=" * 60)
    print("  SPIDER EVAL — Baseline (table/node)")
    print("=" * 60)
    bl_spider_eval(b_pred_path, cfg)

    # ── 6-dimension analysis over the raw_logs just written ────────────────
    # Own try/except: hours of generation are already saved above, a failure
    # here must not look like a failed run (re-run run_all_dimensions.py alone).
    if run_dimensions:
        print("\n" + "=" * 60)
        print("  6-DIMENSION ANALYSIS (metrics -> dimensions)")
        print("=" * 60)
        try:
            run_dimension_analysis(
                raw_logs_dir=raw_logs_dir,
                out_dir=out_dir / "tables",
                dev_json=cfg.dev_json,
                full_dev=sample_ratio >= 1.0,
                ex_per_k=ex_per_k,
                always_dim5=always_dim5,
                qvt_dir=qvt_dir,
            )
        except Exception:
            logger.exception(
                "Dimension analysis failed (non-fatal, raw_logs are saved) -- "
                "re-run: python src/experiments/run_all_dimensions.py"
            )

    return {
        "graphrag": summarize_raw_logs(graphrag_raw_logs, cfg.token_output_weight),
        "baseline": summarize_raw_logs(baseline_raw_logs, cfg.token_output_weight),
    }


# ---------------------------------------------------------------------------
# Ablation stage 3: the same comparison for every model in llm_model_frame
# ---------------------------------------------------------------------------

def run_model_comparison(
    cfg: PipelineConfig,
    models: list[str],
    sample_ratio: float,
    run_dimensions: bool = True,
    ex_per_k: dict | None = None,
    always_dim5: bool = False,
) -> list[dict]:
    """
    Run run_comparison() once per model with everything else fixed (the stage
    1/2 winners in cfg), each into its own model_run_dirs(). Writes
    outputs/tables/model_comparison.csv (one row per model x mode, rewritten
    after every model so a Kaggle timeout keeps the finished ones).
    """
    rows: list[dict] = []
    summary_path = Path("outputs/tables/model_comparison.csv")
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    for model in models:
        logger.info("=" * 60)
        logger.info("MODEL %d/%d: %s", len(rows) // 2 + 1, len(models), model)
        logger.info("=" * 60)
        out_dir, raw_logs_dir, qvt_dir = model_run_dirs(model)
        summary = run_comparison(
            replace(cfg, llm_model=model), sample_ratio,
            run_dimensions=run_dimensions, ex_per_k=ex_per_k, always_dim5=always_dim5,
            out_dir=out_dir, raw_logs_dir=raw_logs_dir, qvt_dir=qvt_dir,
        )
        # run_comparison's LLM is unreachable now; free the VRAM before the next one.
        gc.collect()
        torch.cuda.empty_cache()

        for mode, s in summary.items():
            rows.append({
                "llm_model": model, "mode": mode, "n": s.get("n", 0),
                "ex": round(s.get("ex", 0.0), 2), "esm": round(s.get("esm", 0.0), 2),
                "avg_t_in": round(s.get("t_in", 0.0), 1), "avg_t_out": round(s.get("t_out", 0.0), 1),
                "avg_token_consumption": round(s.get("t", 0.0), 1),
                "few_shot_k": cfg.few_shot_k,
                "tables_pct": cfg.top_k_tables_pct if mode == "graphrag" else cfg.baseline_top_k_tables_pct,
                "columns_pct": cfg.top_k_columns_pct if mode == "graphrag" else "",
                "raw_logs": str(raw_logs_dir / f"{mode}_log.json"),
            })
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    W = 86
    print("\n" + "=" * W)
    print("  MODEL COMPARISON (ablation stage 3) — EX/ESM in-process, T = T_in + α×T_out")
    print("=" * W)
    print(f"  {'model':<36} {'mode':<9} {'EX':>7} {'ESM':>7} {'T_in':>8} {'T_out':>7} {'T':>8}")
    print("-" * W)
    for r in rows:
        print(f"  {r['llm_model'].split('/')[-1]:<36} {r['mode']:<9} {r['ex']:>6.2f}% {r['esm']:>6.2f}% "
              f"{r['avg_t_in']:>8.1f} {r['avg_t_out']:>7.1f} {r['avg_token_consumption']:>8.1f}")
    print("=" * W)
    logger.info("Model comparison → %s", summary_path)
    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphRAG Text-to-SQL Pipeline")
    parser.add_argument(
        "--full-schema", action="store_true",
        help="Bypass GraphRAG and feed the full DB schema to the LLM (ablation mode).",
    )
    parser.add_argument(
        "--skip-sweep", action="store_true",
        help="Skip the top-k sweep (ablation stage 1) and use the config.py fractions as-is.",
    )
    parser.add_argument(
        "--sweep-sample", type=float, default=0.2,
        help="Fraction of dev set used for the top-k sweep (default: 0.2).",
    )
    parser.add_argument(
        "--baseline", action="store_true",
        help="Also run the table/node baseline and print a comparison report.",
    )
    parser.add_argument(
        "--sample", type=float, default=1.0,
        help="Fraction of dev set to evaluate when --baseline is used (default: 1.0).",
    )
    parser.add_argument(
        "--no-dimensions", action="store_true",
        help="With --baseline: skip the 6-dimension analysis after the comparison run.",
    )
    parser.add_argument(
        "--ex-per-k", default=None, metavar="SPEC",
        help="EX per few-shot k for Dimension 6, scale 0-100 (e.g. 0=45.2,1=52.1,3=54.0,5=53.8).",
    )
    parser.add_argument(
        "--ex-per-k-file", default=None, metavar="JSON",
        help="Same as --ex-per-k from a file, e.g. outputs/tables/ablation_ex_per_k_graphrag.json "
             "(written by ablation.py).",
    )
    parser.add_argument(
        "--always-dim5", action="store_true",
        help="Run Dimension 5 even if Dimension 2 does not escalate to it.",
    )
    parser.add_argument(
        "--models", nargs="+", default=None, metavar="MODEL",
        help="With --baseline: ablation stage 3, one comparison per model "
             "('all' = config.py llm_model_frame). Outputs go to outputs/models/<model>/.",
    )
    add_config_override_args(parser)
    args, unknown = parser.parse_known_args()

    config = PipelineConfig(use_full_schema_bypass=args.full_schema)

    # Ablation stage 1: top-k sweep (no LLM, ranked by SLA)
    if not args.skip_sweep and not config.use_full_schema_bypass:
        logger.info("=" * 60)
        logger.info("STEP 0: top-k sweep (pass --skip-sweep to skip)")
        logger.info("=" * 60)
        best = run_sweep_and_get_best(sample_ratio=args.sweep_sample, cfg=config)
        logger.info("Sweep done — updating config: %s", best)
        for attr, value in best.items():
            setattr(config, attr, value)

    # Explicit CLI values win over both config.py and the sweep.
    apply_config_overrides(config, args)
    logger.info(
        "Config: tables_pct=%.2f columns_pct=%.2f threshold=%.2f baseline_tables_pct=%.2f "
        "few_shot_k=%d (both arms) llm=%s",
        config.top_k_tables_pct, config.top_k_columns_pct, config.semantic_similarity_threshold,
        config.baseline_top_k_tables_pct, config.few_shot_k, config.llm_model,
    )

    if args.models and not args.baseline:
        parser.error("--models needs --baseline (stage 3 compares both arms per model).")

    if args.baseline:
        from src.experiments.run_all_dimensions import parse_ex_per_k
        try:
            ex_per_k = parse_ex_per_k(args.ex_per_k, Path(args.ex_per_k_file) if args.ex_per_k_file else None)
        except ValueError as exc:
            parser.error(str(exc))
        if args.models:
            models = config.llm_model_frame if args.models == ["all"] else args.models
            run_model_comparison(
                config, models, sample_ratio=args.sample,
                run_dimensions=not args.no_dimensions,
                ex_per_k=ex_per_k, always_dim5=args.always_dim5,
            )
        else:
            run_comparison(
                config, sample_ratio=args.sample,
                run_dimensions=not args.no_dimensions,
                ex_per_k=ex_per_k, always_dim5=args.always_dim5,
            )
    else:
        main(config)
