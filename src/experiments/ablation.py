"""
Ablation stage 2 — few-shot k, ranked by EX.

Runs the FULL pipeline (LLM inference included) for every k in --k-values
(default: config.py's few_shot_frame) for ONE mode, with the stage-1 top-k
fractions from config.py (or --tables-pct / --columns-pct /
--baseline-tables-pct). Every prediction is scored in-process (ESM/EX via
src/metrics/esm_ex_cm.py — the same evaluator that fills pipeline.py's
raw_logs), so EX per k is stored as a number instead of only being printed
by the Spider CLI (closes open item D, IMPLEMENTATION_DECISIONS.md poin 28).

Baseline and GraphRAG get the SAME k (the baseline is no longer zero-shot).
Best k = highest EX; once ablation_results.csv holds both modes, the shared
recommendation is the k with the highest mean EX of the two (ties -> smaller k).
Run the two modes as separate processes (Kaggle time/memory):

Usage:
    python src/experiments/ablation.py --mode baseline --sample 1.0
    python src/experiments/ablation.py --mode graphrag --sample 1.0
    python src/experiments/ablation.py --mode graphrag --k-values 0 1 3 5 --tables-pct 0.6 --columns-pct 0.5

Outputs:
    outputs/predictions/ablation_{mode}_predictions_k{k}.txt   — SQL predictions
    outputs/predictions/ablation_{mode}_prompts_k{k}.jsonl     — per sample: prompt, tokens, esm, ex
    outputs/tables/ablation_results.csv                        — one row per mode x k (other mode's rows kept)
    outputs/tables/ablation_ex_per_k_{mode}.json               — {k: EX 0-100}, feeds
                                        run_all_dimensions.py --ex-per-k-file (Dimension 6)
"""

import argparse
import csv
import json
import logging
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.config import PipelineConfig, add_config_override_args, apply_config_overrides
from src.core.schema import build_schema_graph, load_spider_schema
from src.generation.few_shot import FewShotIndex, build_few_shot_block, build_few_shot_index
from src.generation.generation import build_prompt, generate_sql_with_token_count, load_model_and_tokenizer
from src.metrics import esm_ex_cm
from src.retrieval.baseline import (
    build_table_graph,
    build_table_index,
    build_table_schema_context,
    evaluate_table_linking,
    retrieve_baseline_tables,
)
from src.retrieval.retrieval import (
    build_schema_context,
    build_schema_index,
    evaluate_schema_linking,
    retrieve_graphrag_schema,
)
from src.utils.sql_normalize import normalize_sql_file_for_parsing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_CSV = Path("outputs/tables/ablation_results.csv")
CSV_FIELDS = [
    "mode", "k", "avg_ex", "avg_esm", "avg_recall", "avg_precision",
    "avg_prompt_tokens", "avg_output_tokens", "avg_token_consumption",
    "n_samples", "tables_pct", "columns_pct", "llm_model",
    "predictions_file", "prompts_file",
]


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class AblationResult:
    k: int
    mode: str
    avg_ex:                float   # EX  0-100 (in-process, eval_exec_match() resmi)
    avg_esm:               float   # ESM 0-100
    avg_recall:            float   # proxy schema linking (NOT SLA)
    avg_precision:         float
    avg_prompt_tokens:     float   # T_in  — input (prompt) tokens
    avg_output_tokens:     float   # T_out — output (generated SQL) tokens
    avg_token_consumption: float   # T = T_in + α × T_out  (the TEP metric)
    n_samples:             int
    predictions_file:      Path
    prompts_file:          Path


def _make_evaluator(cfg: PipelineConfig):
    """evaluate(pred, gold, db_id) -> QueryEvalResult, parser Schema cached per db_id."""
    kmaps = esm_ex_cm.build_kmaps(str(cfg.tables_json))
    schema_cache: dict = {}

    def evaluate(pred_sql: str, gold_sql: str, db_id: str):
        db_path = str(cfg.db_dir / db_id / f"{db_id}.sqlite")
        if db_id not in schema_cache:
            try:
                schema_cache[db_id] = esm_ex_cm.build_schema_for_db(db_path)
            except Exception:
                # evaluate_single_query() turns a None schema into its logged
                # all-zero fallback, so one broken DB cannot kill an hours-long run.
                logger.exception("Cannot open %s — its queries are scored as misses", db_path)
                schema_cache[db_id] = None
        return esm_ex_cm.evaluate_single_query(
            pred_sql, gold_sql, db_id, db_path, schema_cache[db_id], kmaps.get(db_id, {}),
        )

    return evaluate


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
    evaluate,
    mode: str = "graphrag",
) -> AblationResult:
    """
    Run full pipeline for one k value, score every prediction, save predictions.

    Args:
        cache    : SchemaIndex dict for mode="graphrag",
                   TableSchemaIndex dict for mode="baseline".
        evaluate : callable from _make_evaluator().
        mode     : "graphrag" (column-level retrieval) or
                   "baseline" (table-level retrieval).
    """
    cfg = replace(base_cfg, few_shot_k=k, use_full_schema_bypass=False)

    pred_path   = output_dir / f"ablation_{mode}_predictions_k{k}.txt"
    prompt_path = output_dir / f"ablation_{mode}_prompts_k{k}.jsonl"
    recalls, precisions = [], []
    prompt_tokens_list, output_tokens_list, consumption_list = [], [], []
    evals = []

    alpha = base_cfg.token_output_weight  # T = T_in + α × T_out

    with open(pred_path, "w", encoding="utf-8") as pred_file, \
         open(prompt_path, "w", encoding="utf-8") as prompt_file:
        for idx, item in enumerate(tqdm(dev_subset, desc=f"{mode} k={k}", leave=False)):
            db_id    = item["db_id"]
            question = item["question"]
            gold_sql = item["query"]
            prompt, n_in, n_out = "", 0, 0

            try:
                # Same retrieval + few-shot helpers as run_single()/run_single_baseline().
                if mode == "graphrag":
                    column_nodes, _ = retrieve_graphrag_schema(
                        graph, db_id, question, cache[db_id], embed_model, cfg,
                    )
                    r, p = evaluate_schema_linking(gold_sql, column_nodes, graph, db_id)
                    schema_context = build_schema_context(graph, column_nodes)
                else:  # baseline
                    table_nodes = retrieve_baseline_tables(
                        graph, db_id, question, cache[db_id], embed_model, cfg,
                    )
                    r, p = evaluate_table_linking(gold_sql, table_nodes, graph, db_id)
                    schema_context = build_table_schema_context(graph, table_nodes)

                few_shot_block = build_few_shot_block(question, few_shot_index, embed_model, cfg)

                # --- Prompt, generate, count tokens ---
                extracted_values = {
                    "strings": re.findall(r"'([^']*)'", question),
                    "numbers": re.findall(r"\d+", question),
                }
                prompt = build_prompt(question, schema_context, extracted_values, few_shot_block)
                pred_sql, n_out = generate_sql_with_token_count(prompt, llm, tokenizer, cfg)
                n_in = len(tokenizer.encode(prompt))   # T_in  — prompt tokens
                # n_out (T_out) is the real count of tokens the model emitted.

            except Exception:
                logger.exception("Error on question '%s'", question[:60])
                pred_sql, r, p = "SELECT 1", 0.0, 0.0

            # Scored like pipeline.py's raw_logs: the "SELECT 1" fallback counts as a miss.
            ev = evaluate(pred_sql, gold_sql, db_id)
            n_t = int(n_in + alpha * n_out)            # T = T_in + α × T_out

            # Append all metrics together so the lists never desync.
            evals.append(ev)
            recalls.append(r)
            precisions.append(p)
            prompt_tokens_list.append(n_in)
            output_tokens_list.append(n_out)
            consumption_list.append(n_t)

            prompt_file.write(
                json.dumps({
                    "i": idx + 1, "db_id": db_id, "question": question,
                    "tokens_in": n_in, "tokens_out": n_out,
                    "token_consumption": n_t,
                    "esm": ev.esm, "ex": ev.ex, "difficulty": ev.difficulty,
                    "prompt": prompt, "pred_sql": pred_sql,
                }, ensure_ascii=False) + "\n"
            )
            pred_file.write(f"{pred_sql}\n")
            logger.info(
                "  [%d/%d] db=%-20s T=%4d (in=%d out=%d) | EX=%d ESM=%d | R=%.1f%% P=%.1f%%",
                idx + 1, len(dev_subset), db_id, n_t, n_in, n_out, ev.ex, ev.esm,
                r * 100, p * 100,
            )

    def _safe_mean(lst):
        return float(np.mean(lst)) if lst else 0.0

    return AblationResult(
        k=k,
        mode=mode,
        avg_ex=esm_ex_cm.aggregate_ex(evals),
        avg_esm=esm_ex_cm.aggregate_esm(evals),
        avg_recall=_safe_mean(recalls),
        avg_precision=_safe_mean(precisions),
        avg_prompt_tokens=_safe_mean(prompt_tokens_list),
        avg_output_tokens=_safe_mean(output_tokens_list),
        avg_token_consumption=_safe_mean(consumption_list),
        n_samples=len(dev_subset),
        predictions_file=pred_path,
        prompts_file=prompt_path,
    )


# ---------------------------------------------------------------------------
# Best k
# ---------------------------------------------------------------------------

def recommend_k(rows: list[dict]) -> dict:
    """
    Best k by EX (ties -> smaller k, fewer tokens) from ablation_results.csv rows.
    Returns {"graphrag": k, "baseline": k, "shared": k}; a key is missing when
    its data is not there yet. "shared" needs both modes for the same k values
    and maximises their mean EX, since both arms must use one k.
    """
    ex: dict[str, dict[int, float]] = {}
    for row in rows:
        if row.get("avg_ex") not in (None, ""):
            ex.setdefault(row["mode"], {})[int(row["k"])] = float(row["avg_ex"])

    def best(scores: dict[int, float]) -> int:
        return max(scores, key=lambda k: (round(scores[k], 6), -k))

    out = {mode: best(scores) for mode, scores in ex.items() if scores}
    common = set(ex.get("graphrag", {})) & set(ex.get("baseline", {}))
    if common:
        out["shared"] = best({k: (ex["graphrag"][k] + ex["baseline"][k]) / 2 for k in common})
    return out


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_table(results: list[AblationResult], cfg: PipelineConfig) -> None:
    W = 110
    print("\n" + "=" * W)
    print("FEW-SHOT ABLATION RESULTS  (EX/ESM in-process, esm_ex_cm.py)")
    print(f"  Token Consumption T = T_in + {cfg.token_output_weight}×T_out")
    print("=" * W)
    print(f"{'mode':<10} {'k':>3} {'EX':>7} {'ESM':>7} {'recall':>8} {'precision':>10} "
          f"{'T_consume':>10} {'T_in':>8} {'T_out':>7}  predictions file")
    print("-" * W)
    for r in sorted(results, key=lambda x: (x.mode, x.k)):
        print(
            f"{r.mode:<10} {r.k:>3} {r.avg_ex:>6.2f}% {r.avg_esm:>6.2f}% "
            f"{r.avg_recall*100:>7.2f}% {r.avg_precision*100:>9.2f}% {r.avg_token_consumption:>10.1f} "
            f"{r.avg_prompt_tokens:>8.1f} {r.avg_output_tokens:>7.1f}  {r.predictions_file.name}"
        )
    print("=" * W)


def _run_ablation_evals(results: list[AblationResult], cfg: PipelineConfig) -> None:
    """Cross-check with the official Spider CLI (EM + EX) for every ablation prediction file."""
    import subprocess
    evaluator = Path("external/spider_eval/evaluation.py")
    if not evaluator.exists():
        logger.warning(
            "external/spider_eval/evaluation.py not found — cannot auto-run Spider eval.\n"
            "Download it first:\n"
            "  mkdir -p external/spider_eval && cd external/spider_eval\n"
            "  wget https://raw.githubusercontent.com/taoyds/spider/master/evaluation.py\n"
            "  wget https://raw.githubusercontent.com/taoyds/spider/master/process_sql.py"
        )
        return

    # process_sql.py resmi SPIDER cuma mengenali join-type keyword 'join'
    # polos -- predicted/gold SQL yang pakai INNER/CROSS/LEFT/RIGHT/FULL JOIN
    # bikin get_sql() crash di dalam evaluation.py CLI ini juga. Normalize
    # dulu sebelum dioper ke CLI -- lihat src/utils/sql_normalize.py dan
    # context/IMPLEMENTATION_DECISIONS.md poin 11.
    #
    # Gold file sama untuk semua run (satu dev set), jadi dinormalisasi
    # SEKALI per etype di luar loop, bukan diulang per (mode, k). --etype
    # match aman dinormalisasi PENUH (tidak pernah mengeksekusi SQL apapun).
    # --etype exec WAJIB cuma INNER/CROSS JOIN (execution_safe_only=True) --
    # LEFT/RIGHT/FULL JOIN karena itu tetap gagal parse di jalur exec ini,
    # known residual limitation (lihat IMPLEMENTATION_DECISIONS.md poin 11).
    eval_dir = cfg.predictions_file.parent
    normalized_gold = {
        "match": normalize_sql_file_for_parsing(
            cfg.gold_sql, eval_dir / "_normalized_match_gold.txt",
            has_db_id_suffix=True, execution_safe_only=False,
        ),
        "exec": normalize_sql_file_for_parsing(
            cfg.gold_sql, eval_dir / "_normalized_exec_gold.txt",
            has_db_id_suffix=True, execution_safe_only=True,
        ),
    }

    base_args = ["--db", str(cfg.db_dir), "--table", str(cfg.tables_json)]
    for r in sorted(results, key=lambda x: (x.mode, x.k)):
        for etype in ["match", "exec"]:
            label = "Exact Match" if etype == "match" else "Execution Accuracy"
            print("\n" + "=" * 70)
            print(f"  SPIDER EVAL — mode={r.mode}  k={r.k}  {label}")
            print("=" * 70)
            normalized_pred = normalize_sql_file_for_parsing(
                r.predictions_file,
                eval_dir / f"_normalized_{etype}_{r.predictions_file.stem}.txt",
                has_db_id_suffix=False, execution_safe_only=(etype == "exec"),
            )
            subprocess.run(
                ["python", "external/spider_eval/evaluation.py"] + base_args +
                ["--gold", str(normalized_gold[etype]), "--pred", str(normalized_pred), "--etype", etype],
                check=False,
            )


def _save_results(results: list[AblationResult], cfg: PipelineConfig, mode: str) -> list[dict]:
    """
    Merge this run into ablation_results.csv (rows of the OTHER mode are kept,
    so the two separate Kaggle runs end up in one file) and write the Dimension 6
    EX-per-k JSON. Returns all rows now in the CSV.
    """
    rows = []
    if RESULTS_CSV.exists():
        with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
            rows = [row for row in csv.DictReader(f) if row.get("mode") != mode]

    for r in sorted(results, key=lambda x: x.k):
        rows.append({
            "mode":                  r.mode,
            "k":                     r.k,
            "avg_ex":                round(r.avg_ex, 2),
            "avg_esm":               round(r.avg_esm, 2),
            "avg_recall":            round(r.avg_recall, 4),
            "avg_precision":         round(r.avg_precision, 4),
            "avg_prompt_tokens":     round(r.avg_prompt_tokens, 1),
            "avg_output_tokens":     round(r.avg_output_tokens, 1),
            "avg_token_consumption": round(r.avg_token_consumption, 1),
            "n_samples":             r.n_samples,
            "tables_pct":            cfg.top_k_tables_pct if mode == "graphrag" else cfg.baseline_top_k_tables_pct,
            "columns_pct":           cfg.top_k_columns_pct if mode == "graphrag" else "",
            "llm_model":             cfg.llm_model,
            "predictions_file":      str(r.predictions_file),
            "prompts_file":          str(r.prompts_file),
        })

    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, restval="", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    logger.info("CSV saved to %s", RESULTS_CSV)

    ex_path = RESULTS_CSV.parent / f"ablation_ex_per_k_{mode}.json"
    ex_path.write_text(
        json.dumps({str(r.k): round(r.avg_ex, 2) for r in sorted(results, key=lambda x: x.k)}, indent=2),
        encoding="utf-8",
    )
    logger.info("EX per k (Dimension 6 input) saved to %s", ex_path)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    sample_ratio: float,
    k_values: list[int] | None,
    mode: str = "graphrag",
    run_eval: bool = False,
    cfg: PipelineConfig | None = None,
) -> None:
    cfg = cfg or PipelineConfig()
    k_values = k_values or cfg.few_shot_frame
    output_dir = Path("outputs/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)

    if mode == "graphrag":
        logger.info("Top-k (stage 1): tables_pct=%.2f columns_pct=%.2f threshold=%.2f",
                    cfg.top_k_tables_pct, cfg.top_k_columns_pct, cfg.semantic_similarity_threshold)
    else:
        logger.info("Top-k (stage 1): baseline_tables_pct=%.2f", cfg.baseline_top_k_tables_pct)
    logger.info("Make sure these are the sweep.py winners (config.py or --tables-pct/...).")

    logger.info("Loading schema …")
    schema_df = load_spider_schema(cfg.tables_json)

    logger.info("Loading dev set …")
    with open(cfg.dev_json, "r", encoding="utf-8") as f:
        dev_data = json.load(f)

    n = max(1, int(len(dev_data) * sample_ratio))
    rng = np.random.default_rng(cfg.seed)
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
        cache = {
            db_id: build_schema_index(graph, db_id, embed_model)
            for db_id in tqdm(unique_db_ids, desc="Indexing columns")
        }
    else:  # baseline
        logger.info("Building table-level graph (Baseline) …")
        graph = build_table_graph(schema_df)
        cache = {
            db_id: build_table_index(graph, db_id, embed_model)
            for db_id in tqdm(unique_db_ids, desc="Indexing tables")
        }

    # Few-shot index (needed for any k > 0)
    few_shot_index: FewShotIndex | None = None
    if any(k > 0 for k in k_values):
        logger.info("Building few-shot index from training set …")
        few_shot_index = build_few_shot_index(cfg.train_json, embed_model)

    logger.info("Loading foreign-key maps for in-process ESM/EX …")
    evaluate = _make_evaluator(cfg)

    # Run each k
    results: list[AblationResult] = []
    for k in k_values:
        logger.info("=" * 50)
        logger.info("Running ablation: mode=%s k=%d", mode, k)
        result = _run_k(
            k, dev_subset, graph, cache, few_shot_index,
            embed_model, llm, tokenizer, cfg, output_dir, evaluate,
            mode=mode,
        )
        results.append(result)
        logger.info("mode=%s k=%d done → EX=%.2f%% ESM=%.2f%% T=%.1f",
                    mode, k, result.avg_ex, result.avg_esm, result.avg_token_consumption)

    _print_table(results, cfg)
    rows = _save_results(results, cfg, mode)

    best = recommend_k(rows)
    print(f"\nBest k by EX for {mode}: {best[mode]}")
    if "shared" in best:
        print(f"Shared few_shot_k for BOTH modes (highest mean EX): {best['shared']}"
              f"  (graphrag alone: {best['graphrag']}, baseline alone: {best['baseline']})")
        print("-> Put it in config.py few_shot_k (or pass --few-shot-k) before stage 3 (pipeline.py --models).")
    else:
        other = "baseline" if mode == "graphrag" else "graphrag"
        print(f"Run --mode {other} too; the shared k needs both modes in {RESULTS_CSV}.")

    if run_eval:
        logger.info("Cross-checking with the official Spider evaluator …")
        _run_ablation_evals(results, cfg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation stage 2: few-shot k, ranked by EX")
    parser.add_argument(
        "--mode", choices=["graphrag", "baseline"], default="graphrag",
        help="Retrieval mode to ablate (default: graphrag).",
    )
    parser.add_argument(
        "--sample", type=float, default=0.2,
        help="Fraction of dev set to use (default: 0.2)",
    )
    parser.add_argument(
        "--k-values", type=int, nargs="+", default=None,
        help="k values to test (default: config.py few_shot_frame)",
    )
    parser.add_argument(
        "--run-eval", action="store_true",
        help="Also cross-check every prediction file with the official Spider evaluation.py (EM + EX).",
    )
    add_config_override_args(parser)
    args, unknown = parser.parse_known_args()
    cfg = apply_config_overrides(PipelineConfig(), args)
    main(args.sample, args.k_values, mode=args.mode, run_eval=args.run_eval, cfg=cfg)
