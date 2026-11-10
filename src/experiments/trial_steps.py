"""
trial_steps.py — langkah bersama dua notebook trial schema linking (kode custom skripsi, bukan SPIDER).

    notebooks/trial_preprocessing.ipynb : semua kombinasi langkah preprocessing di satu pasangan (T, K)
    notebooks/trial_thresholds.ipynb    : grid T x K untuk preprocessing pilihan (saklar), 1-Stage di tiap T

Isinya dipindah dari graphrag_sla_trial.ipynb tanpa mengubah perhitungan (IMPLEMENTATION_DECISIONS.md poin 42):
retrieval lewat fungsi pipeline yang sama, SLA dari src/metrics/sla.py (macro-average per query), token dengan
tokenizer generasi, perbandingan dengan DDL penuh dari token_reduction.py (poin 35), EX zero-shot lewat
trial_ex.ExScorer dengan cache JSONL. Kode ini hanya menjalankan eksperimen: tidak ada yang memilih konfigurasi
(poin 33).
"""

import json
import os
from dataclasses import dataclass, field, replace
from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.metrics.sla import aggregate_sla, compute_sla, extract_ground_truth_schema
from src.metrics.token_reduction import MEDIAN_COLUMNS, full_db_summary, median_summary

EX_CACHE = Path("outputs") / "tables" / "sla_trial_ex_cache.jsonl"
PREPROCESSING_RECAP = "trial_preprocessing_recap.csv"
THRESHOLDS_RECAP = "trial_thresholds_recap.csv"

# Nilai tiap langkah preprocessing -> field PipelineConfig. Nilai pertama tiap langkah = acuan "mati" untuk efek utama.
STEPS = {
    "stopword": {"none": dict(query_stopword_mode="none"), "edge": dict(query_stopword_mode="edge"),
                 "remove": dict(query_stopword_mode="remove")},
    "value_link": {"off": dict(use_value_linking=False, value_link_numbers=False),
                   "text": dict(use_value_linking=True, value_link_numbers=False),
                   "text + numbers": dict(use_value_linking=True, value_link_numbers=True)},
    "traversal": {"full": dict(traversal_mode="full"), "fk_only": dict(traversal_mode="fk_only")},
    "no_pk": {"not connected": dict(connect_no_pk_tables=False), "connected": dict(connect_no_pk_tables=True)},
    "nama_embed": {"original": dict(embed_natural_names=False), "natural": dict(embed_natural_names=True)},
}
# Pipeline di kolom `pipeline` rekap: graphrag = 2-Stage, baseline = 1-Stage (nama tabel saja, poin 39)
PIPELINES = ("graphrag", "baseline")
DB_FIELDS = ("cols_db", "tables_sent", "tables_db", "tables_pct", "cols_pct",
             "schema_tokens_db", "schema_red_db", "prompt_tokens_k0_db", "prompt_red_db") + MEDIAN_COLUMNS


# ---------------------------------------------------------------------------
# Konfigurasi preprocessing
# ---------------------------------------------------------------------------

def knobs_of(settings: dict) -> dict:
    """{"stopword": "edge", ...} -> field PipelineConfig. Langkah yang tidak disebut tidak diubah."""
    out = {}
    for step, value in settings.items():
        if value not in STEPS[step]:
            raise ValueError(f"{step} = {value!r}; pilihan: {list(STEPS[step])}")
        out.update(STEPS[step][value])
    return out


def settings_of(cfg) -> dict:
    """Label tiap langkah preprocessing dari cfg (kebalikan knobs_of)."""
    return {
        "stopword": cfg.query_stopword_mode,
        "value_link": "off" if not cfg.use_value_linking else ("text + numbers" if cfg.value_link_numbers else "text"),
        "traversal": cfg.traversal_mode,
        "no_pk": "connected" if cfg.connect_no_pk_tables else "not connected",
        "nama_embed": "natural" if cfg.embed_natural_names else "original",
    }


def all_combinations(options: dict) -> list:
    """Semua kombinasi nilai di options ({langkah: [nilai, ...]}), urutan langkah mengikuti STEPS."""
    steps = [s for s in STEPS if s in options]
    for step in steps:
        for value in options[step]:
            knobs_of({step: value})   # nilai salah ketahuan sebelum run, bukan di tengah jalan
    return [dict(zip(steps, values)) for values in product(*(options[s] for s in steps))]


def switches_to_settings(stopword_filter: bool, value_linking: bool, value_numbers: bool, fk_only: bool,
                         connect_no_pk: bool, natural_names: bool) -> dict:
    """Saklar on/off notebook trial threshold -> label preprocessing."""
    return {
        "stopword": "edge" if stopword_filter else "none",
        "value_link": ("text + numbers" if value_numbers else "text") if value_linking else "off",
        "traversal": "fk_only" if fk_only else "full",
        "no_pk": "connected" if connect_no_pk else "not connected",
        "nama_embed": "natural" if natural_names else "original",
    }


def describe(settings: dict) -> str:
    return ", ".join(f"{k}={v}" for k, v in settings.items())


def two_stage_trial(cfg, settings: dict, t: float, k: float) -> tuple:
    """(nama trial, pipeline, cfg) untuk 2-Stage dengan preprocessing `settings` di (T, K)."""
    c = replace(cfg, table_similarity_threshold=t, semantic_similarity_threshold=k, **knobs_of(settings))
    return f"2-stage | {describe(settings_of(c))} | T{t:.2f}/K{k:.2f}", "graphrag", c


def one_stage_trial(cfg, t: float) -> tuple:
    """(nama trial, pipeline, cfg) untuk 1-Stage di T: teks tabel = nama saja, sama dengan Stage 1 2-Stage
    (poin 39, keputusan peneliti: 1-Stage+ tidak dijalankan di trial)."""
    c = replace(cfg, table_similarity_threshold=t, baseline_table_similarity_threshold=None,
                baseline_top_k_tables_pct=0.0, use_full_schema_bypass=False,
                embed_natural_names=True, baseline_embed_columns=False)
    return f"1-stage | T{t:.2f}", "baseline", c


# ---------------------------------------------------------------------------
# Data, index, dan embedding
# ---------------------------------------------------------------------------

class CachedEmbedder:
    """Drop-in untuk SentenceTransformer.encode(): memoize hasil per daftar teks, supaya n-gram yang sama
    tidak di-encode ulang di setiap konfigurasi."""

    def __init__(self, model):
        self.model, self._cache = model, {}

    def encode(self, texts, **kw):
        key = (tuple(texts), tuple(sorted(kw.items())))
        if key not in self._cache:
            self._cache[key] = self.model.encode(texts, **kw)
        return self._cache[key]


@dataclass
class TrialData:
    cfg: object
    schema_df: object
    graph: object               # node = kolom (2-Stage)
    tbl_graph: object           # node = tabel (1-Stage)
    db_schema: dict
    tokenizer: object
    dev_data: list
    indices: list
    samples: list               # [(query_id, item, gold_set)]
    n_failed: int = 0
    n_empty: int = 0
    table_cols: dict = field(default_factory=dict)
    sample_dbs: list = field(default_factory=list)
    n_cols_db: dict = field(default_factory=dict)
    n_tables_db: dict = field(default_factory=dict)
    full_ddl: dict = field(default_factory=dict)
    schema_tokens_db: dict = field(default_factory=dict)
    embed_model: object = None
    embedder: object = None
    col_index: dict = field(default_factory=dict)      # (db, nama natural?) -> SchemaIndex
    tbl_index: dict = field(default_factory=dict)      # (db, embed kolom?) -> TableSchemaIndex


def find_data_path(cfg, repo_root: Path) -> Path:
    """Data Spider: env SPIDER_DATA_PATH -> config (Kaggle) -> data/spider_data lokal."""
    for p in (os.environ.get("SPIDER_DATA_PATH"), cfg.data_path, repo_root / "data" / "spider_data",
              repo_root / "data" / "spider_data" / "spider_data"):
        if p and (Path(p) / "tables.json").exists():
            return Path(p)
    raise FileNotFoundError("tables.json tidak ketemu. Set SPIDER_DATA_PATH atau taruh dataset di data/spider_data/")


def prepare_samples(dev_data: list, indices: list, db_schema: dict, exclude_empty_gold: bool = False) -> tuple:
    """(samples, n_failed, n_empty): gold schema per pertanyaan dari parser resmi Spider (sla.py)."""
    samples, n_failed, n_empty = [], 0, 0
    for i in indices:
        item = dev_data[i]
        try:
            gold = extract_ground_truth_schema(item["query"], item["db_id"], db_schema)
        except Exception:
            n_failed += 1
            continue
        if not gold:
            n_empty += 1
            if exclude_empty_gold:
                continue
        samples.append((i, item, gold))
    return samples, n_failed, n_empty


def attach_databases(data: TrialData) -> TrialData:
    """Ukuran tiap DB di sampel dan DDL penuhnya (pembanding yang sama untuk kedua pipeline, poin 35)."""
    from src.retrieval.baseline import _all_table_nodes, build_table_schema_context
    from src.utils.prompt_stats import columns_per_table

    data.table_cols = columns_per_table(data.graph)
    data.sample_dbs = list(dict.fromkeys(item["db_id"] for _, item, _ in data.samples))
    for db in data.sample_dbs:
        data.n_cols_db[db] = len(data.db_schema[db])
        data.n_tables_db[db] = len(_all_table_nodes(data.tbl_graph, db))
        data.full_ddl[db] = build_table_schema_context(data.tbl_graph, _all_table_nodes(data.tbl_graph, db))
        data.schema_tokens_db[db] = len(data.tokenizer.encode(data.full_ddl[db]))
    return data


def load_trial_data(cfg, sample_ratio: float, exclude_empty_gold: bool = False) -> TrialData:
    """Schema, dua graph, tokenizer generasi, sampel stratified (sama dengan ablation.py, poin 34), gold schema."""
    from transformers import AutoTokenizer

    from src.core.schema import build_schema_graph, load_spider_schema
    from src.retrieval.baseline import build_table_graph
    from src.utils.sampling import stratified_indices
    from src.utils.schema_utils import load_db_schema

    schema_df = load_spider_schema(cfg.tables_json)
    db_schema = load_db_schema(str(cfg.tables_json))
    with open(cfg.dev_json, "r", encoding="utf-8") as f:
        dev_data = json.load(f)
    indices = stratified_indices(dev_data, sample_ratio, cfg.seed)
    samples, n_failed, n_empty = prepare_samples(dev_data, indices, db_schema, exclude_empty_gold)
    # Qwen2.5-Coder 3B/7B/14B memakai tokenizer yang sama, jadi angka token tidak bergantung pada ukuran model
    data = TrialData(cfg=cfg, schema_df=schema_df, graph=build_schema_graph(schema_df), tbl_graph=build_table_graph(schema_df),
                     db_schema=db_schema, tokenizer=AutoTokenizer.from_pretrained(cfg.llm_model), dev_data=dev_data,
                     indices=indices, samples=samples, n_failed=n_failed, n_empty=n_empty)
    return attach_databases(data)


def load_embedder(data: TrialData) -> TrialData:
    from sentence_transformers import SentenceTransformer

    data.embed_model = SentenceTransformer(data.cfg.embedding_model)
    data.embedder = CachedEmbedder(data.embed_model)
    return data


def index_for(data: TrialData, db: str, cfg):
    """Index kolom 2-Stage per (DB, sumber nama); dibangun sekali saat pertama dipakai."""
    from src.retrieval.retrieval import build_schema_index

    key = (db, cfg.embed_natural_names)
    if key not in data.col_index:
        data.col_index[key] = build_schema_index(data.graph, db, data.embed_model, include_table_in_text=False,
                                                 use_natural_names=cfg.embed_natural_names)
    return data.col_index[key]


def table_index_for(data: TrialData, db: str, embed_columns: bool):
    """Index tabel 1-Stage per (DB, teks tabel): nama saja atau nama + kolom (1-Stage+)."""
    from src.retrieval.baseline import build_table_index

    key = (db, embed_columns)
    if key not in data.tbl_index:
        data.tbl_index[key] = build_table_index(data.tbl_graph, db, data.embed_model, use_natural_names=True,
                                                include_columns=embed_columns)
    return data.tbl_index[key]


# ---------------------------------------------------------------------------
# Satu konfigurasi: retrieval per pertanyaan -> SLA, kolom, dan token
# ---------------------------------------------------------------------------

def query_row(data: TrialData, qid, item: dict, gold: set, pred, ddl: str) -> dict:
    """
    ddl = blok CREATE TABLE yang benar-benar masuk prompt. Token dihitung pada string mentah dengan tokenizer
    generasi, sama seperti T_in di pipeline. prompt_tokens_k0 = prompt zero-shot (k belum dipilih di tahap ini);
    bagian non-schema prompt identik untuk kedua pipeline, jadi beda T_in = beda token schema (poin 33).
    """
    from src.generation.generation import build_prompt

    pred = set(pred)
    tables = {s.split(".", 1)[0] for s in pred}
    db = item["db_id"]
    prompt = build_prompt(item["question"], ddl, {}, "")
    t, c = compute_sla(gold, pred, "table"), compute_sla(gold, pred, "column")
    return {
        "qid": qid, "db_id": db, "question": item["question"], "gold": sorted(gold), "pred": sorted(pred),
        "tbl_F1": t.f1, "col_P": c.precision, "col_R": c.recall, "col_F1": c.f1,
        "cols_sent": len(pred), "cols_db": data.n_cols_db[db],
        "cols_tables_full": sum(data.table_cols[(db, tb)] for tb in tables),
        "schema_tokens": len(data.tokenizer.encode(ddl)),
        "prompt_tokens_k0": len(data.tokenizer.encode(prompt)),
        # Dibanding DDL penuh database yang sama (poin 35)
        "tables_sent": len(tables), "tables_db": data.n_tables_db[db], "schema_tokens_db": data.schema_tokens_db[db],
        "prompt_tokens_k0_db": len(data.tokenizer.encode(build_prompt(item["question"], data.full_ddl[db], {}, ""))),
        "gold_sql": item["query"], "_prompt": prompt, "_t": t, "_c": c,
    }


def summarize(per_query: list) -> dict:
    t = aggregate_sla([r["_t"] for r in per_query])
    c = aggregate_sla([r["_c"] for r in per_query])
    # Hanya query dengan gold tidak kosong: gold kosong (SELECT count(*) ...) selalu P = R = 0 di SLA resmi
    ne = aggregate_sla([r["_c"] for r in per_query if r["gold"]])
    return {
        "tbl_P": t.precision, "tbl_R": t.recall, "tbl_F1": t.f1,
        "col_P": c.precision, "col_R": c.recall, "col_F1": c.f1,
        "col_P_ne": ne.precision, "col_R_ne": ne.recall,
        "cols_sent": np.mean([r["cols_sent"] for r in per_query]),
        "cols_db": np.mean([r["cols_db"] for r in per_query]),
        "cols_tables_full": np.mean([r["cols_tables_full"] for r in per_query]),
        "schema_tokens": np.mean([r["schema_tokens"] for r in per_query]),
        "prompt_tokens_k0": np.mean([r["prompt_tokens_k0"] for r in per_query]),
        **full_db_summary(per_query), **median_summary(per_query),
        "n": len(per_query),
    }


def run_2stage(data: TrialData, cfg) -> list:
    from src.retrieval.retrieval import build_schema_context, column_nodes_to_schema, retrieve_graphrag_schema

    out = []
    for qid, item, gold in data.samples:
        db = item["db_id"]
        nodes, _ = retrieve_graphrag_schema(data.graph, db, item["question"], index_for(data, db, cfg), data.embedder, cfg)
        out.append(query_row(data, qid, item, gold, column_nodes_to_schema(data.graph, nodes),
                             build_schema_context(data.graph, nodes)))
    return out


def run_1stage(data: TrialData, cfg) -> list:
    from src.retrieval.baseline import build_table_schema_context, retrieve_baseline_tables, table_nodes_to_schema

    out = []
    for qid, item, gold in data.samples:
        db = item["db_id"]
        index = table_index_for(data, db, cfg.baseline_embed_columns)
        nodes = retrieve_baseline_tables(data.tbl_graph, db, item["question"], index, data.embedder, cfg)
        out.append(query_row(data, qid, item, gold, table_nodes_to_schema(data.tbl_graph, nodes),
                             build_table_schema_context(data.tbl_graph, nodes)))
    return out


def _cache_key(pipeline: str, cfg, n: int) -> tuple:
    return (pipeline, cfg.table_similarity_threshold, cfg.semantic_similarity_threshold, cfg.query_stopword_mode,
            cfg.use_value_linking, cfg.value_link_numbers, cfg.traversal_mode, cfg.connect_no_pk_tables,
            cfg.embed_natural_names, cfg.baseline_embed_columns, cfg.max_ngram, cfg.top_k_tables_pct,
            cfg.top_k_columns_pct, n)


def recap_row(trial: str, pipeline: str, cfg, s: dict) -> dict:
    settings = settings_of(cfg)
    if pipeline != "graphrag":   # 1-Stage tidak punya langkah kolom, value linking, maupun traversal
        settings.update(value_link="-", traversal="-", no_pk="-")
    return {
        "trial": trial, "pipeline": pipeline,
        "table_threshold": cfg.table_similarity_threshold,
        "column_threshold": cfg.semantic_similarity_threshold if pipeline == "graphrag" else None,
        **settings, "max_ngram": cfg.max_ngram,
        "tbl_P": s["tbl_P"], "tbl_R": s["tbl_R"], "col_P": s["col_P"], "col_R": s["col_R"],
        "col_P_ne": s["col_P_ne"], "col_R_ne": s["col_R_ne"],
        "cols_sent": s["cols_sent"], "col_reduction": 1 - s["cols_sent"] / s["cols_db"],
        "cols_tables_full": s["cols_tables_full"], "schema_tokens": s["schema_tokens"],
        "prompt_tokens_k0": s["prompt_tokens_k0"], **{f: s[f] for f in DB_FIELDS}, "n": s["n"],
    }


def run_trials(data: TrialData, trials: list, cache: Optional[dict] = None, progress=None) -> tuple:
    """
    trials: [(nama trial, pipeline, cfg)] dari two_stage_trial() / one_stage_trial().
    cache: hasil per query yang sudah dihitung (dipakai ulang lintas sel); dict baru kalau None.
    Kembalikan (recap per trial, {nama trial: hasil per query}).
    """
    cache = {} if cache is None else cache
    rows, results = [], {}
    for trial, pipeline, c in (progress(trials) if progress else trials):
        key = _cache_key(pipeline, c, len(data.samples))
        if key not in cache:
            cache[key] = run_2stage(data, c) if pipeline == "graphrag" else run_1stage(data, c)
        results[trial] = cache[key]
        rows.append(recap_row(trial, pipeline, c, summarize(cache[key])))
    return pd.DataFrame(rows).drop_duplicates("trial").set_index("trial"), results


def save_recap(recap: pd.DataFrame, tables_dir: Path, filename: str) -> Path:
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / filename
    recap.to_csv(path, float_format="%.4f")
    return path


# ---------------------------------------------------------------------------
# EX zero-shot (butuh GPU)
# ---------------------------------------------------------------------------

def score_ex(recap: pd.DataFrame, results: dict, cfg, tokenizer, cache_path: Path, llm=None, progress=None) -> tuple:
    """
    Isi kolom `ex` untuk tiap trial. Prompt = prompt_tokens_k0 yang sama persis; prompt identik hanya digenerate
    sekali (greedy, deterministik) dan disimpan di cache JSONL, jadi run yang terputus bisa dilanjutkan.
    Kembalikan (recap, llm) supaya model tidak dimuat ulang di sel berikutnya.
    """
    import torch

    from src.experiments.ablation import _make_evaluator
    from src.experiments.trial_ex import ExScorer
    from src.generation.generation import generate_sql_with_token_count, load_model_and_tokenizer

    if not torch.cuda.is_available():
        print("[WARN] GPU tidak tersedia: EX dilewati. Nyalakan accelerator GPU, atau set RUN_EX = False.")
        return recap, llm
    if llm is None:
        llm, _ = load_model_and_tokenizer(cfg)
    scorer = ExScorer(lambda p: generate_sql_with_token_count(p, llm, tokenizer, cfg), _make_evaluator(cfg),
                      cache_path, cfg.llm_model)
    total = sum(len(results[t]) for t in recap.index)
    unique = len({r["_prompt"] for t in recap.index for r in results[t]})
    print(f"{len(recap)} konfigurasi | {total} prompt, {unique} unik (hanya yang unik digenerate)")
    for t in (progress(list(recap.index)) if progress else recap.index):
        recap.loc[t, "ex"] = np.mean([scorer.score(r["_prompt"], r["gold_sql"], r["db_id"]) for r in results[t]])
    print(f"Digenerate: {scorer.generated} | dari cache: {scorer.reused} | gagal (dihitung salah): {scorer.failed}")
    return recap, llm


# ---------------------------------------------------------------------------
# Bahan baca (tidak memilih apa pun)
# ---------------------------------------------------------------------------

_VIEW = {"pipeline": None, "table_threshold": "{:.2f}", "column_threshold": "{:.2f}", "stopword": None,
         "value_link": None, "traversal": None, "no_pk": None, "nama_embed": None, "ex": "{:.1%}", "col_P_ne": "{:.1%}",
         "col_R_ne": "{:.1%}", "tbl_R": "{:.1%}", "cols_sent": "{:.1f}", "tables_pct": "{:.1%}", "cols_pct": "{:.1%}",
         "schema_tokens": "{:.1f}", "prompt_tokens_k0": "{:.1f}", "prompt_red_db": "{:.1%}"}


def styled_recap(recap: pd.DataFrame, target_p: Optional[float] = None, target_r: Optional[float] = None):
    """Kolom inti rekap untuk tampilan notebook (rekap lengkap ada di CSV dan Excel). Baris hijau = memenuhi
    target P/R kolom (non-empty gold): penanda visual saja, bukan pilihan."""
    view = recap[[c for c in _VIEW if c in recap.columns]]

    def _hit(row):
        ok = target_p is not None and row.get("col_P_ne", 0) >= target_p and row.get("col_R_ne", 0) >= target_r
        return ["background-color: #d4edda; color: #155724" if ok else ""] * len(row)

    return view.style.apply(_hit, axis=1).format({c: f for c, f in _VIEW.items() if f and c in view.columns}, na_rep="-")

def compare_trials(results: dict, a: str, b: str) -> pd.DataFrame:
    """Per query: kolom yang ditambahkan dan dibuang trial b dibanding a, dan kolom gold yang ikut masuk/hilang."""
    ra, rb = pd.DataFrame(results[a]), pd.DataFrame(results[b])
    d = ra[["qid", "db_id", "question", "gold"]].copy()
    d["added"] = [sorted(set(y) - set(x)) for x, y in zip(ra["pred"], rb["pred"])]
    d["removed"] = [sorted(set(x) - set(y)) for x, y in zip(ra["pred"], rb["pred"])]
    d["gold_gained"] = [sorted(set(g) & (set(y) - set(x))) for g, x, y in zip(ra["gold"], ra["pred"], rb["pred"])]
    d["gold_lost"] = [sorted(set(g) & (set(x) - set(y))) for g, x, y in zip(ra["gold"], ra["pred"], rb["pred"])]
    return d[(d["added"].str.len() > 0) | (d["removed"].str.len() > 0)]


def compare_with_one_stage(recap: pd.DataFrame) -> pd.DataFrame:
    """
    Tiap baris 2-Stage (T, K) dibanding baris 1-Stage di T yang sama (poin 33). Query sama, jadi
    reduksi = 1 - Σ 2-Stage / Σ 1-Stage.
    """
    g = recap[recap["pipeline"] == "graphrag"]
    b = recap[recap["pipeline"] == "baseline"].set_index("table_threshold")
    if g.empty or b.empty:
        raise ValueError("Rekap tidak punya baris 2-Stage dan 1-Stage: jalankan grid dulu.")
    vs = g[["table_threshold", "column_threshold", "tbl_R", "col_P_ne", "col_R_ne"]].copy()
    vs["col_R_ne_B"] = g["table_threshold"].map(b["col_R_ne"])
    for q, name in (("cols_sent", "cols"), ("schema_tokens", "schema_tok"), ("prompt_tokens_k0", "prompt_tok")):
        base = g["table_threshold"].map(b[q])
        vs[f"{name}_G"], vs[f"{name}_B"], vs[f"{name}_red"] = g[q], base, 1 - g[q] / base
    vs["table_effect"] = 1 - g["cols_tables_full"] / vs["cols_B"]
    vs["prune_effect"] = 1 - g["cols_sent"] / g["cols_tables_full"]
    if "ex" in recap.columns:
        vs["ex_G"], vs["ex_B"] = g["ex"], g["table_threshold"].map(b["ex"])
        vs["ex_delta"] = vs["ex_G"] - vs["ex_B"]
    return vs.sort_values(["table_threshold", "column_threshold"], kind="stable")
