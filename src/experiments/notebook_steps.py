"""
notebook_steps.py — langkah-langkah eval_pipeline.ipynb, supaya sel notebook cukup satu atau dua baris.

Di sini tidak ada metode baru: isinya logika yang sebelumnya tertulis di sel notebook.
1-Stage = "baseline" di kode, 2-Stage = "graphrag" di kode.
"""

# Membuat wadah data sederhana
from dataclasses import dataclass, field
# json untuk membaca dan menulis file JSON
import json
# re untuk mencari angka dan teks berkutip di pertanyaan
import re
# shutil untuk menyalin folder dan membuat zip
import shutil
# subprocess untuk menjalankan evaluator resmi Spider
import subprocess
# sys untuk mendapatkan lokasi Python yang sedang dipakai
import sys
# Path untuk menulis lokasi file
from pathlib import Path
# Any untuk menandai data yang jenisnya bebas
from typing import Any

# numpy untuk rata-rata dan pengambilan sampel acak
import numpy as np
# pandas untuk membuat tabel
import pandas as pd
# openpyxl untuk membaca kembali file Excel yang dibuat
from openpyxl import load_workbook
# Model embedding BGE-M3
from sentence_transformers import SentenceTransformer
# Progress bar
from tqdm.auto import tqdm

# Fungsi membaca schema dan membangun graph per kolom
from src.core.schema import build_schema_graph, load_spider_schema
# Fungsi membuat file Excel
from src.experiments.export_excel import build_workbook
# Fungsi menyimpan hasil metrik
from src.experiments.metrics_summary import save_metrics_results
# Fungsi evaluasi dan pipeline yang dipakai bersama pipeline.py
from src.experiments.pipeline import (
    # Wadah hasil 2-Stage
    PipelineResult,
    # Mengubah node kolom 2-Stage menjadi "tabel.kolom"
    _column_nodes_to_predicted_schema,
    # Menyusun teks confound diagnostics
    _format_confound_diag,
    # Membuat penghitung confound diagnostics
    _new_confound_diag,
    # Mengubah node tabel 1-Stage menjadi "tabel.kolom"
    _table_nodes_to_predicted_schema,
    # Memperbarui penghitung confound diagnostics
    _update_confound_diag,
    # Menghitung analisis dimensi tanpa menyimpan
    compute_dimension_analysis,
    # Menjalankan satu pertanyaan 2-Stage
    run_single,
    # Mengunci sumber acak
    set_seed,  # noqa: F401  (dipanggil notebook lewat steps.set_seed)
)
# Fungsi tampilan tabel, sama dengan yang dipakai Excel
from src.experiments import result_tables as rt
# Fungsi menampilkan tabel di notebook
from src.experiments.result_tables import show
# Fungsi menyimpan hasil dimensi
from src.experiments.run_all_dimensions import save_dimension_outputs
# Fungsi few-shot
from src.generation.few_shot import build_few_shot_block, build_few_shot_index
# Fungsi LLM dan prompt
from src.generation.generation import build_prompt, generate_sql_with_token_count, load_model_and_tokenizer
# Evaluator ESM, EX, dan CM per query
from src.metrics import esm_ex_cm
# Fungsi menghitung SLA dan mengambil kolom gold
from src.metrics.sla import compute_sla, extract_ground_truth_schema
# Fungsi retrieval 1-Stage
from src.retrieval.baseline import (
    # Wadah hasil 1-Stage
    BaselineResult,
    # Membangun graph per tabel
    build_table_graph,
    # Membuat embedding tabel
    build_table_index,
    # Membuat DDL semua kolom tabel terpilih
    build_table_schema_context,
    # Menghitung proxy 1-Stage
    evaluate_table_linking,
    # Menjalankan satu pertanyaan 1-Stage
    run_single_baseline,
    # Memilih tabel yang relevan
    semantic_linking_table_level,
    # Mengubah node tabel menjadi daftar kolom
    table_nodes_to_schema,
    # Menambahkan tabel penghubung
    trace_table_paths,
)
# Fungsi retrieval 2-Stage
from src.retrieval.retrieval import (
    # Membuat DDL kolom terpilih
    build_schema_context,
    # Membuat embedding tabel dan kolom
    build_schema_index,
    # Mengubah node kolom menjadi daftar kolom
    column_nodes_to_schema,
    # Menghitung proxy 2-Stage
    evaluate_schema_linking,
    # Stage 2: memilih kolom
    retrieve_candidate_columns,
    # Stage 1: memilih tabel
    retrieve_candidate_tables,
    # Seluruh proses retrieval, dipakai loop utama
    retrieve_graphrag_schema,
    # Mencari jalur antar kolom
    trace_schema_paths,
    # Menyiapkan graph untuk traversal
    traversal_graph,
    # Mencari kolom dari nilai sel yang disebut di pertanyaan
    value_linked_columns,
)
# Fungsi membaca daftar kolom tiap database
from src.utils.schema_utils import load_db_schema
# Fungsi menyamakan bentuk JOIN untuk evaluator resmi
from src.utils.sql_normalize import normalize_sql_file_for_parsing


# ---------------------------------------------------------------------------
# Wadah data
# ---------------------------------------------------------------------------

# Menyatukan semua alat yang dipakai berulang, supaya pemanggilan fungsi pendek
@dataclass
class Tools:
    # Konfigurasi run
    cfg: Any
    # Graph per kolom untuk 2-Stage
    col_graph: Any
    # Graph per tabel untuk 1-Stage
    tbl_graph: Any
    # Model embedding
    embed_model: Any
    # Model LLM
    llm: Any
    # Tokenizer LLM
    tokenizer: Any
    # Embedding kolom tiap database
    col_cache: dict
    # Embedding tabel tiap database
    tbl_cache: dict
    # Index contoh few-shot
    few_shot_idx: Any
    # Alat evaluasi SQL
    evaluator: Any


# Menyatukan semua hasil dari loop utama
@dataclass
class RunResults:
    # Penghitung confound diagnostics
    confound_diag: dict
    # Hasil ringkas 2-Stage
    graphrag_results: list = field(default_factory=list)
    # Hasil ringkas 1-Stage
    baseline_results: list = field(default_factory=list)
    # Catatan lengkap 2-Stage
    graphrag_logs: list = field(default_factory=list)
    # Catatan lengkap 1-Stage
    baseline_logs: list = field(default_factory=list)
    # Teks SQL 2-Stage
    graphrag_preds: list = field(default_factory=list)
    # Teks SQL 1-Stage
    baseline_preds: list = field(default_factory=list)
    # Daftar error saat generate
    gen_errors: list = field(default_factory=list)
    # Daftar error saat evaluasi
    eval_errors: list = field(default_factory=list)


# Alat untuk menghitung ESM, EX, CM, dan kolom gold
class SqlEvaluator:
    # Menyiapkan data yang cukup dibaca sekali
    def __init__(self, cfg) -> None:
        # Tampilkan status proses
        print("Loading db_schema + foreign-key maps ...")
        # Simpan konfigurasi
        self.cfg = cfg
        # Baca daftar kolom tiap database
        self.db_schema = load_db_schema(str(cfg.tables_json))
        # Baca peta foreign key tiap database
        self.kmaps = esm_ex_cm.build_kmaps(str(cfg.tables_json))
        # Tempat menyimpan objek schema supaya tidak dibuat berulang
        self.schema_cache = {}
        # Tampilkan jumlah database yang terbaca
        print(f"db_schema: {len(self.db_schema)} DB | kmaps: {len(self.kmaps)} DB")

    # Mendapatkan lokasi file SQLite sebuah database
    def db_path(self, db_id: str) -> str:
        # Susun path file .sqlite
        return str(self.cfg.db_dir / db_id / f"{db_id}.sqlite")

    # Mengevaluasi satu prediksi: hasilnya ESM, EX, CM, dan tingkat kesulitan
    def evaluate(self, pred_sql: str, gold_sql: str, db_id: str):
        # Ambil lokasi database
        path = self.db_path(db_id)
        # Buat objek schema kalau belum pernah dibuat
        if db_id not in self.schema_cache:
            # Simpan supaya bisa dipakai lagi
            self.schema_cache[db_id] = esm_ex_cm.build_schema_for_db(path)
        # Jalankan evaluasinya
        return esm_ex_cm.evaluate_single_query(pred_sql, gold_sql, db_id, path, self.schema_cache[db_id], self.kmaps[db_id])

    # Mengambil kolom yang dipakai gold SQL
    def gold_columns(self, gold_sql: str, db_id: str) -> list:
        # Urutkan supaya hasilnya selalu sama
        return sorted(extract_ground_truth_schema(gold_sql, db_id, self.db_schema))


# ---------------------------------------------------------------------------
# Bagian 3-8: persiapan
# ---------------------------------------------------------------------------

# Menampilkan konfigurasi run dalam satu tabel
def show_run_config(cfg, sample_ratio: float, artifact_dir: Path, raw_logs_dir: Path) -> None:
    # Susun pasangan nama dan nilai
    rows = [
        # Lokasi dataset
        ("Data path", str(cfg.data_path)),
        # Model embedding
        ("Embedding model", cfg.embedding_model),
        # Model LLM yang dipilih
        ("LLM model", cfg.llm_model),
        # Threshold tabel
        ("Table threshold (Stage 1, both pipelines)", cfg.table_similarity_threshold),
        # Threshold kolom
        ("Column threshold (Stage 2, GraphRAG)", cfg.semantic_similarity_threshold),
        # Apakah embedding memakai nama natural Spider
        ("Embed natural names", cfg.embed_natural_names),
        # Cara menangani stopword
        ("Query stopword mode", cfg.query_stopword_mode),
        # Batas top-k tabel dan kolom
        ("Top-k tables / columns pct (0 = no cap)", f"{cfg.top_k_tables_pct:.0%} / {cfg.top_k_columns_pct:.0%}"),
        # Batas top-k tabel pada 1-Stage
        ("Baseline top-k tables pct (0 = no cap)", f"{cfg.baseline_top_k_tables_pct:.0%}"),
        # Status value linking
        ("Value linking (text / numbers)", f"{cfg.use_value_linking} / {cfg.value_link_numbers}"),
        # Kolom jalur yang masuk prompt
        ("Traversal mode", cfg.traversal_mode),
        # Apakah tabel tanpa PK disambung di graph
        ("Connect tables without PK", cfg.connect_no_pk_tables),
        # Panjang n-gram maksimum
        ("max_ngram", cfg.max_ngram),
        # Jumlah contoh few-shot
        ("Few-shot k (both pipelines)", cfg.few_shot_k),
        # Cara memilih few-shot
        ("Few-shot selection (both pipelines)", cfg.few_shot_selection),
        # Bobot token output pada rumus T
        ("α (output-token weight)", cfg.token_output_weight),
        # Jenis run
        ("SAMPLE_RATIO", f"{sample_ratio} ({'full run' if sample_ratio >= 1.0 else 'trial, outputs/trial/'})"),
        # Folder hasil model ini
        ("Results directory", str(artifact_dir)),
        # Folder raw_logs model ini
        ("raw_logs directory", str(raw_logs_dir)),
    ]
    # Jadikan tabel dengan semua nilai berupa teks, lalu tampilkan
    show(pd.DataFrame(rows, columns=["Parameter", "Value"]).astype({"Value": str}), "Run configuration")


# Membaca pertanyaan dev set dan mengambil sampel kalau perlu
def load_dev_questions(cfg, sample_ratio: float) -> list:
    # Buka file dev set Spider
    with open(cfg.dev_json, encoding="utf-8") as f:
        # Baca semua pertanyaan dan gold SQL
        dev_data = json.load(f)
    # Run penuh memakai seluruh data
    if sample_ratio >= 1.0:
        # Tampilkan jumlah pertanyaan
        print(f"Dev set: {len(dev_data)} pertanyaan, semuanya dipakai")
        # Kembalikan semua data
        return dev_data
    # Hitung jumlah sampel, minimal 1
    n = max(1, int(len(dev_data) * sample_ratio))
    # Buat pengacak dengan seed tetap supaya sampelnya selalu sama
    rng = np.random.default_rng(cfg.seed)
    # Pilih posisi data secara acak tanpa pengulangan
    indices = rng.choice(len(dev_data), size=n, replace=False)
    # Tampilkan jumlah sampel
    print(f"Dev set: {n} dari {len(dev_data)} pertanyaan dipakai")
    # Ambil data terpilih dengan urutan aslinya
    return [dev_data[i] for i in sorted(indices)]


# Menulis gold SQL sesuai sampel, untuk evaluator resmi
def write_gold_file(questions: list, artifact_dir: Path) -> Path:
    # Tentukan lokasi file
    gold_path = artifact_dir / "gold_subset.sql"
    # Buka file untuk ditulis
    with open(gold_path, "w", encoding="utf-8") as f:
        # Proses tiap pertanyaan
        for item in questions:
            # Tulis "SQL<TAB>db_id", format yang diminta evaluator resmi
            f.write(item["query"].strip() + "\t" + item["db_id"] + "\n")
    # Kembalikan lokasi file
    return gold_path


# Membangun kedua graph schema
def build_graphs(cfg) -> tuple:
    # Baca tables.json menjadi tabel berisi semua kolom
    schema_df = load_spider_schema(cfg.tables_json)
    # Bangun graph per kolom untuk 2-Stage
    col_graph = build_schema_graph(schema_df)
    # Bangun graph per tabel untuk 1-Stage
    tbl_graph = build_table_graph(schema_df)
    # Tampilkan ukuran graph 2-Stage
    print(f"Graph 2-Stage: {col_graph.number_of_nodes():,} node, {col_graph.number_of_edges():,} edge")
    # Tampilkan ukuran graph 1-Stage
    print(f"Graph 1-Stage: {tbl_graph.number_of_nodes():,} node, {tbl_graph.number_of_edges():,} edge")
    # Kembalikan kedua graph
    return col_graph, tbl_graph


# Memuat model embedding dan LLM
def load_models(cfg) -> tuple:
    # Tampilkan model embedding yang dimuat
    print(f"Loading embedding model: {cfg.embedding_model}")
    # Muat model embedding
    embed_model = SentenceTransformer(cfg.embedding_model)
    # Tampilkan LLM yang dimuat
    print(f"Loading LLM: {cfg.llm_model} (4-bit) ...")
    # Muat LLM dan tokenizer sesuai MODEL yang dipilih
    llm, tokenizer = load_model_and_tokenizer(cfg)
    # Tanda bahwa kedua model sudah siap
    print("Model siap.")
    # Kembalikan ketiganya
    return embed_model, llm, tokenizer


# Membuat embedding schema dan index few-shot
def build_indexes(cfg, questions: list, col_graph, tbl_graph, embed_model) -> tuple:
    # Ambil daftar database di sampel tanpa duplikat
    db_ids = list(dict.fromkeys(item["db_id"] for item in questions))
    # Tampilkan jumlah database
    print(f"{len(db_ids)} database unik di sampel ini")
    # Tempat embedding kolom dan tabel per database
    col_cache, tbl_cache = {}, {}
    # Proses tiap database dengan progress bar
    for db_id in tqdm(db_ids, desc="Index schema"):
        # Buat embedding kolom untuk 2-Stage
        col_cache[db_id] = build_schema_index(col_graph, db_id, embed_model, use_natural_names=cfg.embed_natural_names)
        # Buat embedding tabel untuk 1-Stage
        tbl_cache[db_id] = build_table_index(tbl_graph, db_id, embed_model, use_natural_names=cfg.embed_natural_names)
    # Kalau k = 0, kedua varian berjalan tanpa contoh
    if cfg.few_shot_k == 0:
        # Tampilkan keterangannya
        print("few_shot_k=0 -> zero-shot untuk kedua pipeline")
        # Kembalikan tanpa index few-shot
        return col_cache, tbl_cache, None
    # Tampilkan status proses
    print("Building few-shot index dari train set ...")
    # Buat embedding contoh dari train set
    few_shot_idx = build_few_shot_index(cfg.train_json, embed_model, cfg)
    # Tampilkan jumlah contoh dan cara seleksinya
    print(f"Few-shot index: {len(few_shot_idx.examples)} contoh training (selection={few_shot_idx.selection})")
    # Kembalikan semua index
    return col_cache, tbl_cache, few_shot_idx


# ---------------------------------------------------------------------------
# Bagian 9-10: contoh satu pertanyaan
# ---------------------------------------------------------------------------

# Mengambil angka dan teks berkutip dari pertanyaan
def _extract_values(question: str) -> dict:
    # Kembalikan keduanya untuk prompt
    return {"strings": re.findall(r"'([^']*)'", question), "numbers": re.findall(r"\d+", question)}


# Mengubah nama node "db.tabel.kolom" menjadi "tabel.kolom"
def _short(node: str) -> str:
    # Buang nama database di depan
    return node.split(".", 1)[1]


# Menampilkan proses 2-Stage untuk satu pertanyaan; hasilnya tidak masuk hasil akhir
def walk_through_2stage(tools: Tools, item: dict) -> dict:
    # Ambil konfigurasi supaya penulisan lebih pendek
    cfg = tools.cfg
    # Pisahkan pertanyaan, gold SQL, dan nama database
    q, gold, db = item["question"], item["query"], item["db_id"]
    # Tampilkan nama database
    print(f"DB       : {db}")
    # Tampilkan pertanyaan
    print(f"Question : {q}")
    # Tampilkan gold SQL
    print(f"Gold SQL : {gold}")

    # Stage 1: pilih tabel kandidat
    cand_tables = retrieve_candidate_tables(q, tools.col_cache[db], tools.embed_model, cfg)
    # Tampilkan tabel yang terpilih
    print(f"\n[Stage 1] threshold tabel {cfg.table_similarity_threshold} -> {len(cand_tables)} tabel: {cand_tables}")

    # Stage 2: pilih kolom di dalam tabel kandidat
    detected = retrieve_candidate_columns(q, tools.col_cache[db], cand_tables, tools.embed_model, cfg)
    # Tampilkan kolom yang terpilih
    print(f"[Stage 2] threshold kolom {cfg.semantic_similarity_threshold} -> {[_short(n) for n in detected]}")

    # Value linking: cari kolom dari nilai sel yang disebut di pertanyaan
    linked = value_linked_columns(tools.col_graph, db, q, cfg) if cfg.use_value_linking else []
    # Gabungkan kolom Stage 2 dan value linking tanpa duplikat
    anchors = list(dict.fromkeys(detected + linked))
    # Tampilkan kolom hasil value linking
    print(f"[Value linking] {[_short(n) for n in linked]}")

    # Graph traversal: cari jalur terpendek antar kolom
    c_nodes, paths, relations = trace_schema_paths(traversal_graph(tools.col_graph, cfg), db, anchors)
    # Mode fk_only hanya mengambil kolom JOIN
    if cfg.traversal_mode == "fk_only":
        # Ambil kedua ujung setiap edge foreign key
        path_nodes = [n for r in relations if r["relation"] == "foreign_key" for n in (r["from"], r["to"])]
    # Mode full mengambil semua kolom di jalur
    else:
        # Ambil semua node pada setiap jalur
        path_nodes = [n for p in paths for n in p]
    # Gabungkan anchor dan kolom jalur tanpa duplikat
    column_nodes = list(dict.fromkeys(c_nodes + path_nodes))
    # Tampilkan total kolom
    print(f"[Traversal] {len(paths)} jalur, mode={cfg.traversal_mode}, total {len(column_nodes)} kolom masuk prompt")
    # Pastikan contoh ini sama dengan fungsi yang dipakai loop utama
    assert column_nodes == retrieve_graphrag_schema(tools.col_graph, db, q, tools.col_cache[db], tools.embed_model, cfg)[0], \
        "walk-through tidak sama dengan retrieve_graphrag_schema()"

    # Ubah kolom terpilih menjadi teks CREATE TABLE
    schema_context = build_schema_context(tools.col_graph, column_nodes)
    # Tampilkan DDL yang dikirim ke LLM
    print(f"\n[Schema yang dikirim ke LLM]\n{schema_context}")
    # Hitung SLA kolom untuk pertanyaan ini
    sla = compute_sla(set(tools.evaluator.gold_columns(gold, db)), set(column_nodes_to_schema(tools.col_graph, column_nodes)))
    # Tampilkan nilai SLA
    print(f"\nSLA kolom: P={sla.precision:.2f} R={sla.recall:.2f} F1={sla.f1:.2f}")

    # Ambil contoh few-shot dan susun menjadi teks
    few_shot_block = build_few_shot_block(q, tools.few_shot_idx, tools.embed_model, cfg)
    # Tampilkan contoh, atau keterangan kalau kosong
    print(f"\n[Few-shot]\n{few_shot_block if few_shot_block else '(kosong: few_shot_k=0)'}")
    # Susun prompt lengkap untuk LLM
    prompt = build_prompt(q, schema_context, _extract_values(q), few_shot_block)
    # Hitung jumlah token prompt
    n_in = len(tools.tokenizer.encode(prompt))
    # Hasilkan SQL dan hitung token output
    pred, n_out = generate_sql_with_token_count(prompt, tools.llm, tools.tokenizer, cfg)
    # Hitung ESM, EX, dan CM
    evaluation = tools.evaluator.evaluate(pred, gold, db)
    # Tampilkan SQL prediksi
    print(f"\nPred SQL : {pred}")
    # Tampilkan hasil singkat
    print(f"T_in={n_in} T_out={n_out} | ESM={evaluation.esm} EX={evaluation.ex} | Difficulty={evaluation.difficulty}")
    # Kembalikan hasilnya untuk dibandingkan dengan 1-Stage
    return {"few_shot_block": few_shot_block, "pred": pred, "n_in": n_in, "n_out": n_out, "eval": evaluation, "sla": sla}


# Menampilkan proses 1-Stage untuk pertanyaan yang sama, lalu membandingkannya dengan 2-Stage
def walk_through_1stage(tools: Tools, item: dict, two_stage: dict) -> None:
    # Ambil konfigurasi supaya penulisan lebih pendek
    cfg = tools.cfg
    # Pisahkan pertanyaan, gold SQL, dan nama database
    q, gold, db = item["question"], item["query"], item["db_id"]
    # Pilih tabel yang relevan dengan pertanyaan
    detected = semantic_linking_table_level(tools.tbl_cache[db], q, tools.embed_model, cfg)
    # Tampilkan tabel yang terpilih
    print(f"[Pilih tabel] threshold {cfg.table_similarity_threshold} -> {len(detected)} tabel: {detected}")
    # Tambahkan tabel penghubung yang dibutuhkan untuk JOIN
    table_nodes = trace_table_paths(tools.tbl_graph, detected)
    # Tampilkan daftar tabel akhir
    print(f"[Path tracing] tabel final: {table_nodes}")
    # Ubah tabel terpilih menjadi teks CREATE TABLE
    context = build_table_schema_context(tools.tbl_graph, table_nodes)
    # Tampilkan DDL yang dikirim ke LLM
    print(f"\n[Schema yang dikirim ke LLM]\n{context}")
    # Hitung SLA kolom untuk pertanyaan ini
    sla = compute_sla(set(tools.evaluator.gold_columns(gold, db)), set(table_nodes_to_schema(tools.tbl_graph, table_nodes)))
    # Susun prompt dengan few-shot yang sama dengan 2-Stage
    prompt = build_prompt(q, context, _extract_values(q), two_stage["few_shot_block"])
    # Hitung jumlah token prompt
    n_in = len(tools.tokenizer.encode(prompt))
    # Hasilkan SQL dan hitung token output
    pred, n_out = generate_sql_with_token_count(prompt, tools.llm, tools.tokenizer, cfg)
    # Hitung ESM, EX, dan CM
    evaluation = tools.evaluator.evaluate(pred, gold, db)
    # Susun baris 2-Stage
    row_2 = {"Variant": "2-Stage", "Predicted SQL": two_stage["pred"], "T_in": two_stage["n_in"], "T_out": two_stage["n_out"],
             # Nilai ESM, EX, dan SLA kolom 2-Stage
             "ESM": two_stage["eval"].esm, "EX": two_stage["eval"].ex, "SLA column F1": two_stage["sla"].f1}
    # Susun baris 1-Stage
    row_1 = {"Variant": "1-Stage", "Predicted SQL": pred, "T_in": n_in, "T_out": n_out,
             # Nilai ESM, EX, dan SLA kolom 1-Stage
             "ESM": evaluation.esm, "EX": evaluation.ex, "SLA column F1": sla.f1}
    # Tampilkan perbandingan kedua varian
    show(rt.with_formats(pd.DataFrame([row_1, row_2]), {"SLA column F1": "pct"}), "Contoh satu pertanyaan: 1-Stage vs 2-Stage")


# ---------------------------------------------------------------------------
# Bagian 11: loop utama
# ---------------------------------------------------------------------------

# Menjalankan 2-Stage untuk satu pertanyaan
def _run_2stage(tools: Tools, item: dict, i: int, results: RunResults) -> dict:
    # Coba jalankan
    try:
        # Terima SQL, proxy, schema, kolom terpilih, dan jumlah token
        pred, recall, precision, _schema, nodes, n_in, n_out = run_single(
            # Input pertanyaan, gold SQL, dan database
            item["question"], item["query"], item["db_id"],
            # Graph, model, dan konfigurasi
            tools.col_graph, tools.embed_model, tools.llm, tools.tokenizer, tools.cfg,
            # Index yang sudah dibuat
            schema_index=tools.col_cache[item["db_id"]], few_shot_index=tools.few_shot_idx,
        )
    # Kalau generate gagal, catat errornya dan lanjut
    except Exception as e:
        # Catat varian, posisi, database, dan pesan error
        results.gen_errors.append(("graphrag", i, item["db_id"], repr(e)))
        # Pakai SQL pengganti supaya run tidak berhenti
        pred, recall, precision, nodes, n_in, n_out = "SELECT 1", 0.0, 0.0, [], 0, 0
    # Kembalikan hasilnya sebagai dictionary
    return {"pred": pred, "recall": recall, "precision": precision, "nodes": nodes, "n_in": n_in, "n_out": n_out}


# Menjalankan 1-Stage untuk satu pertanyaan
def _run_1stage(tools: Tools, item: dict, i: int, results: RunResults) -> dict:
    # Coba jalankan
    try:
        # Terima SQL, proxy, tabel terpilih, dan jumlah token
        pred, recall, precision, nodes, n_in, n_out = run_single_baseline(
            # Input pertanyaan, gold SQL, dan database
            item["question"], item["query"], item["db_id"],
            # Graph, model, dan konfigurasi
            tools.tbl_graph, tools.embed_model, tools.llm, tools.tokenizer, tools.cfg,
            # Index yang sudah dibuat
            table_index=tools.tbl_cache[item["db_id"]], few_shot_index=tools.few_shot_idx,
        )
    # Kalau generate gagal, catat errornya dan lanjut
    except Exception as e:
        # Catat varian, posisi, database, dan pesan error
        results.gen_errors.append(("baseline", i, item["db_id"], repr(e)))
        # Pakai SQL pengganti supaya run tidak berhenti
        pred, recall, precision, nodes, n_in, n_out = "SELECT 1", 0.0, 0.0, [], 0, 0
    # Kembalikan hasilnya sebagai dictionary
    return {"pred": pred, "recall": recall, "precision": precision, "nodes": nodes, "n_in": n_in, "n_out": n_out}


# Menyusun satu catatan raw_logs
def _log_entry(query_id: str, item: dict, gold_schema: list, predicted_schema: list, side: dict, evaluation) -> dict:
    # Isinya mengikuti skema di EVALUATION_ANALYSIS_GUIDE.md Bagian 2
    return {
        # Identitas pertanyaan
        "query_id": query_id, "db_id": item["db_id"], "difficulty": evaluation.difficulty,
        # SQL gold dan prediksi
        "gold_sql": item["query"], "predicted_sql": side["pred"],
        # Kolom gold
        "gold_schema": gold_schema,
        # Kolom yang dikirim ke LLM
        "predicted_schema": predicted_schema,
        # Jumlah token input dan output
        "token_input": side["n_in"], "token_output": side["n_out"],
        # Hasil ESM dan EX
        "esm_result": evaluation.esm, "ex_result": evaluation.ex,
        # Hasil CM per klausa
        "cm_per_clause": evaluation.cm_per_clause,
    }


# Mengevaluasi kedua prediksi dan menyimpan catatannya
def _evaluate_and_log(tools: Tools, item: dict, i: int, two: dict, one: dict, results: RunResults) -> None:
    # Coba evaluasi; kalau gagal, prediksi tetap tersimpan
    try:
        # Ambil database dan gold SQL
        db_id, gold_sql = item["db_id"], item["query"]
        # Ambil kolom yang dipakai gold SQL
        gold_schema = tools.evaluator.gold_columns(gold_sql, db_id)
        # Evaluasi prediksi 2-Stage
        eval_2 = tools.evaluator.evaluate(two["pred"], gold_sql, db_id)
        # Evaluasi prediksi 1-Stage
        eval_1 = tools.evaluator.evaluate(one["pred"], gold_sql, db_id)
        # Perbarui penghitung confound; ini tidak mengubah nilai metrik
        _update_confound_diag(results.confound_diag, tools.evaluator.db_path(db_id), gold_sql, two["pred"], eval_2, one["pred"], eval_1)
        # Buat ID pertanyaan dari posisinya
        query_id = f"dev_{i + 1:04d}"
        # Ubah kolom terpilih 2-Stage menjadi "tabel.kolom"
        schema_2 = _column_nodes_to_predicted_schema(tools.col_graph, two["nodes"])
        # Ubah tabel terpilih 1-Stage menjadi "tabel.kolom"
        schema_1 = _table_nodes_to_predicted_schema(tools.tbl_graph, one["nodes"])
        # Simpan catatan 2-Stage
        results.graphrag_logs.append(_log_entry(query_id, item, gold_schema, schema_2, two, eval_2))
        # Simpan catatan 1-Stage
        results.baseline_logs.append(_log_entry(query_id, item, gold_schema, schema_1, one, eval_1))
    # Catat error evaluasi
    except Exception as e:
        # Simpan posisi, database, dan pesan error
        results.eval_errors.append((i, item["db_id"], repr(e)))


# Menjalankan kedua varian untuk semua pertanyaan
def run_all_questions(tools: Tools, questions: list) -> RunResults:
    # Siapkan wadah hasil
    results = RunResults(confound_diag=_new_confound_diag())
    # Proses tiap pertanyaan dengan progress bar
    for i, item in enumerate(tqdm(questions, desc="1-Stage + 2-Stage")):
        # Jalankan 2-Stage
        two = _run_2stage(tools, item, i, results)
        # Jalankan 1-Stage
        one = _run_1stage(tools, item, i, results)
        # Simpan hasil ringkas 2-Stage
        results.graphrag_results.append(PipelineResult(index=i + 1, db_id=item["db_id"], question=item["question"], gold_sql=item["query"],
                                                       # SQL prediksi dan nilai proxy
                                                       pred_sql=two["pred"], recall=two["recall"], precision=two["precision"]))
        # Simpan hasil ringkas 1-Stage
        results.baseline_results.append(BaselineResult(index=i + 1, db_id=item["db_id"], question=item["question"], gold_sql=item["query"],
                                                       # SQL prediksi dan nilai proxy
                                                       pred_sql=one["pred"], recall=one["recall"], precision=one["precision"]))
        # Simpan teks SQL 2-Stage
        results.graphrag_preds.append(two["pred"].strip())
        # Simpan teks SQL 1-Stage
        results.baseline_preds.append(one["pred"].strip())
        # Evaluasi dan simpan catatan lengkap
        _evaluate_and_log(tools, item, i, two, one, results)
        # Tampilkan progres pada pertanyaan pertama, tiap 20 pertanyaan, dan pertanyaan terakhir
        if i == 0 or (i + 1) % 20 == 0 or (i + 1) == len(questions):
            # Nomor pertanyaan dan database
            print(f"\n[{i + 1:>4}/{len(questions)}] DB: {item['db_id']}")
            # Pertanyaan
            print(f"  Q       : {item['question']}")
            # Gold SQL
            print(f"  Gold    : {item['query']}")
            # Prediksi 2-Stage
            print(f"  2-Stage : {two['pred']}")
            # Prediksi 1-Stage
            print(f"  1-Stage : {one['pred']}")
    # Kembalikan semua hasil
    return results


# Menampilkan ringkasan loop utama
def show_loop_summary(results: RunResults) -> None:
    # Susun jumlah pertanyaan, catatan, dan error
    rows = [
        # Jumlah pertanyaan yang diproses
        {"Item": "Samples processed", "Value": len(results.graphrag_preds)},
        # Jumlah catatan raw_logs
        {"Item": "raw_logs entries (per pipeline)", "Value": len(results.graphrag_logs)},
        # Jumlah error saat generate
        {"Item": "Generation errors", "Value": len(results.gen_errors)},
        # Jumlah error saat evaluasi
        {"Item": "Evaluation errors", "Value": len(results.eval_errors)},
    ]
    # Tampilkan ringkasannya
    show(pd.DataFrame(rows), "Main loop")
    # Tampilkan rincian error generate kalau ada
    for pipeline, i, db_id, error in results.gen_errors:
        # Satu baris per error
        print(f"[ERROR generate] {pipeline} | posisi {i} | {db_id} | {error}")
    # Tampilkan rincian error evaluasi kalau ada
    for i, db_id, error in results.eval_errors:
        # Satu baris per error
        print(f"[ERROR evaluasi] posisi {i} | {db_id} | {error}")


# ---------------------------------------------------------------------------
# Bagian 12: simpan prediksi dan raw_logs (input metrik, dan pengaman hasil generate)
# ---------------------------------------------------------------------------

# Menentukan lokasi file prediksi kedua varian
def prediction_paths(artifact_dir: Path) -> dict:
    # graphrag = 2-Stage, baseline = 1-Stage
    return {"GraphRAG": artifact_dir / "predictions" / "predictions.txt",
            # Lokasi file prediksi 1-Stage
            "Baseline": artifact_dir / "predictions" / "baseline_predictions.txt"}


# Menyimpan prediksi dan raw_logs supaya hasil generate aman kalau langkah berikutnya gagal
def save_generation_results(results: RunResults, artifact_dir: Path, raw_logs_dir: Path) -> None:
    # Ambil lokasi file prediksi
    paths = prediction_paths(artifact_dir)
    # Tulis prediksi 2-Stage, satu SQL per baris
    paths["GraphRAG"].write_text("\n".join(results.graphrag_preds) + "\n", encoding="utf-8")
    # Tulis prediksi 1-Stage, satu SQL per baris
    paths["Baseline"].write_text("\n".join(results.baseline_preds) + "\n", encoding="utf-8")
    # Buka file raw_logs 2-Stage
    with open(raw_logs_dir / "graphrag_log.json", "w", encoding="utf-8") as f:
        # Simpan catatan per pertanyaan
        json.dump(results.graphrag_logs, f, ensure_ascii=False, indent=2)
    # Buka file raw_logs 1-Stage
    with open(raw_logs_dir / "baseline_log.json", "w", encoding="utf-8") as f:
        # Simpan catatan per pertanyaan
        json.dump(results.baseline_logs, f, ensure_ascii=False, indent=2)
    # Proses kedua file prediksi
    for label, path in paths.items():
        # Hitung baris yang tidak kosong
        n_lines = len([line for line in path.read_text(encoding="utf-8").split("\n") if line.strip()])
        # Evaluator resmi mencocokkan per baris, jadi jumlahnya harus sama dengan jumlah pertanyaan
        print(f"{label}: {n_lines} baris prediksi dari {len(results.graphrag_preds)} pertanyaan -> {path}")
    # Tampilkan lokasi raw_logs
    print(f"raw_logs -> {raw_logs_dir}/ ({len(results.graphrag_logs)} graphrag, {len(results.baseline_logs)} baseline)")


# ---------------------------------------------------------------------------
# Bagian 13: tampilan metrik
# ---------------------------------------------------------------------------

# Menampilkan semua tabel dari satu metrik
def show_metric(metrics_results: dict, metric_name: str) -> None:
    # Proses keenam metrik
    for sheet_name, title, tables in rt.metric_tables(metrics_results):
        # Lewati metrik lain
        if sheet_name != metric_name:
            # Lanjut ke metrik berikutnya
            continue
        # Tampilkan judul metrik
        print(title)
        # Proses tiap tabel metrik itu
        for table in tables:
            # Tampilkan tabelnya
            show(table)


# Menampilkan tabel pendukung
def show_diagnostics(results: RunResults, alpha: float) -> None:
    # Tampilkan EX, ESM, dan token per tingkat kesulitan
    show(rt.by_difficulty(results.baseline_logs, results.graphrag_logs, alpha))
    # Tampilkan pengecekan prediksi: fallback, kosong, dan identik dengan gold
    show(rt.prediction_checks(results.baseline_logs, results.graphrag_logs))
    # Tampilkan jumlah query per kombinasi EX dan ESM
    show(rt.outcome_matrix(results.baseline_logs, results.graphrag_logs))
    # Tampilkan perbandingan EX per query antar varian
    show(rt.paired_ex(results.baseline_logs, results.graphrag_logs))
    # Tampilkan pemakaian fitur SQL pada prediksi
    show(rt.sql_feature_usage(results.baseline_logs, results.graphrag_logs))
    # Tampilkan confound diagnostics
    show(rt.confound_table(results.confound_diag))


# ---------------------------------------------------------------------------
# Bagian 14: evaluator resmi Spider
# ---------------------------------------------------------------------------

# Menjalankan evaluation.py resmi untuk satu jenis evaluasi
def _run_spider_once(cfg, label: str, etype: str, gold_file: Path, pred_file: Path) -> pd.DataFrame:
    # Panggil evaluation.py sebagai program terpisah
    r = subprocess.run(
        # Program yang dijalankan
        [sys.executable, "external/spider_eval/evaluation.py",
         # File gold dan prediksi
         "--gold", str(gold_file), "--pred", str(pred_file),
         # Folder database, schema, dan jenis evaluasi
         "--db", str(cfg.db_dir), "--table", str(cfg.tables_json), "--etype", etype],
        # Tangkap outputnya sebagai teks
        capture_output=True, text=True,
    )
    # Ubah output teks menjadi tabel
    parsed = rt.parse_spider_output(r.stdout)
    # Beri peringatan kalau evaluator gagal atau skornya tidak terbaca
    if r.returncode != 0 or parsed.empty:
        # Tampilkan varian, jenis evaluasi, dan kode keluar
        print(f"[WARN] {label} --etype {etype}: exit code {r.returncode}, {len(parsed)} baris skor terbaca")
        # Tampilkan akhir output untuk membantu mencari penyebabnya
        print(r.stdout[-1500:] or r.stderr[-1500:])
    # Kembalikan tabel skor
    return parsed


# Menjalankan evaluator resmi (ESM dan EX) untuk satu varian
def _run_spider_for(cfg, label: str, pred_path: Path, gold_path: Path, tmp: Path) -> pd.DataFrame:
    # Ambil nama file prediksi tanpa ekstensi
    stem = pred_path.stem
    # ESM: semua jenis JOIN boleh disamakan karena SQL tidak dieksekusi
    match_gold = normalize_sql_file_for_parsing(gold_path, tmp / f"{stem}_match_gold.txt", has_db_id_suffix=True, execution_safe_only=False)
    # File prediksi untuk ESM
    match_pred = normalize_sql_file_for_parsing(pred_path, tmp / f"{stem}_match_pred.txt", has_db_id_suffix=False, execution_safe_only=False)
    # EX: hanya INNER dan CROSS JOIN yang disamakan karena SQL dieksekusi
    exec_gold = normalize_sql_file_for_parsing(gold_path, tmp / f"{stem}_exec_gold.txt", has_db_id_suffix=True, execution_safe_only=True)
    # File prediksi untuk EX
    exec_pred = normalize_sql_file_for_parsing(pred_path, tmp / f"{stem}_exec_pred.txt", has_db_id_suffix=False, execution_safe_only=True)
    # Jalankan evaluasi ESM
    match_scores = _run_spider_once(cfg, label, "match", match_gold, match_pred)
    # Jalankan evaluasi EX
    exec_scores = _run_spider_once(cfg, label, "exec", exec_gold, exec_pred)
    # Gabungkan kedua skor
    return pd.concat([match_scores, exec_scores], ignore_index=True)


# Menjalankan cross-check evaluator resmi untuk kedua varian dan menampilkan hasilnya
def run_spider_check(cfg, artifact_dir: Path, gold_path: Path) -> dict:
    # Folder file sementara
    tmp = artifact_dir / "spider_eval_tmp"
    # Buat foldernya kalau belum ada
    tmp.mkdir(parents=True, exist_ok=True)
    # Tempat skor tiap varian
    scores = {}
    # Proses kedua file prediksi
    for label, pred_path in prediction_paths(artifact_dir).items():
        # Jalankan evaluator resmi
        scores[label] = _run_spider_for(cfg, label, pred_path, gold_path, tmp)
    # Tampilkan ringkasan skor resmi
    show(rt.spider_summary(scores))
    # Kembalikan skornya untuk disimpan di akhir
    return scores


# ---------------------------------------------------------------------------
# Bagian 15: dimensi
# ---------------------------------------------------------------------------

# Membaca EX per k dari hasil ablation.py kalau filenya ada
def read_ex_per_k(cfg) -> dict | None:
    # Nama file bergantung pada cara seleksi few-shot
    tag = "graphrag" if cfg.few_shot_selection == "question" else f"graphrag_{cfg.few_shot_selection}"
    # Lokasi file hasil ablation
    path = Path(f"outputs/tables/ablation_ex_per_k_{tag}.json")
    # Kalau file tidak ada, Dimensi 6 akan dilewati
    if not path.exists():
        # Tampilkan keterangannya
        print(f"Hasil ablation belum ada: {path}")
        # Tidak ada data
        return None
    # Ubah isi file menjadi {k: EX}
    ex_per_k = {int(k): v for k, v in json.loads(path.read_text()).items()}
    # Tampilkan nilai yang terbaca
    print(f"EX_PER_K dari {path}: {ex_per_k}")
    # Kembalikan datanya
    return ex_per_k


# Menampilkan status dan tabel semua dimensi
def show_dimensions(dim_results: dict) -> None:
    # Tampilkan status tiap dimensi
    show(rt.dim_status(dim_results))
    # Proses Dimensi 1 sampai 6
    for n, name in rt.DIM_NAMES.items():
        # Ubah hasil dimensi itu menjadi tabel
        tables = rt.DIM_TABLES[n](dim_results)
        # Tampilkan judul dimensi
        print(f"\nDimensi {n}: {name}" + ("" if tables else " (tidak ada tabel, lihat status di atas)"))
        # Proses tiap tabel dimensi itu
        for table in tables:
            # Tampilkan tabelnya
            show(table)


# ---------------------------------------------------------------------------
# Bagian 16: simpan semua hasil
# ---------------------------------------------------------------------------

# Menulis laporan teks dan penghitung confound diagnostics
def _save_comparison_report(results: RunResults, tables_dir: Path) -> None:
    # Hitung recall proxy 2-Stage dalam persen
    g_r = np.mean([r.recall for r in results.graphrag_results]) * 100
    # Hitung precision proxy 2-Stage dalam persen
    g_p = np.mean([r.precision for r in results.graphrag_results]) * 100
    # Hitung recall proxy 1-Stage dalam persen
    b_r = np.mean([r.recall for r in results.baseline_results]) * 100
    # Hitung precision proxy 1-Stage dalam persen
    b_p = np.mean([r.precision for r in results.baseline_results]) * 100
    # Susun isi laporan
    lines = [
        # Baris kosong dan garis pembuka
        "", "=" * 70,
        # Judul laporan dan garis
        "  COMPARISON REPORT — GraphRAG (column/node) vs Baseline (table/node)", "=" * 70,
        # Nama kolom dan garis
        f"  {'Metric':<26} {'GraphRAG':>10}  {'Baseline':>10}  {'Delta (G-B)':>12}", "-" * 70,
        # Baris recall
        f"  {'Recall (proxy)':<26} {g_r:>9.2f}%  {b_r:>9.2f}%  {g_r-b_r:>+11.2f}%",
        # Baris precision
        f"  {'Precision (proxy)':<26} {g_p:>9.2f}%  {b_p:>9.2f}%  {g_p-b_p:>+11.2f}%",
        # Garis pemisah
        "-" * 70,
        # Catatan bahwa nilai ini bukan SLA
        "  (proxy = name-matching internal; SLA resmi dihitung dari raw_logs via sla.py)",
        # Garis penutup dan baris kosong
        "=" * 70, "",
    ]
    # Simpan laporan ditambah blok confound diagnostics
    (tables_dir / "comparison_report.txt").write_text("\n".join(lines) + _format_confound_diag(results.confound_diag), encoding="utf-8")
    # Simpan penghitung confound sebagai JSON
    (tables_dir / "confound_diagnostics.json").write_text(json.dumps(results.confound_diag, indent=2), encoding="utf-8")


# Menulis skor evaluator resmi ke CSV
def _save_spider_scores(spider_scores: dict, tables_dir: Path) -> None:
    # Tambahkan nama varian pada tiap tabel yang berisi skor
    frames = [df.assign(Pipeline=label) for label, df in spider_scores.items() if not df.empty]
    # Kalau tidak ada skor, tidak ada yang disimpan
    if not frames:
        # Keluar dari fungsi
        return
    # Gabungkan semua skor menjadi satu tabel
    combined = pd.concat(frames, ignore_index=True)
    # Pindahkan kolom Pipeline ke paling depan
    combined = combined[["Pipeline"] + [c for c in combined.columns if c != "Pipeline"]]
    # Simpan ke CSV
    combined.to_csv(tables_dir / "spider_official_eval.csv", index=False)


# Membuat zip berisi seluruh hasil model untuk diunduh dari Kaggle
def _make_kaggle_download(model: str, xlsx: Path, artifact_dir: Path) -> None:
    # Di luar Kaggle, langkah ini dilewati
    if not Path("/kaggle/working").exists():
        # Keluar dari fungsi
        return
    # Salin Excel supaya mudah diunduh
    shutil.copy(xlsx, f"/kaggle/working/hasil_skripsi_{model}.xlsx")
    # Buat zip berisi seluruh hasil model ini
    zip_path = shutil.make_archive(f"/kaggle/working/hasil_{model}", "zip", root_dir=artifact_dir.parent, base_dir=artifact_dir.name)
    # Tampilkan file yang bisa diunduh
    print(f"Unduh dari /kaggle/working: hasil_skripsi_{model}.xlsx dan {Path(zip_path).name}")


# Menyimpan semua hasil ke JSON dan Excel, setelah semuanya ditampilkan
def save_all_results(model: str, results: RunResults, metrics_results: dict, spider_scores: dict,
                     # Hasil dimensi dan folder tujuan
                     dim_results: dict, dim_report: str, artifact_dir: Path, raw_logs_dir: Path) -> Path:
    # Folder tabel model ini
    tables_dir = artifact_dir / "tables"
    # Simpan keenam metrik ke JSON
    print(f"Tersimpan: {save_metrics_results(metrics_results, tables_dir)}")
    # Simpan hasil dimensi ke JSON dan teks
    save_dimension_outputs(dim_results, dim_report, tables_dir)
    # Simpan laporan teks dan confound diagnostics
    _save_comparison_report(results, tables_dir)
    # Simpan skor evaluator resmi
    _save_spider_scores(spider_scores, tables_dir)
    # Buat satu file Excel dari semua hasil
    xlsx = build_workbook(raw_logs_dir, tables_dir, tables_dir / "hasil_skripsi.xlsx", skip_missing=True)
    # Tampilkan lokasi Excel
    print(f"Tersimpan: {xlsx}")
    # Buka Excel untuk mengambil daftar sheet
    workbook = load_workbook(xlsx)
    # Tampilkan daftar sheet
    print(f"Sheet: {', '.join(workbook.sheetnames)}")
    # Tutup file Excel
    workbook.close()
    # Lokasi salinan raw_logs di dalam folder hasil
    raw_copy = artifact_dir / "raw_logs"
    # Pada run penuh, raw_logs ada di folder lain, jadi perlu disalin
    if raw_logs_dir != raw_copy:
        # Salin raw_logs supaya ikut masuk zip
        shutil.copytree(raw_logs_dir, raw_copy, dirs_exist_ok=True)
    # Siapkan file unduhan kalau berjalan di Kaggle
    _make_kaggle_download(model, xlsx, artifact_dir)
    # Kembalikan lokasi Excel
    return xlsx
