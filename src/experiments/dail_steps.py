"""
dail_steps.py — langkah-langkah notebooks/eval_dail.ipynb (baseline DAIL-SQL), supaya sel notebook pendek.

Evaluasinya sengaja memakai fungsi yang SAMA dengan eval_pipeline.ipynb (notebook_steps.py,
metrics_summary.py, result_tables.py, timing.py), jadi angka DAIL-SQL sebanding apple-to-apple
dengan GraphRAG 1-Stage/2-Stage:
- sampel dev set, gold file, model LLM, evaluator ESM/EX/CM, kolom gold: notebook_steps.py
- satu catatan raw_logs per pertanyaan: notebook_steps._log_entry() (skema yang sama) -> dail_log.json
- SLA, ESM, EX, CM, Token, QVT: fungsi agregasi metrics_summary.py; TEP: rumus yang sama dengan DAIL-SQL
  sebagai B (pembanding), hanya kalau raw_logs 1-Stage/2-Stage model yang sama sudah ada
- latensi: nama tahap yang sama dengan pipeline kita (timing.py), ditambah total per pass DAIL

Tidak ada metrik atau formula baru. File eval_pipeline.ipynb dan modul di atas tidak diubah;
hasil DAIL ditulis ke file terpisah (dail_*) supaya tidak menimpa hasil 1-Stage/2-Stage.
Lihat context/IMPLEMENTATION_DECISIONS.md poin 43.
"""

# Membuat wadah data sederhana
from dataclasses import dataclass, field
# json untuk membaca dan menulis file JSON
import json
# shutil untuk menyalin file dan membuat zip
import shutil
# time untuk mengukur waktu menulis Excel dan zip
import time
# Counter untuk menghitung kolom per tabel
from collections import Counter
# Path untuk menulis lokasi file
from pathlib import Path
# Any untuk menandai data yang jenisnya bebas
from typing import Any, Optional

# pandas untuk membuat tabel
import pandas as pd
# openpyxl untuk membuat file Excel
from openpyxl import Workbook
# Progress bar
from tqdm.auto import tqdm

# Langkah eval_pipeline yang dipakai ulang apa adanya
from src.experiments import notebook_steps as steps
# Agregasi keenam metrik, sama dengan metrics_results.json 1-Stage/2-Stage
from src.experiments import metrics_summary as ms
# Fungsi tabel dan format angka, sama dengan notebook dan Excel utama
from src.experiments import result_tables as rt
# Fungsi menampilkan tabel di notebook
from src.experiments.result_tables import show
# Penulis sheet Excel yang sama dengan hasil_skripsi.xlsx
from src.experiments.export_excel import Sheet
# Pembentuk grup parafrase QVT, sama dengan build_qvt_variations.py
from src.experiments.build_qvt_variations import build_groups, join_with_raw_logs
# Fungsi memuat LLM, sama dengan pipeline kita
from src.generation.generation import load_model_and_tokenizer
# Wrapper DAIL-SQL
from src.retrieval import dail as dail_mod
# Pencatat latensi per tahap
from src.utils import timing
# Fungsi menghitung ukuran schema di prompt
from src.utils.prompt_stats import prompt_stats

# Kunci kondisi -> label tampilan. DAIL-SQL selalu kolom pertama karena ia pembanding (B)
LABELS = {"dail": "DAIL-SQL", "baseline": "1-stage", "graphrag": "2-stage"}
# Nama file raw_logs DAIL, di folder raw_logs model yang sama dengan 1-Stage/2-Stage
DAIL_LOG = "dail_log.json"
# Nama file hasil metrik DAIL
DAIL_METRICS = "dail_metrics_results.json"
# Tahap latensi tambahan khusus DAIL (keduanya sudah termasuk di dalam end-to-end)
DAIL_LATENCY_STEPS = {"dail_pass1": "DAIL pass 1: preliminary SQL (part of end-to-end)",
                      "dail_pass2": "DAIL pass 2: final SQL (part of end-to-end)"}
# Semua tahap latensi yang ditampilkan: urutan pipeline kita lalu tahap DAIL
LATENCY_STEPS = {**rt.LATENCY_STEPS, **DAIL_LATENCY_STEPS}


# ---------------------------------------------------------------------------
# Wadah data
# ---------------------------------------------------------------------------

# Menyatukan semua alat yang dipakai berulang
@dataclass
class DailTools:
    # Konfigurasi run
    cfg: Any
    # Objek DAIL yang sudah disiapkan (dataset, linking, dua objek prompt)
    dail: Any
    # Model LLM
    llm: Any
    # Tokenizer LLM
    tokenizer: Any
    # Alat evaluasi SQL, sama dengan eval_pipeline
    evaluator: Any
    # Jumlah kolom tiap (db, tabel), untuk prompt_stats
    table_cols: Counter


# Menyatukan semua hasil loop utama
@dataclass
class DailResults:
    # Catatan raw_logs per pertanyaan
    logs: list = field(default_factory=list)
    # Teks SQL final per pertanyaan, urut sesuai sampel
    preds: list = field(default_factory=list)
    # Prompt dan output mentah kedua pass, untuk audit
    traces: list = field(default_factory=list)
    # Daftar error saat generate
    gen_errors: list = field(default_factory=list)
    # Daftar error saat evaluasi
    eval_errors: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Persiapan
# ---------------------------------------------------------------------------

# Menampilkan konfigurasi DAIL dalam satu tabel
def show_run_config(cfg, sample_ratio: float, artifact_dir: Path, raw_logs_dir: Path) -> None:
    # Susun pasangan nama dan nilai
    rows = [
        # Lokasi dataset
        ("Data path", str(cfg.data_path)),
        # Lokasi kode DAIL dan hasil preprocessing-nya
        ("DAIL code / preprocessed data", f"{cfg.dail_dir} / {cfg.dail_data_dir}"),
        # LLM yang menggantikan GPT-4
        ("LLM model (both passes)", cfg.llm_model),
        # Representasi schema dan organisasi contoh
        ("Prompt representation / example organization", f"{cfg.dail_prompt_repr} / {cfg.dail_example_type}"),
        # Selector pass 1 dan pass 2
        ("Selector pass 1 / pass 2", f"{cfg.dail_pre_selector} / {cfg.dail_selector}"),
        # Jumlah contoh
        ("k (examples)", cfg.dail_k_shot),
        # Batas generate dan stop string
        ("max_new_tokens / stop strings", f"{cfg.max_new_tokens} / {cfg.dail_stop_strings}"),
        # Bobot token output pada rumus T
        ("α (output-token weight)", cfg.token_output_weight),
        # Jenis run
        ("SAMPLE_RATIO", f"{sample_ratio} ({'full run' if sample_ratio >= 1.0 else 'trial, outputs/trial/'})"),
        # Folder hasil
        ("Results directory", str(artifact_dir)),
        # Folder raw_logs
        ("raw_logs directory", str(raw_logs_dir)),
    ]
    # Tampilkan sebagai tabel teks
    show(pd.DataFrame(rows, columns=["Parameter", "Value"]).astype({"Value": str}), "DAIL-SQL run configuration")


# Memuat LLM saja (DAIL tidak memakai BGE-M3)
def load_llm(cfg) -> tuple:
    # Tampilkan LLM yang dimuat
    print(f"Loading LLM: {cfg.llm_model} (4-bit) ...")
    # Muat LLM dan tokenizer dengan fungsi yang sama seperti pipeline kita, dan catat waktunya
    with timing.run_step("load_llm"):
        llm, tokenizer = load_model_and_tokenizer(cfg)
    # Tanda model siap
    print("Model siap.")
    # Kembalikan keduanya
    return llm, tokenizer


# Menyiapkan DAIL: cek hasil preprocessing, baca linking, buat index contoh kedua pass
def load_dail(cfg):
    # Cek jumlah baris linking dulu supaya kesalahan terlihat sebelum langkah yang lama
    counts = dail_mod.check_preprocessed(cfg)
    # Tampilkan jumlahnya
    print(f"Preprocessing DAIL OK: {counts}")
    # Siapkan dataset dan kedua objek prompt, dan catat total waktunya
    with timing.run_step("dail_setup"):
        dail = dail_mod.load_dail(cfg)
    # Tampilkan ukuran pool contoh dan jumlah pertanyaan dev
    print(f"DAIL siap: {len(dail.pre_prompt.train_json)} contoh train, {len(dail.targets)} pertanyaan dev unik")
    # Kembalikan objek DAIL
    return dail


# Menghitung kolom per tabel dari db_schema evaluator (sama isinya dengan columns_per_table() dari graph)
def columns_per_table(evaluator) -> Counter:
    # Satu hitungan per (db, tabel); kolom sudah huruf kecil dan tanpa "*"
    return Counter((db, column.split(".", 1)[0]) for db, columns in evaluator.db_schema.items() for column in set(columns))


# ---------------------------------------------------------------------------
# Contoh satu pertanyaan
# ---------------------------------------------------------------------------

# Menampilkan kedua pass DAIL untuk satu pertanyaan; hasilnya tidak masuk hasil akhir
def walk_through(tools: DailTools, item: dict, show_prompts: bool = True) -> dict:
    # Tampilkan database, pertanyaan, dan gold SQL
    print(f"DB       : {item['db_id']}\nQuestion : {item['question']}\nGold SQL : {item['query']}")
    # Jalankan kedua pass
    out = dail_mod.run_single_dail(item, tools.dail, tools.llm, tools.tokenizer)
    # k = 0 hanya punya satu pass (tanpa SQL awal)
    if "pass1" in out["prompts"]:
        # Tampilkan prompt pass 1 kalau diminta
        if show_prompts:
            # Prompt pass 1 lengkap: schema seluruh DB, k contoh, pertanyaan
            print(f"\n[Prompt pass 1]\n{out['prompts']['pass1']}")
        # Tampilkan SQL awal dan skeleton-nya
        print(f"\n[Pass 1] SQL awal : {out['pre_sql']}\n[Pass 1] skeleton : {out['pre_skeleton']}")
    # Tampilkan prompt pass 2 kalau diminta
    if show_prompts:
        # Prompt pass 2: contoh dipilih ulang dengan DAIL Selection
        print(f"\n[Prompt pass 2]\n{out['prompts']['pass2']}")
    # Hitung ESM, EX, dan CM
    evaluation = tools.evaluator.evaluate(out["pred"], item["query"], item["db_id"])
    # Tampilkan hasil akhir
    print(f"\n[Pass 2] SQL final: {out['pred']}")
    # Tampilkan token per pass dan hasil evaluasi
    print(f"Token per pass: {out['passes']} | T_in={out['n_in']} T_out={out['n_out']} | "
          f"ESM={evaluation.esm} EX={evaluation.ex} | Difficulty={evaluation.difficulty}")
    # Kembalikan hasilnya
    return out


# ---------------------------------------------------------------------------
# Loop utama
# ---------------------------------------------------------------------------

# Menjalankan DAIL untuk satu pertanyaan, dengan latensi per tahap
def _run_one(tools: DailTools, item: dict, i: int, results: DailResults) -> dict:
    # Coba jalankan
    try:
        # Kumpulkan waktu tiap tahap untuk pertanyaan ini
        with timing.collect() as latency:
            # Pass 1 lalu pass 2
            out = dail_mod.run_single_dail(item, tools.dail, tools.llm, tools.tokenizer)
        # Rapikan latensi: build_prompt tanpa build_ddl/few_shot, lalu total retrieval dan end-to-end
        out["latency"] = dail_mod.finish_latency(latency)
    # Kalau generate gagal, catat errornya dan lanjut
    except Exception as e:  # noqa: BLE001 - sama dengan eval_pipeline: run tidak berhenti
        # Catat posisi, database, dan pesan error
        results.gen_errors.append(("dail", i, item["db_id"], repr(e)))
        # SQL pengganti yang sama dengan pipeline kita; latensi tidak dicatat
        out = {"pred": "SELECT 1", "n_in": 0, "n_out": 0, "latency": None, "pre_sql": None, "pre_skeleton": None,
               "pre_skeleton_error": None, "passes": None, "prompts": None, "raw": None}
    # Kembalikan hasilnya
    return out


# Mengevaluasi satu prediksi DAIL dan menyimpan catatannya
def _evaluate_and_log(tools: DailTools, item: dict, i: int, out: dict, results: DailResults) -> None:
    # Coba evaluasi; kalau gagal, prediksi tetap tersimpan
    try:
        # Ambil database dan gold SQL
        db_id, gold_sql = item["db_id"], item["query"]
        # Kolom yang dipakai gold SQL
        gold_schema = tools.evaluator.gold_columns(gold_sql, db_id)
        # Evaluasi, dan catat waktunya
        with timing.collect() as eval_time, timing.step("evaluation"):
            evaluation = tools.evaluator.evaluate(out["pred"], gold_sql, db_id)
        # DAIL mengirim seluruh schema DB: semua kolom tables.json, format "tabel.kolom"
        predicted_schema = sorted(tools.evaluator.db_schema[db_id])
        # Token schema dan jumlah kolom, dengan blok CREATE TABLE yang benar-benar masuk prompt DAIL
        stats = prompt_stats(predicted_schema, db_id, tools.table_cols, dail_mod.schema_context(tools.dail, db_id), tools.tokenizer)
        # ID pertanyaan dari posisinya, sama dengan eval_pipeline
        query_id = f"dev_{i + 1:04d}"
        # Catatan dengan skema raw_logs yang sama persis dengan 1-Stage/2-Stage
        entry = steps._log_entry(query_id, item, gold_schema, predicted_schema, stats, out, evaluation, eval_time)
        # Rincian khusus DAIL: SQL awal, skeleton, dan token per pass (tidak dibaca metrik)
        entry["dail"] = {k: out[k] for k in ("pre_sql", "pre_skeleton", "pre_skeleton_error", "passes")}
        # Simpan catatannya
        results.logs.append(entry)
    # Catat error evaluasi
    except Exception as e:  # noqa: BLE001
        # Simpan posisi, database, dan pesan error
        results.eval_errors.append((i, item["db_id"], repr(e)))


# Menjalankan DAIL untuk semua pertanyaan, dan mencatat total waktunya
@timing.run_step("main_loop_dail")
def run_all_questions(tools: DailTools, questions: list) -> DailResults:
    # Siapkan wadah hasil
    results = DailResults()
    # Proses tiap pertanyaan dengan progress bar
    for i, item in enumerate(tqdm(questions, desc="DAIL-SQL")):
        # Jalankan kedua pass
        out = _run_one(tools, item, i, results)
        # Simpan teks SQL final
        results.preds.append(out["pred"].strip())
        # Simpan prompt dan output mentah untuk audit
        results.traces.append({"query_id": f"dev_{i + 1:04d}", "db_id": item["db_id"], "question": item["question"],
                               "prompts": out["prompts"], "raw": out["raw"], "pre_sql": out["pre_sql"], "pred": out["pred"]})
        # Evaluasi dan simpan catatan lengkap
        _evaluate_and_log(tools, item, i, out, results)
        # Tampilkan progres pada pertanyaan pertama, tiap 20 pertanyaan, dan pertanyaan terakhir
        if i == 0 or (i + 1) % 20 == 0 or (i + 1) == len(questions):
            # Nomor, database, pertanyaan, gold, SQL awal, dan SQL final
            print(f"\n[{i + 1:>4}/{len(questions)}] DB: {item['db_id']}\n  Q       : {item['question']}\n"
                  f"  Gold    : {item['query']}\n  Pass 1  : {out['pre_sql']}\n  DAIL-SQL: {out['pred']}")
    # Kembalikan semua hasil
    return results


# Menampilkan ringkasan loop utama
def show_loop_summary(results: DailResults) -> None:
    # Jumlah pertanyaan, catatan, error, dan skeleton yang gagal dibuat
    rows = [{"Item": "Samples processed", "Value": len(results.preds)},
            {"Item": "raw_logs entries", "Value": len(results.logs)},
            {"Item": "Generation errors", "Value": len(results.gen_errors)},
            {"Item": "Evaluation errors", "Value": len(results.eval_errors)},
            {"Item": "Preliminary SQL without skeleton (pass 2 falls back to question similarity)",
             "Value": sum(1 for e in results.logs if e["dail"]["pre_skeleton_error"])}]
    # Tampilkan ringkasannya
    show(pd.DataFrame(rows), "Main loop (DAIL-SQL)")
    # Rincian error generate
    for pipeline, i, db_id, error in results.gen_errors:
        # Satu baris per error
        print(f"[ERROR generate] {pipeline} | posisi {i} | {db_id} | {error}")
    # Rincian error evaluasi
    for i, db_id, error in results.eval_errors:
        # Satu baris per error
        print(f"[ERROR evaluasi] posisi {i} | {db_id} | {error}")


# ---------------------------------------------------------------------------
# Simpan prediksi dan raw_logs
# ---------------------------------------------------------------------------

# Lokasi file prediksi DAIL
def prediction_path(artifact_dir: Path) -> Path:
    # Di folder prediksi model yang sama, nama berbeda dari 1-Stage/2-Stage
    return artifact_dir / "predictions" / "dail_predictions.txt"


# Menyimpan prediksi, raw_logs, dan prompt supaya hasil generate aman
def save_generation_results(results: DailResults, artifact_dir: Path, raw_logs_dir: Path) -> None:
    # Tulis prediksi, satu SQL per baris (format evaluator resmi)
    prediction_path(artifact_dir).write_text("\n".join(results.preds) + "\n", encoding="utf-8")
    # Tulis raw_logs DAIL
    (raw_logs_dir / DAIL_LOG).write_text(json.dumps(results.logs, ensure_ascii=False, indent=2), encoding="utf-8")
    # Tulis prompt dan output mentah kedua pass, satu baris per pertanyaan
    with open(artifact_dir / "predictions" / "dail_prompts.jsonl", "w", encoding="utf-8") as f:
        # Satu objek JSON per baris
        for trace in results.traces:
            # Tulis barisnya
            f.write(json.dumps(trace, ensure_ascii=False) + "\n")
    # Tampilkan ringkasan lokasi
    print(f"DAIL-SQL: {len(results.preds)} prediksi -> {prediction_path(artifact_dir)}")
    # Lokasi raw_logs
    print(f"raw_logs -> {raw_logs_dir / DAIL_LOG} ({len(results.logs)} entri)")


# ---------------------------------------------------------------------------
# Metrik (fungsi agregasi yang sama dengan metrics_results.json 1-Stage/2-Stage)
# ---------------------------------------------------------------------------

# Membaca raw_logs 1-Stage/2-Stage model yang sama, kalau ada dan pertanyaannya sama dengan DAIL
def load_comparison_logs(raw_logs_dir: Path, dail_logs: list) -> tuple[dict, list]:
    # Tempat raw_logs yang bisa dibandingkan dan catatan alasannya
    logs, notes = {}, []
    # Daftar pertanyaan DAIL: query_id, database, dan gold SQL harus sama persis
    dail_keys = [(e["query_id"], e["db_id"], e["gold_sql"]) for e in dail_logs]
    # Cek kedua kondisi
    for cond in ("baseline", "graphrag"):
        # Lokasi file raw_logs
        path = raw_logs_dir / f"{cond}_log.json"
        # Lewati kalau belum ada
        if not path.exists():
            # Catat alasannya
            notes.append(f"{LABELS[cond]}: {path} belum ada (jalankan eval_pipeline.ipynb dengan MODEL dan SAMPLE_RATIO yang sama)")
            # Lanjut ke kondisi berikutnya
            continue
        # Baca raw_logs
        entries = json.loads(path.read_text(encoding="utf-8"))
        # Pertanyaannya harus sama dan urutannya sama, supaya perbandingannya berpasangan
        if [(e["query_id"], e["db_id"], e["gold_sql"]) for e in entries] != dail_keys:
            # Catat alasannya
            notes.append(f"{LABELS[cond]}: pertanyaan di {path} berbeda dengan run DAIL ini (SAMPLE_RATIO/seed berbeda?), tidak dibandingkan")
            # Lanjut ke kondisi berikutnya
            continue
        # Simpan untuk dibandingkan
        logs[cond] = entries
    # Kembalikan raw_logs dan catatannya
    return logs, notes


# Membangun data QVT DAIL dari raw_logs (hanya run penuh), dengan fungsi yang sama dengan 1-Stage/2-Stage
def build_dail_qvt(dev_json: Path, dail_logs: list, qvt_dir: Path) -> Optional[Path]:
    # Baca dev set
    dev_data = json.loads(Path(dev_json).read_text(encoding="utf-8"))
    # QVT memetakan query_id ke posisi dev.json, jadi butuh seluruh dev set
    if len(dail_logs) != len(dev_data):
        # Tidak bisa dibangun dari run sampel
        return None
    # Grup pertanyaan parafrase (SQL gold identik), sama dengan build_qvt_variations.py
    qvt_data = join_with_raw_logs(build_groups(dev_data), dail_logs)
    # Pastikan foldernya ada
    qvt_dir.mkdir(parents=True, exist_ok=True)
    # Lokasi file QVT DAIL
    path = qvt_dir / "dail_qvt.json"
    # Tulis datanya
    path.write_text(json.dumps(qvt_data, ensure_ascii=False, indent=2), encoding="utf-8")
    # Kembalikan lokasinya
    return path


# Menghitung metrik semua kondisi yang tersedia
def compute_metrics(cfg, raw_logs_dir: Path, sample_ratio: float, qvt_dir: Path) -> dict:
    # Baca raw_logs DAIL
    dail_logs = json.loads((raw_logs_dir / DAIL_LOG).read_text(encoding="utf-8"))
    # Baca raw_logs 1-Stage/2-Stage yang sebanding
    others, notes = load_comparison_logs(raw_logs_dir, dail_logs)
    # Semua kondisi, DAIL-SQL lebih dulu
    logs = {"dail": dail_logs, **others}
    # Bobot token output
    alpha = cfg.token_output_weight
    # Run penuh: bangun data QVT DAIL; 1-Stage/2-Stage memakai file QVT yang dibuat eval_pipeline
    is_full = sample_ratio >= 1.0
    # Kondisi yang data QVT-nya tersedia untuk run ini
    qvt_conditions = None
    # Hanya run penuh
    if is_full:
        # Bangun QVT DAIL
        built = build_dail_qvt(cfg.dev_json, dail_logs, qvt_dir)
        # DAIL selalu, kondisi lain kalau filenya ada
        qvt_conditions = (["dail"] if built else []) + [c for c in others if (qvt_dir / f"{c}_qvt.json").exists()]
    # Catat waktu menghitung metrik
    with timing.run_step("metrics_dail"):
        # Metrik per kondisi dengan fungsi agregasi yang sama
        conditions = {c: {"sla": ms._sla(l), "esm": ms._accuracy(l, "esm_result"), "ex": ms._accuracy(l, "ex_result"),
                          "cm": ms._cm(l), "token": ms._token(l, alpha),
                          "qvt": ms._qvt(c, qvt_dir if is_full else None, qvt_conditions)}
                      for c, l in logs.items()}
        # TEP dengan DAIL-SQL sebagai B: (ΔEX/EX_DAIL) / (ΔT/T_DAIL), Δ = kondisi − DAIL
        tep = {c: ms._tep(dail_logs, l, alpha) for c, l in others.items()}
    # Identitas run, sama dengan metrics_results.json, ditambah pengaturan DAIL
    meta = {**ms.run_metadata(cfg, sample_ratio), "alpha": alpha, "condition_labels": {c: LABELS[c] for c in logs},
            "queries": {c: len(l) for c, l in logs.items()}, "comparison_notes": notes,
            "dail": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(cfg).items() if k.startswith("dail_")},
            "tep_definition": "TEP of each condition with DAIL-SQL as B: (EX_c - EX_DAIL)/EX_DAIL / ((T_c - T_DAIL)/T_DAIL)"}
    # Tampilkan catatan perbandingan
    for note in notes:
        # Satu baris per catatan
        print(f"[INFO] {note}")
    # Kembalikan semua hasil
    return {"meta": meta, "conditions": conditions, "tep": tep}


# Membaca raw_logs semua kondisi yang dipakai compute_metrics(), untuk tabel per pertanyaan
def condition_logs(raw_logs_dir: Path, results: dict) -> dict:
    # Baca file tiap kondisi yang ada di hasil metrik
    names = {"dail": DAIL_LOG, "baseline": "baseline_log.json", "graphrag": "graphrag_log.json"}
    # Kembalikan {kondisi: raw_logs}
    return {c: json.loads((raw_logs_dir / names[c]).read_text(encoding="utf-8")) for c in results["conditions"]}


# ---------------------------------------------------------------------------
# Tabel (angka saja, format sama dengan result_tables.py)
# ---------------------------------------------------------------------------

# Label kolom tiap kondisi yang ada
def _labels(results: dict) -> list:
    # Urutan: DAIL-SQL, 1-stage, 2-stage
    return [(c, LABELS[c]) for c in ("dail", "baseline", "graphrag") if c in results["conditions"]]


# Tabel ringkasan keenam metrik, satu kolom per kondisi
def summary_table(results: dict) -> pd.DataFrame:
    # Kondisi dan isi metriknya
    conds, m = _labels(results), results["conditions"]
    # Tempat baris
    rows = []

    # Menambah satu baris
    def add(label: str, get, kind: str, note: str = ""):
        # Nilai per kondisi; kalau datanya tidak ada, kosong
        values = {}
        # Isi tiap kolom
        for c, lab in conds:
            # Ambil nilainya
            try:
                values[lab] = get(c)
            # Data tidak ada untuk kondisi ini
            except (KeyError, TypeError):
                values[lab] = None
        # Simpan barisnya
        rows.append({"Metric": label, **values, "Note": note, "_kind": kind})

    # SLA tabel dan kolom
    add("SLA table F1", lambda c: m[c]["sla"]["table"]["f1"], "pct")
    add("SLA column F1", lambda c: m[c]["sla"]["column"]["f1"], "pct")
    add("SLA column precision", lambda c: m[c]["sla"]["column"]["precision"], "pct")
    add("SLA column recall", lambda c: m[c]["sla"]["column"]["recall"], "pct")
    # ESM dan EX
    add("ESM", lambda c: m[c]["esm"]["overall"], "pct")
    add("EX", lambda c: m[c]["ex"]["overall"], "pct")
    # CM per klausa, tidak digabung
    for clause in rt.CLAUSES:
        # Lewati klausa yang kosong di semua kondisi
        if any(m[c]["cm"].get(clause) is not None for c, _ in conds):
            add(f"CM {rt.CLAUSE_LABEL[clause]}", lambda c, k=clause: m[c]["cm"][k], "pct")
    # Token
    add("Mean input tokens (T_in)", lambda c: m[c]["token"]["t_in"], "num1")
    add("Mean output tokens (T_out)", lambda c: m[c]["token"]["t_out"], "num1")
    add("Mean token consumption (T)", lambda c: m[c]["token"]["t_mean"], "num1")
    # TEP terhadap DAIL-SQL; kolom DAIL-SQL kosong karena ia pembandingnya
    add("TEP (vs DAIL-SQL)", lambda c: None if c == "dail" else results["tep"][c].get("tep"), "num4",
        "; ".join(f"{LABELS[c]}: {t['reason']}" for c, t in results["tep"].items() if t.get("reason")))
    # QVT
    add("QVT", lambda c: m[c]["qvt"].get("qvt"), "pct", m["dail"]["qvt"].get("reason", ""))
    # Kolom angka
    cols = [lab for _, lab in conds]
    # Tabel dengan format per baris
    return rt.with_formats(pd.DataFrame(rows), {}, {"cols": cols}, "Six metrics: DAIL-SQL" + (" vs " + ", ".join(cols[1:]) if len(cols) > 1 else ""))


# EX, ESM, dan T per tingkat kesulitan
def by_difficulty_table(results: dict) -> pd.DataFrame:
    # Kondisi dan isi metriknya
    conds, m = _labels(results), results["conditions"]
    # Tempat baris
    rows = []
    # Tiap tingkat kesulitan ditambah All
    for d in rt.DIFFICULTIES + ["all"]:
        # Baris satu tingkat
        row = {"Difficulty": rt.DIFF_LABEL[d]}
        # Isi tiap kondisi
        for c, lab in conds:
            # EX dan ESM
            for metric, key in (("EX", "ex"), ("ESM", "esm")):
                # Nilai keseluruhan atau per tingkat
                row[f"{metric} {lab}"] = m[c][key]["overall"] if d == "all" else m[c][key]["by_difficulty"].get(d)
            # Token consumption
            row[f"T {lab}"] = m[c]["token"]["t_mean"] if d == "all" else m[c]["token"]["t_mean_by_difficulty"].get(d)
        # Simpan kalau ada isinya
        if any(not rt.is_missing(v) for k, v in row.items() if k != "Difficulty"):
            rows.append(row)
    # Tabel
    df = pd.DataFrame(rows)
    # Format: EX/ESM persen, T satu desimal
    return rt.with_formats(df, {k: ("num1" if k.startswith("T ") else "pct") for k in df.columns if k != "Difficulty"},
                           title="Metrics by difficulty")


# SLA per kondisi, level, dan populasi
def sla_table(results: dict) -> pd.DataFrame:
    # Tempat baris
    rows = []
    # Tiap kondisi
    for c, lab in _labels(results):
        # Tiap level dan populasi, sama dengan tabel SLA eval_pipeline
        for key, level, population in (("table", "Table", "All queries"), ("column", "Column", "All queries"),
                                       ("column_non_empty_gold", "Column", "Non-empty gold schema")):
            # Hasil SLA
            r = results["conditions"][c]["sla"][key]
            # Simpan kalau ada
            if r:
                rows.append({"Variant": lab, "Level": level, "Population": population,
                             "Precision": r["precision"], "Recall": r["recall"], "F1": r["f1"]})
    # Tabel
    return rt.with_formats(pd.DataFrame(rows), {"Precision": "pct", "Recall": "pct", "F1": "pct"}, title="SLA: precision, recall, F1")


# CM per klausa
def cm_table(results: dict) -> pd.DataFrame:
    # Kondisi
    conds = _labels(results)
    # Satu baris per klausa
    rows = [{"Clause": rt.CLAUSE_LABEL[k], **{lab: results["conditions"][c]["cm"].get(k) for c, lab in conds}} for k in rt.CLAUSES]
    # Tabel
    return rt.with_formats(pd.DataFrame(rows), {lab: "pct" for _, lab in conds}, title="CM per clause")


# Token per kondisi, TEP, dan rincian per pass DAIL
def token_tables(results: dict, dail_logs: list) -> list:
    # Kondisi
    conds = _labels(results)
    # Satu baris per ukuran token
    rows = [{"Measure": label, **{lab: results["conditions"][c]["token"].get(key) for c, lab in conds}}
            for key, label in (("t_in", "Mean input tokens (T_in)"), ("t_out", "Mean output tokens (T_out)"),
                               ("t_mean", "Mean token consumption (T)"), ("t_median", "Median token consumption (T)"))]
    # Tabel token
    token = rt.with_formats(pd.DataFrame(rows), {lab: "num1" for _, lab in conds}, title="Token consumption")
    # TEP tiap kondisi terhadap DAIL-SQL
    tep_rows = [{"Comparison": f"{LABELS[c]} vs DAIL-SQL", "Status": t["status"],
                 "ΔEX (pp)": t["delta_ex"] * 100 if "delta_ex" in t else None, "ΔT": t.get("delta_t"),
                 "TEP": t.get("tep"), "Reason": t.get("reason", "")} for c, t in results["tep"].items()]
    # Tabel TEP
    tep = rt.with_formats(pd.DataFrame(tep_rows), {"ΔEX (pp)": "pp", "ΔT": "dnum1", "TEP": "num4"}, title="TEP (B = DAIL-SQL)")
    # Pertanyaan yang generate-nya berhasil (punya rincian per pass)
    ok = [e for e in dail_logs if e["dail"]["passes"]]
    # Rata-rata satu nilai per pass
    mean = lambda vals: sum(vals) / len(vals) if vals else None
    # Satu baris per pass
    pass_rows = [{"Pass": name, "Mean T_in": mean([e["dail"]["passes"][p]["n_in"] for e in ok]),
                  "Mean T_out": mean([e["dail"]["passes"][p]["n_out"] for e in ok]),
                  "Mean examples in prompt": mean([e["dail"]["passes"][p]["n_examples"] for e in ok])}
                 for p, name in (("pass1", "Pass 1: preliminary SQL"), ("pass2", "Pass 2: final SQL"))
                 # Pass yang tidak dijalankan (pass 1 di k = 0) tidak ditampilkan
                 if any(p in e["dail"]["passes"] for e in ok)]
    # Tabel per pass
    passes = rt.with_formats(pd.DataFrame(pass_rows), {"Mean T_in": "num1", "Mean T_out": "num1", "Mean examples in prompt": "num2"},
                             title="DAIL-SQL tokens per pass (T_in and T_out above are the sum of both passes)")
    # Kembalikan ketiganya
    return [token, tep, passes]


# QVT per kondisi
def qvt_table(results: dict) -> pd.DataFrame:
    # Satu baris per kondisi
    rows = [{"Variant": lab, "Status": q["status"], "QVT": q.get("qvt"), "Queries included": q.get("queries_included"),
             "Queries total": q.get("queries_total"), "Reason": q.get("reason", "")}
            for c, lab in _labels(results) for q in [results["conditions"][c]["qvt"]]]
    # Tabel
    return rt.with_formats(pd.DataFrame(rows), {"QVT": "pct", "Queries included": "int", "Queries total": "int"}, title="QVT")


# Pengecekan prediksi DAIL
def prediction_checks(dail_logs: list, k_shot: int) -> pd.DataFrame:
    # Normalisasi teks SQL untuk dibandingkan
    norm = lambda s: " ".join((s or "").strip().rstrip(";").split()).lower()
    # Pertanyaan yang generate-nya berhasil
    ok = [e for e in dail_logs if e["dail"]["passes"]]
    # Satu baris hitungan
    row = {"Queries": len(dail_logs),
           "Fallback 'SELECT 1'": sum(norm(e["predicted_sql"]) == "select 1" for e in dail_logs),
           "Empty prediction": sum(not (e["predicted_sql"] or "").strip() for e in dail_logs),
           "Identical to gold (text)": sum(norm(e["predicted_sql"]) == norm(e["gold_sql"]) for e in dail_logs),
           "Final SQL = preliminary SQL": sum(norm(e["predicted_sql"]) == norm(e["dail"]["pre_sql"]) for e in ok),
           "Preliminary SQL without skeleton": sum(1 for e in ok if e["dail"]["pre_skeleton_error"]),
           f"Pass 2 with fewer than {k_shot} examples": sum(e["dail"]["passes"]["pass2"]["n_examples"] < k_shot for e in ok),
           "EX=1": sum(e["ex_result"] for e in dail_logs), "ESM=1": sum(e["esm_result"] for e in dail_logs)}
    # Tabel
    return rt.with_formats(pd.DataFrame([row]), {k: "int" for k in row}, title="Prediction checks (DAIL-SQL)")


# Satu baris per pertanyaan DAIL
def per_query_table(dail_logs: list, alpha: float) -> pd.DataFrame:
    # Tempat baris
    rows = []
    # Tiap catatan
    for e in dail_logs:
        # Rincian per pass, kosong kalau generate gagal
        passes = e["dail"]["passes"] or {}
        # Satu baris
        rows.append({"Query ID": e["query_id"], "DB": e["db_id"], "Difficulty": rt.DIFF_LABEL[e["difficulty"]],
                     "Gold SQL": e["gold_sql"], "Preliminary SQL (pass 1)": e["dail"]["pre_sql"], "Predicted SQL": e["predicted_sql"],
                     "EX": e["ex_result"], "ESM": e["esm_result"], "T_in": e["token_input"], "T_out": e["token_output"],
                     "T": rt._t(e, alpha), "T_in pass 1": passes.get("pass1", {}).get("n_in"),
                     "T_in pass 2": passes.get("pass2", {}).get("n_in"), "Schema tokens": e.get("schema_tokens"),
                     "Columns in prompt": len(e["predicted_schema"])})
    # Tabel
    return rt.with_formats(pd.DataFrame(rows), {"EX": "int", "ESM": "int", "T_in": "int", "T_out": "int", "T": "num1",
                                                "T_in pass 1": "int", "T_in pass 2": "int", "Schema tokens": "int",
                                                "Columns in prompt": "int"}, title="Per query (DAIL-SQL)")


# ---------------------------------------------------------------------------
# Latensi (tahap sama dengan pipeline kita; pertanyaan pertama dibuang dari statistik seperti eval_pipeline)
# ---------------------------------------------------------------------------

# Latensi per tahap untuk semua kondisi yang ada
def latency_tables(logs_by_cond: dict, run: Optional[dict] = None) -> list:
    # Tabel waktu sekali-jalan, GPU, dan VRAM kalau ada
    tables = [rt._latency_run_table(run)] if run else []
    # Baris per tahap, per tingkat kesulitan, dan ukuran turunan
    by_step, by_diff, derived = [], [], []
    # Tiap kondisi
    for c, logs in logs_by_cond.items():
        # Catatan dengan latensi, tanpa pertanyaan pertama (pemanasan model)
        entries = rt._latency_entries(logs)
        # Lewati kalau kosong
        if not entries:
            continue
        # Tiap tahap
        for key, name in LATENCY_STEPS.items():
            # Nilai tahap ini
            values = [e["latency"][key] for e in entries if key in e["latency"]]
            # Simpan kalau ada
            if values:
                by_step.append({"Variant": LABELS[c], "Step": name, **rt._latency_row(values)})
            # Per tingkat kesulitan
            for d in rt.DIFFICULTIES:
                # Nilai tahap ini pada tingkat d
                values = [e["latency"][key] for e in entries if e["difficulty"] == d and key in e["latency"]]
                # Simpan kalau ada
                if values:
                    by_diff.append({"Variant": LABELS[c], "Difficulty": rt.DIFF_LABEL[d], "Step": name, **rt._latency_row(values)})

        # Total satu tahap
        def total(key, entries=entries):
            return sum(e["latency"].get(key, 0.0) for e in entries)

        # Waktu generate dan end-to-end
        generate, end_to_end = total("llm_generate"), total("end_to_end")
        # Ukuran turunan, sama dengan tabel eval_pipeline
        derived.append({"Variant": LABELS[c],
                        "Output tokens per second": sum(e["token_output"] for e in entries) / generate if generate else None,
                        "Input tokens per generation second": sum(e["token_input"] for e in entries) / generate if generate else None,
                        "Retrieval share of end-to-end": total("retrieval_total") / end_to_end if end_to_end else None,
                        "Generation share of end-to-end": generate / end_to_end if end_to_end else None})
    # Format kolom waktu
    formats = {"Queries": "int", "Mean (s)": "num3", "Median (s)": "num3", "P95 (s)": "num3"}
    # Kalau ada latensi per pertanyaan, tambahkan ketiga tabel
    if by_step:
        tables += [rt.with_formats(pd.DataFrame(by_step), formats, title="Per-question latency by step"),
                   rt.with_formats(pd.DataFrame(derived), {"Output tokens per second": "num1", "Input tokens per generation second": "num1",
                                                           "Retrieval share of end-to-end": "pct", "Generation share of end-to-end": "pct"},
                                   title="Derived latency measures"),
                   rt.with_formats(pd.DataFrame(by_diff), formats, title="Per-question latency by difficulty")]
    # Kembalikan semua tabel
    return tables


# Latensi per pertanyaan DAIL, semua tahap dalam detik
def latency_per_query(dail_logs: list) -> Optional[pd.DataFrame]:
    # Satu baris per pertanyaan yang punya latensi
    rows = [{"Query ID": e["query_id"], "DB": e["db_id"], "Difficulty": rt.DIFF_LABEL[e["difficulty"]],
             **{name: e["latency"].get(key) for key, name in LATENCY_STEPS.items()}} for e in dail_logs if e.get("latency")]
    # Tidak ada latensi
    if not rows:
        return None
    # Tabel
    return rt.with_formats(pd.DataFrame(rows), {name: "num3" for name in LATENCY_STEPS.values()}, title="Latency per question (DAIL-SQL)")


# ---------------------------------------------------------------------------
# Tampilan di notebook
# ---------------------------------------------------------------------------

# Semua grup tabel, dipakai notebook DAN Excel supaya angkanya sama
def table_groups(cfg, results: dict, logs_by_cond: dict, spider_scores: Optional[dict] = None,
                 run: Optional[dict] = None) -> list:
    # raw_logs DAIL
    dail_logs = logs_by_cond["dail"]
    # Bobot token output
    alpha = cfg.token_output_weight
    # [(nama sheet, [tabel])]
    groups = [("Summary", [summary_table(results)]),
              ("By Difficulty", [by_difficulty_table(results)]),
              ("M1 SLA", [sla_table(results)]),
              ("M4 CM", [cm_table(results)]),
              ("M5 Token TEP", token_tables(results, dail_logs)),
              ("M6 QVT", [qvt_table(results)]),
              ("SQL Diagnostics", [prediction_checks(dail_logs, cfg.dail_k_shot)])]
    # Evaluator resmi kalau sudah dijalankan
    if spider_scores:
        groups.append(("Spider Official", [rt.spider_summary(spider_scores)]))
    # Tabel per pertanyaan
    groups.append(("Per Query", [per_query_table(dail_logs, alpha)]))
    # Latensi
    groups.append(("Latency", latency_tables(logs_by_cond, run)))
    # Latensi per pertanyaan
    per_query_latency = latency_per_query(dail_logs)
    # Hanya kalau ada
    if per_query_latency is not None:
        groups.append(("Latency Per Query", [per_query_latency]))
    # Kembalikan semua grup
    return groups


# Menampilkan satu grup tabel
def show_group(groups: list, name: str) -> None:
    # Cari grup yang namanya cocok
    for group_name, tables in groups:
        # Lewati grup lain
        if group_name != name:
            continue
        # Tampilkan semua tabelnya
        for table in tables:
            show(table)


# ---------------------------------------------------------------------------
# Evaluator resmi Spider (fungsi yang sama dengan eval_pipeline)
# ---------------------------------------------------------------------------

# Menjalankan evaluation.py resmi untuk prediksi DAIL
def run_spider_check(cfg, artifact_dir: Path, gold_path: Path) -> dict:
    # Folder file sementara khusus DAIL
    tmp = artifact_dir / "spider_eval_tmp" / "dail"
    # Buat foldernya
    tmp.mkdir(parents=True, exist_ok=True)
    # Jalankan ESM dan EX resmi, dan catat waktunya
    with timing.run_step("spider_official/DAIL-SQL"):
        scores = {"DAIL-SQL": steps._run_spider_for(cfg, "DAIL-SQL", prediction_path(artifact_dir), gold_path, tmp)}
    # Tampilkan ringkasannya
    show(rt.spider_summary(scores))
    # Kembalikan skornya
    return scores


# ---------------------------------------------------------------------------
# Simpan semua hasil
# ---------------------------------------------------------------------------

# Menulis grup tabel ke satu file Excel, dengan penulis sheet yang sama dengan hasil_skripsi.xlsx
def write_workbook(groups: list, out: Path) -> Path:
    # Nomor tabel Excel dimulai dari awal
    Sheet._n_tables = 0
    # Workbook kosong
    wb = Workbook()
    # Buang sheet bawaan
    wb.remove(wb.active)
    # Tiap grup jadi satu sheet
    for name, tables in groups:
        # Sheet baru
        sheet = Sheet(wb, name)
        # Tulis semua tabelnya
        for table in tables:
            sheet.table(table)
        # Kolom pengenal dibekukan pada tabel per pertanyaan
        sheet.finish("D" if name in ("Per Query", "Latency Per Query") else "A")
    # Pastikan foldernya ada
    out.parent.mkdir(parents=True, exist_ok=True)
    # Simpan
    wb.save(out)
    # Kembalikan lokasinya
    return out


# Menyimpan metrik, skor resmi, latensi, dan Excel DAIL; folder terpisah supaya tidak menimpa hasil 1-Stage/2-Stage
def save_all_results(cfg, model: str, results: dict, logs_by_cond: dict, spider_scores: dict,
                     artifact_dir: Path, raw_logs_dir: Path) -> Path:
    # Folder tabel DAIL
    tables_dir = artifact_dir / "tables" / "dail"
    # Buat foldernya
    tables_dir.mkdir(parents=True, exist_ok=True)
    # Simpan metrik
    (tables_dir / DAIL_METRICS).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    # Gabungkan skor evaluator resmi yang berhasil dibaca
    frames = [df.assign(Pipeline=label) for label, df in (spider_scores or {}).items() if not df.empty]
    # Simpan kalau ada
    if frames:
        pd.concat(frames, ignore_index=True).to_csv(tables_dir / "spider_official_eval.csv", index=False)
    # Simpan waktu sekali-jalan, GPU, dan VRAM (latency_run.json di folder DAIL)
    run_path = timing.save_run(tables_dir)
    # Mulai mengukur waktu Excel dan zip
    start = time.perf_counter()
    # Semua tabel, sama dengan yang tampil di notebook
    groups = table_groups(cfg, results, logs_by_cond, spider_scores, json.loads(run_path.read_text(encoding="utf-8")))
    # Tulis Excel
    xlsx = write_workbook(groups, tables_dir / "hasil_dail.xlsx")
    # Tampilkan lokasi file
    print(f"Tersimpan: {tables_dir / DAIL_METRICS}\nTersimpan: {xlsx}")
    # Di Kaggle, siapkan zip berisi hasil DAIL saja
    if Path("/kaggle/working").exists():
        # Folder sementara untuk isi zip
        bundle = Path("/kaggle/working") / f"hasil_dail_{model}"
        # Mulai dari folder kosong
        shutil.rmtree(bundle, ignore_errors=True)
        # Salin tabel DAIL
        shutil.copytree(tables_dir, bundle / "tables" / "dail")
        # Folder prediksi
        (bundle / "predictions").mkdir(parents=True)
        # Salin prediksi dan prompt DAIL
        for name in ("dail_predictions.txt", "dail_prompts.jsonl"):
            # Hanya kalau ada
            if (artifact_dir / "predictions" / name).exists():
                shutil.copy(artifact_dir / "predictions" / name, bundle / "predictions" / name)
        # Folder raw_logs
        (bundle / "raw_logs").mkdir()
        # Salin raw_logs DAIL
        shutil.copy(raw_logs_dir / DAIL_LOG, bundle / "raw_logs" / DAIL_LOG)
        # Salin Excel supaya mudah diunduh
        shutil.copy(xlsx, f"/kaggle/working/hasil_dail_{model}.xlsx")
        # Buat zip
        zip_path = shutil.make_archive(str(bundle), "zip", root_dir=bundle)
        # Tampilkan file yang bisa diunduh
        print(f"Unduh dari /kaggle/working: hasil_dail_{model}.xlsx dan {Path(zip_path).name}")
    # Tampilkan waktu menulis Excel dan zip
    print(f"Waktu menulis Excel dan zip: {time.perf_counter() - start:.1f} detik")
    # Kembalikan lokasi Excel
    return xlsx
