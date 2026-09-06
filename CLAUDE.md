# CLAUDE.md — GraphRAG Text-to-SQL Pipeline

Skripsi: **Graph-Based Retrieval-Augmented Generation untuk Text-to-SQL**
Dataset: Spider benchmark
Environment: Kaggle Notebook (GPU T4, memory terbatas)

---


## Konteks Penelitian

Penelitian ini menganalisis trade-off antara performa SQL generation dan efisiensi token consumption. Ada dua pendekatan utama yang dibandingkan:

- **Baseline RAG** — table-level retrieval: retrieve top-k tabel, kirim semua kolom dari tabel terpilih ke LLM
- **GraphRAG** — two-stage column-level retrieval + graph traversal + path pruning: kirim hanya kolom yang relevan

Token efficiency GraphRAG berasal dari selektivitas retrieval — bukan dari seluruh database, hanya k tabel → k kolom → pruned columns yang masuk ke prompt.

---

## Struktur File

```
thesis-2803/
├── src/
│   ├── core/
│   │   ├── config.py       — semua hyperparameter dan path (sumber kebenaran tunggal)
│   │   └── schema.py       — parsing tables.json Spider → DataFrame → NetworkX graph (column-level)
│   ├── retrieval/
│   │   ├── retrieval.py    — GraphRAG: two-stage semantic linking, graph traversal, path pruning, context builder
│   │   └── baseline.py     — Baseline: table-level graph, table-level semantic linking, context builder
│   ├── generation/
│   │   ├── generation.py   — prompt builder, SQL cleaner, model loading (4-bit quantized), greedy decode
│   │   └── few_shot.py     — few-shot index: pre-compute training set embeddings, dynamic retrieval
│   ├── experiments/
│   │   ├── pipeline.py     — orchestration utama, CLI entry point, Spider evaluation
│   │   ├── sweep.py        — hyperparameter sweep: top_k_tables × top_k_columns (tanpa LLM)
│   │   ├── ablation.py     — ablation study: few-shot k={0,1,3,5} × {baseline, graphrag}
│   │   └── run_all_dimensions.py — entry point evaluasi: baca raw_logs, jalankan 6 dimensi analisis
│   ├── utils/               — helper evaluasi: schema_utils.py (load/validasi db_schema), sql_execution.py (eksekusi SQLite aman)
│   ├── metrics/             — metrik skripsi level query: sla.py, esm_ex_cm.py, token_consumption.py, tep.py, qvt.py
│   └── dimensions/          — 6 dimensi analisis (agregasi + interpretasi dari metrics/): dim1_efficiency.py … dim6_ablation.py
├── external/
│   └── spider_eval/        — official SPIDER evaluation.py + process_sql.py (di-wget manual, dipakai juga oleh src/metrics/)
├── data/
│   ├── raw_logs/            — INPUT dimensi analisis: baseline_log.json, graphrag_log.json (skema di context/EVALUATION_ANALYSIS_GUIDE.md Bagian 2) — diproduksi oleh `pipeline.py`'s `run_comparison()` (via `--baseline`), BELUM oleh `ablation.py`, lihat TBD di bawah
│   ├── db_schema/           — schema per db_id untuk validasi SLA
│   └── qvt_variations/      — dataset paraphrase NL question untuk metrik QVT (Dimensi 4)
├── context/
│   ├── EVALUATION_ANALYSIS_GUIDE.md — source of truth formula & alur berpikir untuk src/metrics/ dan src/dimensions/ (JANGAN ubah formula/threshold di situ tanpa konfirmasi peneliti)
│   ├── IMPLEMENTATION_DECISIONS.md — catatan SEMUA keputusan peneliti saat implementasi (konflik guide vs kode/tooling resmi SPIDER, dan alasannya) — baca ini sebelum mengubah perilaku src/metrics/ yang terasa "aneh"
│   ├── METRICS_EXPLAINED.md — companion EVALUATION_ANALYSIS_GUIDE.md dalam bahasa non-formula: apa arti tiap angka METRIK, bukan cuma formulanya (SLA, Token Consumption, TEP, QVT sejauh ini — ESM/EX/CM ditambah seiring progres)
│   ├── DIMENSIONS_EXPLAINED.md — satu level di atas METRICS_EXPLAINED.md: apa yang disimpulkan satu DIMENSI (kombinasi beberapa metrik) yang tidak kelihatan dari satu metrik saja (Dimensi 1 sejauh ini — sisanya ditambah seiring progres)
│   └── RESEARCHER_TODO.md — checklist hal yang jadi tanggung jawab PENELITI (bukan Claude Code): data yang harus disiapkan/dikumpulkan manual, eksperimen yang harus dijalankan (butuh GPU/Kaggle), keputusan yang perlu didiskusikan dulu. **Baca file ini setiap kali peneliti bertanya "hal apa yang belum kita siapkan?"** atau pertanyaan sejenis soal kesiapan/prasyarat
├── notebooks/               — EDA notebooks + notebooks/compiled/ (Kaggle-ready compiled notebook)
│   └── eval_pipeline.ipynb  — notebook GraphRAG vs Baseline side-by-side (ESM/EX via Spider eval + schema recall/precision). Ditulis SEBELUM refactor foldering — import & path masih flat layout (`from config import ...`, `evaluation.py` di cwd), belum disesuaikan ke `src.*`/`external/spider_eval/`
├── outputs/
│   ├── plots/               — EDA plot PNGs
│   ├── predictions/         — predictions.txt, baseline_predictions.txt, ablation_*_predictions_k*.txt, ablation_*_prompts_k*.jsonl
│   ├── logs/                — baseline_log.txt
│   └── tables/              — sweep_results.csv, ablation_results.csv, baseline_results.csv, comparison_report.txt
├── requirements.txt          — dependency untuk src/metrics/ & src/dimensions/ (jalan lokal, bukan Kaggle — lihat komentar di file)
└── CLAUDE.md                — file ini (tetap di root)
```

Semua modul di `src/` dipanggil sebagai `src.<paket>.<modul>` (mis. `from src.core.config import PipelineConfig`). Entry point CLI (`pipeline.py`, `sweep.py`, `ablation.py`, `baseline.py`, `run_all_dimensions.py`) masih bisa dijalankan langsung dengan `python src/experiments/pipeline.py` dkk — setiap entry point punya bootstrap `sys.path.insert(...)` di baris import supaya import `src.*`-nya tetap resolve meski dijalankan sebagai script, bukan module.

**Status `src/utils/`, `src/metrics/`, `src/dimensions/`:** hasil merge dari `evaluation_pipeline/`, sebuah folder template yang tadinya dibuat Claude tanpa melihat codebase ini (dari sesi brainstorming metrik/dimensi terpisah). Formula dan alur sudah final dari proposal (lihat `context/EVALUATION_ANALYSIS_GUIDE.md`).
- **`src/metrics/` — SEMUA sudah diimplementasi penuh** (bukan lagi stub): `sla.py`, `esm_ex_cm.py`, `token_consumption.py`, `tep.py`, `qvt.py`. `src/utils/` juga penuh: `schema_utils.py`, `sql_normalize.py` (fix parser JOIN-keyword, lihat `IMPLEMENTATION_DECISIONS.md` poin 11), `raw_logs.py` (helper grouping per difficulty, dipakai lintas `src/dimensions/`).
- **`src/dimensions/` — 5 dari 6 dimensi sudah diimplementasi**: Dimensi 1 (Efisiensi), 2 (Struktur SQL), 3 (Komponen SQL), 5 (Bottleneck), 6 (Ablation Few-Shot, interpretasi saja — belum ada data nyata dari `ablation.py`). **Masih stub: Dimensi 4** (Robustness/QVT) — blocked di `data/qvt_variations/` yang masih kosong (lihat `RESEARCHER_TODO.md`).
- ✅ **Sudah divalidasi terhadap Spider dev set ASLI** (2026-08-26, bukan cuma sanity test SQLite sintetis lagi) — 1034 query dev set real (`data/spider_data/`, di luar git, lihat `.gitignore`), round-trip test lewat `esm_ex_cm.py`/`sla.py`: 0 exception, 0 kegagalan ESM. Satu limitation data upstream ditemukan (1 baris rusak encoding di `wta_1.sqlite`, pengaruh 0.19% dev set) dan didokumentasikan, bukan diperbaiki (lihat `IMPLEMENTATION_DECISIONS.md` poin 16 untuk alasan lengkap). `run_comparison()` di `pipeline.py` yang menulis `data/raw_logs/*.json` dari pipeline GraphRAG/Baseline sungguhan masih belum pernah dijalankan (butuh GPU/Kaggle, lihat `RESEARCHER_TODO.md`) — yang sudah divalidasi adalah lapisan metrics (`src/metrics/`), bukan pipeline retrieval+generation penuh.
- **Semua keputusan implementasi** (konflik guide vs kode yang sudah ada, konflik guide vs keterbatasan tooling resmi SPIDER, dan open items yang belum diputuskan) ada di **`context/IMPLEMENTATION_DECISIONS.md`** — 20 keputusan tercatat sejauh ini, termasuk formula Token Consumption, EX order-sensitivity, label difficulty, casing CM clause, keterbatasan `union_all`, fix parser JOIN-keyword, hasil validasi dev set asli, bug fix `_set_op_clause_match()` (UNION/INTERSECT/EXCEPT salah skor akibat mutasi in-place), definisi Baseline final (poin 18), proxy schema-linking vs SLA (poin 19), dan confound diagnostics read-only (poin 20).

---

## Arsitektur Pipeline (GraphRAG)

```
Pertanyaan user (bahasa alami)
        │
        ▼
[1] Two-Stage Semantic Retrieval  ← BGE-M3 embeddings, n-gram matching
        │  Stage 1: top-k table selection (cosine similarity)
        │  Stage 2: top-k column selection dari tabel kandidat
        ▼
[2] Graph Traversal               ← NetworkX shortest-path pada schema graph
        │  Input: kolom hasil retrieval sebagai anchor nodes
        │  Output: jalur konektivitas antar tabel (untuk JOIN)
        ▼
[3] Path Pruning                  ← buang intermediate nodes yang bukan PK/FK/direct hit
        ▼
[4] Schema Context Builder        ← CREATE TABLE DDL dengan FK annotations
        ▼
[5] Few-Shot Retrieval            ← cosine similarity terhadap training set embeddings
        ▼
[6] Prompt Builder                ← Hybrid: ASP + CRP + TRP + ODP
        │  ### Task
        │  ### Database Schema   (DDL hasil retrieval)
        │  ### Examples          (few-shot, jika k > 0)
        │  ### Question
        │  ### Instructions
        │  ### Answer\n```sql
        ▼
[7] LLM Inference                 ← Qwen2.5-Coder-7B-Instruct, 4-bit NF4 quantized
        ▼
[8] SQL Cleaner                   ← strip markdown, fix dialect quirks, normalize aliases
        ▼
predictions.txt → Spider Official Eval (EM + EX)
```

---

## Arsitektur Pipeline (Baseline)

```
Pertanyaan user
        │
        ▼
[1] Table-Level Semantic Linking  ← single query embedding vs table embeddings
        │  (bukan n-gram, bukan two-stage)
        │  Embed format: "table <name> columns <col1> <col2> ..."
        ▼
[2] Table Path Tracing            ← shortest path antar tabel via FK edges
        ▼
[3] Schema Context Builder        ← CREATE TABLE DDL, semua kolom dari tabel terpilih
        │  (TIDAK ada pruning — seluruh kolom tabel ikut masuk)
        ▼
[4] Prompt Builder + LLM + SQL Cleaner  (sama dengan GraphRAG)
        ▼
baseline_predictions.txt → Spider Official Eval
```

---

## Schema Graph

Dibangun dari `tables.json` Spider via `schema.py`:

- **Node** = satu kolom, identifier unik: `{database}.{table}.{column}`
- **Node attributes**: database, table, column, column_type, is_pk
- **Edge type 1** `belongs_to_pk` — kolom non-PK → PK tabel yang sama
- **Edge type 2** `foreign_key` — kolom FK → kolom PK di tabel lain
- Total: 8.747 nodes, 2.854 edges

Graph baseline (`baseline.py`) berbeda — node-nya per tabel, bukan per kolom.

---

## Model

| Komponen | Model | Keterangan |
|---|---|---|
| Embedding | `BAAI/bge-m3` | Dipakai untuk schema linking DAN few-shot retrieval |
| LLM | `Qwen/Qwen2.5-Coder-7B-Instruct` | 4-bit NF4 quantized via BitsAndBytes |

Embedding model **sama** untuk dua keperluan berbeda — schema linking dan few-shot retrieval. Jangan ganti salah satu tanpa ganti keduanya.

---

## Konfigurasi (`config.py`)

Semua parameter ada di `PipelineConfig` dataclass. **Jangan hardcode nilai di file lain** — selalu referensikan dari `cfg`.

| Parameter | Nilai saat ini | Status | Keterangan |
|---|---|---|---|
| `embedding_model` | `BAAI/bge-m3` | Final | |
| `llm_model` | `Qwen/Qwen2.5-Coder-7B-Instruct` | Final | |
| `top_k_tables` | `3` | TBD | Ditentukan oleh sweep.py |
| `top_k_columns` | `5` | TBD | Ditentukan oleh sweep.py |
| `semantic_similarity_threshold` | `0.35` | TBD | Fallback jika top_k_columns=0 |
| `few_shot_k` | `3` | TBD | Hasil ablation study |
| `few_shot_same_db_first` | `True` | Final | Prioritaskan contoh dari DB yang sama |
| `max_ngram` | `3` | Final | Segmentasi query sampai trigram |
| `max_new_tokens` | `256` | Final | Budget generasi LLM. `config.py` = `256`; baris ini sebelumnya keliru tertulis `200` — kode selalu pakai `cfg.max_new_tokens`. |
| `temperature` | `0.0` | Final | Greedy decoding, deterministik |
| `load_in_4bit` | `True` | Final | Kuantisasi NF4 |
| `token_output_weight` | `3.0` | Final | α/μ dalam T = T_in + α×T_out — fixed dari proposal subbab 3.8.2.5, diupdate dari 1.0 (lihat riwayat resolusi konflik di "Evaluasi Metrik") |

Parameter bertanda **TBD** akan diupdate setelah sweep dan ablation selesai.

---

## Ablation Study

Cross design 2×4 — jalankan terpisah 2x untuk menghindari OOM:

**Run 1 — Baseline RAG:**
```bash
python src/experiments/ablation.py --mode baseline --k-values 0 1 3 5 --sample 1.0
```

**Run 2 — GraphRAG:**
```bash
python src/experiments/ablation.py --mode graphrag --k-values 0 1 3 5 --sample 1.0
```

Output per run:
- `outputs/predictions/ablation_{mode}_predictions_k{k}.txt` — SQL predictions → masuk ke Spider `external/spider_eval/evaluation.py` untuk EM/EX
- `outputs/predictions/ablation_{mode}_prompts_k{k}.jsonl` — per sample: i, db_id, question, tokens_in, tokens_out, token_consumption, prompt, pred_sql
- `outputs/tables/ablation_results.csv` — avg_recall, avg_precision, avg_prompt_tokens (T_in), avg_output_tokens (T_out), avg_token_consumption (T) per mode×k → dasar perhitungan TEP

| | k=0 | k=1 | k=3 | k=5 |
|---|---|---|---|---|
| Baseline RAG | ✓ | ✓ | ✓ | ✓ |
| GraphRAG | ✓ | ✓ | ✓ | ✓ |

Tujuan ablation: cari **elbow point** — nilai k di mana penambahan few-shot example sudah tidak signifikan meningkatkan performa relatif terhadap token cost. Hipotesis: lonjakan terbesar ada di k=0→1, setelah itu diminishing returns.

---

## Evaluasi Metrik

| Metrik | Tool | Keterangan |
|---|---|---|
| Exact Set Match (ESM) | Spider `external/spider_eval/evaluation.py --etype match` | |
| Execution Accuracy (EX) | Spider `external/spider_eval/evaluation.py --etype exec` | |
| Component Match (CM) | Spider `external/spider_eval/evaluation.py` | |
| Token Consumption | `avg_token_consumption` di `ablation_results.csv` | T = T_in + α×T_out. T_in = prompt tokens, T_out = generated SQL tokens, α = `token_output_weight` di config.py (**3.0**, fixed dari proposal). Diukur inline per sample setelah generate_sql. |
| TEP (Token Elasticity of Performance) | Custom metric, hitung post-hoc | TEP_G = (ΔEX_G/EX_B) / (ΔT_G/T_B) — elastisitas performa GraphRAG relatif terhadap konsumsi token vs Baseline. ΔEX_G = EX_G − EX_B, ΔT_G = T_G − T_B. Hitung dari `ablation_results.csv` + EX dari Spider eval. |
| QVT (Query Variance Testing) | Custom metric | Stabilitas output terhadap variasi pertanyaan |
| Schema Linking Accuracy (SLA) | `src/metrics/sla.py` | Precision/recall/F1 retrieval vs gold schema, table-level & column-level terpisah, parser resmi SPIDER, macro-average per query. **INI metrik SLA yang dilaporkan.** Recall/precision yang di-print `retrieval.py`/`baseline.py` saat run = proxy internal (name-matching tanpa kualifikasi `table.column`, table+column dicampur satu set) — untuk progress/sweep saja, JANGAN dilaporkan sebagai SLA (`IMPLEMENTATION_DECISIONS.md` poin 19). |

Definisi lengkap + algoritma step-by-step untuk keenam metrik di atas (termasuk SLA
dan breakdown 6 dimensi analisis skripsi) ada di **`context/EVALUATION_ANALYSIS_GUIDE.md`**
— itu source of truth-nya, tabel di atas cuma ringkasan. Implementasi metrik ada di
`src/metrics/` (semua penuh), agregasi + interpretasi per dimensi ada di
`src/dimensions/` (5 dari 6 penuh; Dimensi 4/QVT masih stub — lihat status di atas).

✅ **Konflik formula Token Consumption (μ vs α) dan keputusan EX order-sensitivity
sudah diselesaikan** — lihat `context/IMPLEMENTATION_DECISIONS.md` poin 1 dan 5
untuk detail konflik + alasan lengkap. Ringkas: `token_output_weight` = **3.0**
(bukan 1.0), dan EX tetap pakai `eval_exec_match()` resmi SPIDER apa adanya
(order-sensitive, beda dari algoritma literal di guide 1.3(c) langkah 3).

### Confound diagnostics (read-only, tidak mengubah metrik)

`run_comparison()` di `pipeline.py` menulis blok **CONFOUND DIAGNOSTICS** ke
`outputs/tables/comparison_report.txt` — instrumentasi read-only untuk dua
keterbatasan evaluasi, TIDAK memengaruhi nilai ESM/EX/CM/raw_logs (lihat
`IMPLEMENTATION_DECISIONS.md` poin 20):
- **Frekuensi outer JOIN** (`LEFT`/`RIGHT`/`FULL`) di gold vs prediksi tiap
  kondisi. Sejak poin 11, in-process evaluator menormalisasi outer JOIN → `JOIN`
  untuk *parsing* (string mentah tetap dieksekusi untuk EX), jadi tidak ada lagi
  "silent ESM/CM=0". Yang dihitung sekarang: prediksi outer-JOIN yang `ESM=1`
  tapi `EX=0` — kandidat di mana normalisasi parsing menyembunyikan beda semantik
  nyata.
- **Disagreement EX order-sensitivity** (poin 5) — jumlah query yang EX resmi
  (order-sensitive) = 0 tapi perbandingan multiset baris (order-insensitive)
  match gold, dipecah per kondisi + subset yang gold-nya punya `ORDER BY`.
  `ex_result` di raw_logs TETAP 100% dari `eval_exec_match()` resmi.

### Few-shot: sumber data & bias yang harus disebut di metodologi

Few-shot example diambil HANYA dari `train_spider.json` (bukan dev/test).
`few_shot_same_db_first=True` menaikkan contoh dengan `db_id` yang sama ke atas
top-k. Karena split train/dev Spider berbagi sebagian `db_id`, banyak contoh
few-shot berbagi schema dengan pertanyaan dev — ini **retrieval dari train set,
bukan kebocoran dev/test**, tapi WAJIB disebut eksplisit di bab metodologi.

Spider evaluation scripts harus didownload manual ke `external/spider_eval/`:
```bash
mkdir -p external/spider_eval && cd external/spider_eval
wget https://raw.githubusercontent.com/taoyds/spider/master/evaluation.py
wget https://raw.githubusercontent.com/taoyds/spider/master/process_sql.py
```

### Atribusi kode: SPIDER resmi vs custom

**Ingat ini setiap kali menulis dokumentasi/penjelasan/komentar untuk apapun yang menyentuh evaluasi** (`src/metrics/`, `src/dimensions/`, `context/*.md`) — selalu tandai jelas mana yang mana:

- **Source code resmi SPIDER** — benar-benar berasal dari `external/spider_eval/evaluation.py` / `process_sql.py` (di-download dari repo resmi `taoyds/spider` di GitHub), dipanggil apa adanya. Contoh: `Evaluator.eval_exact_match()`, `Evaluator.eval_partial_match()`, `eval_exec_match()`, `eval_hardness()`, `Schema`, `get_sql()`, `build_foreign_key_map_from_json()`, `rebuild_sql_val()`/`rebuild_sql_col()`.
- **Kode custom untuk skripsi ini** — ditulis khusus untuk project ini, MEMBUNGKUS atau MEMPERLUAS kode resmi di atas (SPIDER sendiri tidak menyediakan ini secara publik). Contoh: `evaluate_single_query()`, `build_kmaps()`, `build_schema_for_db()`, `_from_clause_match()`, `_set_op_clause_match()`, `aggregate_esm/ex/cm()` (semua di `esm_ex_cm.py`); `extract_ground_truth_schema()`, `_collect_col_ids()`, `_col_id_to_table_column()`, `compute_sla()`, `aggregate_sla()` (semua di `sla.py`) — SLA sendiri BUKAN metrik resmi SPIDER, itu spesifik untuk proposal skripsi ini.

Jangan biarkan pembaca (termasuk sesi Claude Code berikutnya) salah asumsi sesuatu itu "dari SPIDER" padahal ditulis sendiri, atau sebaliknya — penting untuk akurasi bab metodologi yang menyebut "reuse resmi SPIDER evaluation script".

---

## Paths (Kaggle)

```python
data_path = Path("/kaggle/input/datasets/alrette/spiderdataset/spider_data")
# tables.json  → data_path / "tables.json"
# dev.json     → data_path / "dev.json"
# train.json   → data_path / "train_spider.json"
# database/    → data_path / "database"
# gold SQL     → data_path / "dev_gold.sql"
```

---

## Data Storage

**Tidak ada vector database eksternal.** Semua embedding disimpan sebagai `torch.Tensor` di memori Python selama runtime:

- `SchemaIndex` — embedding tabel dan kolom per database, di-cache di `schema_cache` dict
- `FewShotIndex` — embedding seluruh training set (~7000 pertanyaan), built once at startup
- Semua hilang saat session selesai — ini by design untuk research pipeline

---

## Hal yang Jangan Diubah Tanpa Diskusi

- Format prompt di `generation.py` — sudah di-tune, perubahan kecil berdampak besar ke output LLM
- Struktur graph di `schema.py` — node identifier `{db}.{table}.{col}` dipakai di banyak tempat
- `few_shot_same_db_first=True` — sudah jadi keputusan desain final
- Greedy decoding (`temperature=0.0`, `do_sample=False`) — perlu deterministik untuk reprodusibilitas
- **Definisi Baseline = table-level retrieval** (`src/retrieval/baseline.py`) — FINAL per 2026-09-06. Full-schema bypass (`--full-schema` / `use_full_schema_bypass=True`) adalah mode ablation saja, BUKAN baseline skripsi. Lihat `IMPLEMENTATION_DECISIONS.md` poin 18.

## Hal yang Masih TBD (update setelah eksperimen)

- Nilai final `top_k_tables`, `top_k_columns`, `semantic_similarity_threshold` → tunggu hasil `sweep.py`
- Nilai final `few_shot_k` → tunggu hasil ablation study
- ✅ `max_new_tokens`: konsisten `256` di `config.py`, `generation.py` (`cfg.max_new_tokens`), README, dan tabel config di atas — baris tabel yang keliru `200` diperbaiki 2026-09-06
- `data/raw_logs/{baseline,graphrag}_log.json` sudah diproduksi oleh `run_comparison()` di `pipeline.py` (via `python src/experiments/pipeline.py --baseline`) — **belum** oleh `ablation.py` (masih pakai jalur lama tanpa evaluasi ESM/EX/CM per query). Pipeline retrieval+generation PENUH (GraphRAG/Baseline sungguhan) belum pernah dijalankan end-to-end di Kaggle (torch tidak tersedia di environment dev lokal). **Lapisan metrics (`src/metrics/`) SUDAH divalidasi terhadap Spider dev set asli** (2026-08-26, 1034 query nyata, lihat `IMPLEMENTATION_DECISIONS.md` poin 16) — yang belum tervalidasi spesifik hanya pipeline retrieval+generation-nya sendiri, bukan lapisan evaluasi/metrics-nya

---

## CLI Quick Reference

```bash
# GraphRAG — full dev set
python src/experiments/pipeline.py --skip-sweep

# GraphRAG — dengan sweep otomatis dulu
python src/experiments/pipeline.py

# Baseline — table-level retrieval
python src/retrieval/baseline.py --sample 1.0

# Full schema bypass (eksperimen, belum jadi mode resmi)
python src/experiments/pipeline.py --full-schema

# Ablation few-shot
python src/experiments/ablation.py --k-values 0 1 3 5

# Hyperparameter sweep saja (tanpa LLM, cepat)
python src/experiments/sweep.py --sample 0.2

# Perbandingan GraphRAG vs Baseline
python src/experiments/pipeline.py --baseline
```
