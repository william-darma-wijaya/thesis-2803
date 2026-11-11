# DAIL-SQL sebagai Baseline Skripsi — Penjelasan

Dokumen ini menjelaskan (1) arsitektur DAIL-SQL yang sebenarnya, (2) apa yang diubah dan disesuaikan supaya perbandingan dengan GraphRAG 1-Stage dan 2-Stage apple-to-apple, dan (3) isi tiap file. Keputusan lengkapnya dicatat di `IMPLEMENTATION_DECISIONS.md` poin 43.

Sumber: Gao et al. (2023), *Text-to-SQL Empowered by Large Language Models: A Benchmark Evaluation*, dan repo `github.com/beachwang/DAIL-SQL` (commit `2061f68`, 2025-03-07).

---

## 1. Arsitektur DAIL-SQL yang sebenarnya

DAIL-SQL **tidak memangkas schema**: seluruh schema DB selalu dikirim ke LLM. Kekuatannya ada pada **cara memilih contoh few-shot** (DAIL Selection) dan **cara menyusunnya di prompt** (DAIL Organization).

### 1.1 Alur

```
                 ┌──────────────── PREPROCESSING (sekali) ─────────────────┐
dev.json ───────►│ Schema linking ala RAT-SQL (CoreNLP + GloVe 42B)        │
train_spider +   │  • sc_link: token pertanyaan yang cocok nama tabel/kolom │
train_others ───►│  • cv_link: token yang cocok nilai sel / angka          │
                 └───────────────────────┬─────────────────────────────────┘
                                         ▼
Pertanyaan ──► [1] Question masking
                   "How many singers are from France?" → "How many <mask> are from <unk>?"
                   (token yang cocok schema → <mask>, token yang cocok nilai → <unk>)
                                         ▼
               [2] PASS 1 — seleksi EUCDISQUESTIONMASK
                   embedding all-mpnet-base-v2 dari pertanyaan ter-mask,
                   ambil 9 contoh train terdekat (jarak Euclidean)
                                         ▼
               [3] Prompt pass 1 → LLM → SQL awal (preliminary SQL)
                                         ▼
               [4] Skeleton SQL awal, mis. "select count ( _ ) from _ where _"
                                         ▼
               [5] PASS 2 — DAIL Selection (EUCDISMASKPRESKLSIMTHR)
                   urutkan contoh by jarak pertanyaan ter-mask, DAHULUKAN contoh
                   yang skeleton SQL-nya mirip SQL awal (Jaccard ≥ 0.85);
                   sisa kuota diisi contoh terdekat lain
                                         ▼
               [6] Prompt pass 2 → LLM → SQL final
                                         ▼
               [7] Cleaner: rapikan spasi, potong di "/*", awali "SELECT"
```

### 1.2 Bentuk prompt (sama untuk pass 1 dan pass 2)

```
/* Some SQL examples are provided based on similar problems: */      ← DAIL Organization:
/* Answer the following: How many aircrafts do we have? */              hanya pasangan
SELECT count(*) FROM Aircraft                                           pertanyaan–SQL,
... (9 contoh)                                                          tanpa schema contoh

/* Given the following database schema: */                         ← Code Representation:
CREATE TABLE singer ( ... )                                            CREATE TABLE seluruh DB
CREATE TABLE concert ( ..., FOREIGN KEY ... )                          dari sqlite_master

/* Answer the following: How many singers do we have? */
SELECT                                                              ← LLM melanjutkan dari sini
```

### 1.3 Konfigurasi resmi DAIL-SQL (`run_dail_sql.sh`)

| Komponen | Nilai |
|---|---|
| Representasi | `SQL` (Code Representation, seluruh schema) |
| Organisasi | `QA` (pertanyaan + SQL saja) |
| Seleksi | pass 1 `EUCDISQUESTIONMASK`, pass 2 `EUCDISMASKPRESKLSIMTHR` |
| k | 9 contoh |
| Pool contoh | `train_spider` + `train_others` (8.659 pertanyaan) |
| Embedding seleksi | `all-mpnet-base-v2` (CPU), jarak Euclidean |
| Threshold skeleton | Jaccard 0.85 |
| LLM | GPT-4, maks. 200 token |
| Self-consistency | opsional (5 sampel, suhu 1.0, voting hasil eksekusi) |
| Evaluasi di paper | Spider **test**, EM + EX (test-suite) |

---

## 2. Yang diubah dan disesuaikan supaya apple-to-apple

Prinsip: **bagian arsitektur ikut DAIL apa adanya; bagian model dan evaluasi ikut pipeline skripsi.** Yang dibandingkan adalah best config vs best config: best config DAIL dari paper-nya sendiri, dan best config 1-Stage/2-Stage dari trial dan ablation skripsi.

### 2.1 Tetap sama dengan DAIL (arsitektur tidak disentuh)

| Komponen | Status |
|---|---|
| Preprocessing schema linking dan masking | Fungsi DAIL asli |
| Prompt: representasi `SQL`, organisasi `QA`, urutan isi prompt | Kelas prompt DAIL asli |
| Seleksi contoh pass 1 dan pass 2, mpnet, threshold 0.85, k = 9 | Kelas selector DAIL asli |
| Pool contoh `train_spider` + `train_others` | Sama |
| DDL dari `sqlite_master` (seluruh DB) | Fungsi DAIL asli |
| Cleaner SQL | Disalin apa adanya dari `ask_llm.py` |

### 2.2 Diganti: model

| | DAIL asli | Skripsi | Alasan |
|---|---|---|---|
| LLM pass 2 | GPT-4 | Qwen2.5-Coder sesuai `MODEL`, 4-bit NF4 | Sama dengan 1-Stage/2-Stage |
| LLM pass 1 (SQL awal) | GPT-4 (README juga menyebut Graphix) | **Qwen yang sama** | Tidak ada model lain yang membantu DAIL |
| Cara memanggil | API chat OpenAI | Prompt mentah tanpa chat template, fungsi load model yang sama | Persis cara pipeline skripsi memanggil Qwen |

### 2.3 Pengaturan generate (disamakan dengan pipeline skripsi)

| | DAIL asli | Skripsi | Alasan |
|---|---|---|---|
| Decoding | suhu 0, voting opsional | greedy, voting mati | Deterministik, sama dengan 1-Stage/2-Stage |
| Panjang output | 200 token | `max_new_tokens` = 256 | Sama dengan 1-Stage/2-Stage |
| Stop | `;` di mode completion DAIL | `;` | Prompt DAIL diakhiri `SELECT `, jadi Qwen dipakai seperti model completion. Tanpa stop, Qwen terus menulis contoh baru dan membuang token |

### 2.4 Data (disamakan)

| | DAIL asli | Skripsi |
|---|---|---|
| Set evaluasi | Spider test | Spider **dev**, sampel stratified yang **sama** dengan `eval_pipeline.ipynb` (`SAMPLE_RATIO`, seed) |
| ID pertanyaan | — | `dev_0001`, ... sama dengan raw_logs 1-Stage/2-Stage, jadi bisa dibandingkan per query |

### 2.5 Evaluasi (fungsi yang sama)

| Metrik | Cara pada DAIL | Sama dengan 1-Stage/2-Stage? |
|---|---|---|
| ESM, EX, CM | `SqlEvaluator` (`src/metrics/esm_ex_cm.py`) | Ya, fungsi yang sama |
| SLA | `predicted_schema` = semua kolom DB, karena DAIL mengirim seluruh schema | Ya, rumus yang sama. Recall tinggi, precision rendah: memang karakter DAIL |
| Token | T = T_in + α·T_out, dengan T_in/T_out = **pass 1 + pass 2** | Rumus sama. Dihitung dua kali karena DAIL memang memanggil LLM dua kali |
| TEP | Rumus sama, dengan **DAIL sebagai B**: 1-Stage vs DAIL dan 2-Stage vs DAIL | Rumus sama, pembanding baru |
| QVT | Grup parafrase yang sama dari `dev.json` (hanya run penuh) | Ya |
| Reduksi kolom/token | `prompt_stats()` dengan DDL yang benar-benar masuk prompt DAIL | Ya |
| Evaluator resmi Spider | `evaluation.py` lewat fungsi yang sama | Ya |
| Format raw_logs | `_log_entry()` yang sama, ditambah field `dail` | Ya |

### 2.6 Latensi (nama tahap disamakan)

| Tahap | Pada DAIL |
|---|---|
| `retrieval_search` | 0, karena DAIL tidak punya retrieval schema |
| `build_ddl` | Ambil CREATE TABLE seluruh DB |
| `few_shot` | Masking + seleksi contoh mpnet |
| `build_prompt`, `prompt_tokenize`, `llm_generate`, `sql_clean` | Artinya sama dengan pipeline skripsi |
| `end_to_end` | Rumus yang sama, dijumlah atas pass 1 + pass 2 |
| `dail_pass1`, `dail_pass2` | Tambahan: total per pass, tidak ikut dijumlah lagi |
| `dail_setup` (sekali per run) | Baca linking + buat index mpnet, setara `build_indexes` pipeline skripsi |

### 2.7 Cara menjalankan diubah, perilakunya tidak

DAIL asli dijalankan sebagai **script CLI per tahap**: buat semua prompt pass 1 untuk seluruh dev, panggil GPT-4 untuk semuanya, lalu baru buat prompt pass 2. Di skripsi, kelas prompt dan selector DAIL dipanggil **per pertanyaan** (pass 1 → pass 2 langsung). Hasilnya sama, tapi ada dua keuntungan:

- sampel stratified yang sama dengan run 1-Stage/2-Stage bisa dipakai;
- latensi per pertanyaan bisa diukur dengan tahap yang sama.

### 2.8 Perbedaan yang sengaja dibiarkan (ditulis di metodologi)

1. **Pool contoh DAIL lebih besar** (`train_spider` + `train_others`) dibanding 1-Stage/2-Stage (`train_spider`). Ini bagian dari arsitektur DAIL. Keduanya tetap retrieval dari train set, bukan kebocoran dev.
2. **Embedding seleksi DAIL = mpnet**, bukan BGE-M3.
3. **k = 9**, bukan k hasil ablation skripsi. Itu best config DAIL sendiri.
4. **Konfigurasi DAIL dicari untuk GPT-4**, jadi belum tentu optimal untuk Qwen (keterbatasan). Baseline tidak wajib dituning ulang.
5. **DDL DAIL dibaca dari file `.sqlite`**, sedangkan SLA memakai `tables.json`. Isi kolomnya sama; yang berbeda hanya teks DDL-nya.
6. **`max_new_tokens` 256** (DAIL 200) dan stop `;`.

---

## 3. Isi tiap file

### 3.1 `external/dail_sql/` — kode DAIL asli (TIDAK diedit)

Salinan commit `2061f68`, tanpa `.git`, `img/`, dan `results/`. Sama perlakuannya dengan `external/spider_eval/`.

| File | Fungsi | Dipakai? |
|---|---|---|
| `data_preprocess.py` | Schema linking (`schema_linking_producer()`) | Ya, lewat `dail_preprocess.py` |
| `prompt/prompt_builder.py` | `prompt_factory()`: merakit kelas prompt dari representasi + organisasi + selector | Ya |
| `prompt/PromptReprTemplate.py` | `SQLPrompt`: bagian "Given the following database schema" | Ya |
| `prompt/ExampleFormatTemplate.py` | `QuestionSqlExampleStyle`: format contoh QA | Ya |
| `prompt/ExampleSelectorTemplate.py` | Semua selector, termasuk dua yang dipakai DAIL-SQL | Ya |
| `prompt/PromptICLTemplate.py` | `format()`: menggabungkan contoh + schema + pertanyaan | Ya |
| `utils/data_builder.py` | Membaca dev/train + linking, menghitung skeleton | Ya, lewat subclass |
| `utils/utils.py` | `get_sql_for_database()` (DDL), `sql2skeleton()`, `jaccard_similarity()` | Ya |
| `utils/linking_utils/` | Masking pertanyaan, CoreNLP | Ya |
| `utils/post_process.py` | `process_duplication()` (potong di `/*`) | Ya |
| `generate_question.py`, `ask_llm.py`, `llm/chatgpt.py`, `run_*.sh` | Script CLI dan pemanggil OpenAI | **Tidak**. Logikanya dipanggil langsung per pertanyaan; cleaner `ask_llm.py` disalin |

### 3.2 `src/retrieval/dail.py` — wrapper DAIL (kode custom skripsi)

Pola yang sama dengan `external/spider_eval/` yang dibungkus `src/metrics/`.

| Fungsi | Isi |
|---|---|
| `import_dail()` | Memasukkan `external/dail_sql/` ke `sys.path` dan meng-import paket DAIL (`utils`, `prompt`). Berhenti dengan pesan jelas kalau ada paket lain bernama sama |
| `check_preprocessed()` | Memastikan `data/dail/` ada dan jumlah baris linking = jumlah pertanyaan (DAIL mencocokkannya per posisi) |
| `_make_dataset()` | Subclass dataset DAIL yang hanya mengganti lokasi file (Kaggle input read-only + `data/dail/`) |
| `load_dail()` | Sekali per run: baca DB, dev, dan train beserta linking, lalu buat objek prompt pass 1 dan pass 2 (index mpnet). Memasang pencatat waktu `build_ddl` dan `few_shot` tanpa mengubah hasil |
| `run_single_dail()` | Inti per pertanyaan: pass 1 → SQL awal → `sql2skeleton()` → pass 2 → SQL final. Mengembalikan SQL, token (jumlah kedua pass), rincian per pass, dan prompt |
| `_one_pass()` | Satu pass: `format()` DAIL → generate → cleaner → hitung T_in |
| `_generate()` | Panggilan Qwen: greedy, `max_new_tokens`, stop `;`, sama seperti `generate_sql_with_token_count()`, tapi mengembalikan teks mentah supaya dibersihkan cleaner DAIL |
| `dail_postprocess()` | Cleaner yang disalin dari `ask_llm.py` |
| `schema_context()` | Teks CREATE TABLE yang masuk prompt, untuk `schema_tokens` |
| `finish_latency()` | Mengurangkan `build_ddl` dan `few_shot` dari `build_prompt` supaya `end_to_end` tidak menghitung dua kali |

### 3.3 `src/experiments/dail_preprocess.py` — preprocessing sekali

1. Gabungkan `train_spider.json` + `train_others.json` menjadi `data/dail/train_spider_and_others.json`.
2. Panggil `schema_linking_producer()` DAIL asli, hasilnya `data/dail/enc/{test,train}_schema-linking.jsonl`.
3. Cek jumlah baris (1034 dev, 8659 train).

Butuh Java, CoreNLP 2018-10-05, `torchtext` (GloVe 42B, ~5 GB), `bpemb`, jadi dijalankan di environment terpisah. Hasilnya di-upload sebagai Kaggle Dataset.

```bash
python src/experiments/dail_preprocess.py --spider-dir <folder spider_data> --out-dir data/dail
```

### 3.4 `src/experiments/dail_steps.py` — langkah notebook DAIL

| Kelompok | Fungsi | Memakai ulang |
|---|---|---|
| Persiapan | `show_run_config`, `load_llm`, `load_dail`, `columns_per_table` | `load_model_and_tokenizer()` |
| Contoh | `walk_through()`: prompt pass 1, SQL awal, skeleton, prompt pass 2, SQL final | `SqlEvaluator` |
| Loop | `run_all_questions()`, `_run_one()`, `_evaluate_and_log()` | `_log_entry()`, `prompt_stats()`, `timing` |
| Simpan generate | `save_generation_results()`: prediksi, prompt (audit), `dail_log.json` | — |
| Metrik | `compute_metrics()`, `load_comparison_logs()` (log 1-Stage/2-Stage dipakai hanya kalau pertanyaannya sama persis), `build_dail_qvt()` | `metrics_summary._sla/_accuracy/_cm/_token/_qvt/_tep`, `build_groups()` |
| Tabel | `summary_table`, `by_difficulty_table`, `sla_table`, `cm_table`, `token_tables` (termasuk per pass), `qvt_table`, `prediction_checks`, `per_query_table`, `latency_tables`, `latency_per_query` | Format `result_tables.py` |
| Evaluator resmi | `run_spider_check()` | `_run_spider_for()` |
| Simpan | `write_workbook()`, `save_all_results()`: JSON, Excel, CSV, latensi, zip Kaggle ke `tables/dail/` | `Sheet` dari `export_excel.py` |

### 3.5 `notebooks/eval_dail.ipynb` — notebook

| Bagian | Isi |
|---|---|
| Pembuka | Tabel "DAIL asli vs skripsi" dan "evaluasi sama dengan eval_pipeline" |
| 0 | Cara preprocessing (opsional `RUN_PREPROCESS`) |
| 1-3 | Setup, import, konfigurasi (`MODEL`, `SAMPLE_RATIO`, `DAIL_DATA_DIR`) |
| 4-7 | Data, LLM, siapkan DAIL, evaluator |
| 8 | Contoh satu pertanyaan |
| 9-10 | Loop, lalu simpan prediksi dan raw_logs |
| 11 | Metrik: ringkasan, per tingkat kesulitan, SLA, CM, token + TEP, QVT, latensi |
| 12 | Evaluator resmi Spider |
| 13 | Simpan semua |
| Penutup | Catatan metodologi |

Jalankan dengan `MODEL` dan `SAMPLE_RATIO` yang sama dengan `eval_pipeline.ipynb` supaya perbandingannya berpasangan.

### 3.6 File hasil run DAIL

Semua terpisah dari hasil 1-Stage/2-Stage, jadi tidak saling menimpa.

| File | Isi |
|---|---|
| `predictions/dail_predictions.txt` | SQL final, satu per baris |
| `predictions/dail_prompts.jsonl` | Prompt dan output mentah kedua pass (audit) |
| `raw_logs/dail_log.json` | Catatan per pertanyaan, skema sama dengan `baseline_log.json`/`graphrag_log.json`, ditambah field `dail` (SQL awal, skeleton, token per pass) |
| `tables/dail/dail_metrics_results.json` | Metrik semua kondisi yang dibandingkan + pengaturan DAIL |
| `tables/dail/hasil_dail.xlsx` | Semua tabel notebook |
| `tables/dail/spider_official_eval.csv` | Skor evaluator resmi |
| `tables/dail/latency_run.json` | Waktu sekali-jalan, GPU, puncak VRAM |
| `qvt_variations/dail_qvt.json` | Data QVT DAIL (run penuh saja) |

### 3.7 File lama yang diubah

| File | Perubahan |
|---|---|
| `src/core/config.py` | Tambah field `dail_*` (nilai DAIL asli + stop `;`). Nilai lama tidak berubah |
| `.gitignore` | Abaikan cache GloVe (`data/dail_cache/`) dan CoreNLP (`external/dail_sql/third_party/`) |
| `CLAUDE.md` | Struktur file, status baseline, TBD, perintah CLI |
| `context/IMPLEMENTATION_DECISIONS.md` | Poin 43 dan status open item F |
| `context/RESEARCHER_TODO.md` | Langkah yang harus dijalankan peneliti |

---

## 4. Dependensi dan catatan teknis

- **Saat run (Kaggle):** `sentence-transformers`, `transformers`, `bitsandbytes`, `nltk` (punkt, stopwords), `sqlparse`, dan **`sql_metadata<3`**. `sql_metadata` 3.x mengubah API token dan membuat `sql2skeleton()` DAIL crash (sudah diverifikasi), jadi dipin ke 2.x.
- **Saat preprocessing (sekali):** Java + Stanford CoreNLP 2018-10-05, `stanford-corenlp`, `torchtext` (GloVe 42B), `bpemb`, `attrs`, `nltk`. `torchtext` sering bentrok dengan versi torch di Kaggle, jadi sebaiknya environment terpisah.
- **Status pengujian:** smoke test lokal memakai kode prompt/selector DAIL asli dan data Spider asli, tapi linking sintetis, encoder palsu, dan LLM palsu. Hasilnya: prompt 9 contoh yang diakhiri `SELECT `, skeleton, cleaner, token per pass, latensi tanpa hitung ganda, raw_logs, metrik, TEP, dan Excel berjalan benar. **Belum dijalankan:** preprocessing asli (CoreNLP), mpnet asli, dan Qwen di GPU.

## 5. Belum diputuskan

- Penggabungan tiga kondisi (DAIL, 1-Stage, 2-Stage) dalam satu Excel, Dimensi 1-6, dan `combine_models.py`.
- Pasangan TEP yang dilaporkan di skripsi (1-Stage vs DAIL, 2-Stage vs DAIL, dan/atau 2-Stage vs 1-Stage).
