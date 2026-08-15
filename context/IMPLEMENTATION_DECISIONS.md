# Implementation Decisions — Evaluation Pipeline

> **Tujuan file ini:** `EVALUATION_ANALYSIS_GUIDE.md` adalah source of truth untuk formula/threshold, dan melarang mengubahnya tanpa konfirmasi eksplisit peneliti (lihat Bagian 4 poin 1-2 di situ). File ini adalah **catatan konfirmasi tersebut** — setiap kali implementasi menemukan ambiguitas, konflik dengan kode yang sudah ada, atau konflik dengan apa yang benar-benar bisa dilakukan tooling resmi SPIDER, keputusan peneliti dicatat di sini beserta alasannya. Kalau ada sesi Claude Code berikutnya yang mau mengubah salah satu keputusan ini, **baca alasannya dulu di sini**, jangan asumsikan itu oversight.

Format tiap entri: **Konflik/ambiguitas** → **Keputusan** → **Alasan** → **Lokasi implementasi**.

---

## 1. Formula Token Consumption: μ = 3 (bukan α = 1.0)

**Konflik:** `EVALUATION_ANALYSIS_GUIDE.md` Bagian 1.5 fix `T = T_in + μ·T_out` dengan `μ = 3` dari proposal (subbab 3.8.2.5), "jangan diubah kecuali peneliti eksplisit minta". Tapi `PipelineConfig.token_output_weight` (dipanggil α di kode) di `src/core/config.py` sudah punya default `1.0` "untuk local model", dan nilai ini SUDAH dipakai live di `src/experiments/ablation.py` untuk menghitung `avg_token_consumption`.

**Keputusan (2026-08-08):** μ = 3 menang. `PipelineConfig.token_output_weight` diupdate dari `1.0` → `3.0`.

**Alasan:** μ = 3 adalah nilai fixed dari proposal skripsi (subbab 3.8.2.5) — bukan sekadar default kode yang bisa dipilih bebas.

**Catatan:** belum ada data ablation nyata tersimpan di `outputs/` saat keputusan ini dibuat, jadi tidak ada hasil lama yang perlu di-generate ulang dengan nilai baru.

**Lokasi implementasi:** `src/core/config.py` (`token_output_weight: float = 3.0`), `src/experiments/ablation.py` (`_print_table()`'s fallback juga diupdate untuk selalu mengikuti `PipelineConfig()` default, bukan hardcode `1.0` lagi), `src/metrics/token_consumption.py` (`MU = 3`, dengan catatan HARUS tetap sama dengan `config.py`).

---

## 2. `external/spider_eval/` — reuse root, bukan duplikat

**Konteks:** folder template `evaluation_pipeline/` (dari sesi brainstorming terpisah, lihat entri 3 di bawah) datang dengan `external/spider_eval/README.md`-nya sendiri (instruksi download manual, belum ada file asli). Root project sudah punya `external/spider_eval/evaluation.py` + `process_sql.py` yang asli (sudah di-download sebelumnya).

**Keputusan:** hapus folder `external/spider_eval/` versi template, seluruh `src/metrics/` reuse `external/spider_eval/` yang di root.

**Alasan:** satu copy, tidak ada risiko dua file `evaluation.py`/`process_sql.py` yang beda versi saling drift.

---

## 3. `src/metrics/`, `src/dimensions/`, `src/utils/` — pindah ke `src/` root, bukan folder terpisah

**Konteks:** `evaluation_pipeline/` adalah folder template yang dibuat Claude di sesi brainstorming SEBELUM melihat codebase asli ini (metrik/dimensi dirancang blind, tanpa tahu struktur `src/core/`, `src/retrieval/`, dst. yang sudah ada). Root `CLAUDE.md` versi lama menyebut `evaluation_pipeline/` sebagai "proyek terpisah".

**Keputusan:** dibongkar total — `metrics/`, `dimensions/`, `utils/` masuk ke `src/` (sejajar dengan `core/`, `retrieval/`, `generation/`, `experiments/`), `run_all_dimensions.py` masuk ke `src/experiments/`, `data/` (raw_logs, db_schema, qvt_variations) naik ke root, `requirements.txt` naik ke root. `context/` folder HANYA berisi `EVALUATION_ANALYSIS_GUIDE.md` (dan file ini).

**Alasan:** metrik/dimensi analisis adalah bagian integral dari satu pipeline penelitian yang sama, bukan proyek yang benar-benar independen — import lintas `src.*` jauh lebih bersih daripada dua package terpisah yang saling bergantung.

---

## 4. Raw-logs pipeline: diimplementasi penuh, bukan cuma plumbing shell

**Konteks:** awalnya "bikin raw_logs pipeline" dikira cuma soal orkestrasi (loop + tulis JSON). Ternyata field `esm_result`/`ex_result`/`cm_per_clause`/`gold_schema`/`difficulty` di skema raw_logs TIDAK BISA diisi tanpa implementasi nyata ESM/EX/CM (`src/metrics/esm_ex_cm.py`) dan ekstraksi gold schema pakai SQL parser (`src/metrics/sla.py`'s `extract_ground_truth_schema()`) — dua-duanya sebelumnya stub.

**Keputusan:** implementasi penuh, bukan stub dengan placeholder null.

**Alasan:** tanpa ini, `data/raw_logs/*.json` tidak bisa dipakai sama sekali oleh `src/dimensions/` nanti — bukan "plumbing yang bisa diselesaikan belakangan", tapi prasyarat keras.

**Lokasi implementasi:** `src/metrics/esm_ex_cm.py` (penuh: `evaluate_single_query()`, `build_kmaps()`, `build_schema_for_db()`, `aggregate_esm/ex/cm()`), `src/metrics/sla.py` (`extract_ground_truth_schema()` saja — `compute_sla()`/`aggregate_sla()` masih stub, baru dibutuhkan di Dimensi 5), `src/utils/schema_utils.py` (penuh), `src/experiments/pipeline.py`'s `run_comparison()` (memanggil semuanya per-query, tulis `data/raw_logs/{graphrag,baseline}_log.json`), `run_single()`/`run_single_baseline()` (diperluas untuk juga menghitung token_input/token_output).

**Sudah divalidasi:** sanity test end-to-end terhadap database SQLite sintetis (bukan cuma syntax check) — ESM/EX/CM/gold_schema/JSON serialization semua diverifikasi berperilaku benar, termasuk kasus predicted SQL invalid, alias table, dan literal WHERE yang beda (ESM tetap match, EX tidak). **Belum pernah dijalankan terhadap Spider dev set asli / di Kaggle** — `torch` tidak tersedia di environment dev lokal.

---

## 5. EX order-sensitivity: pakai semantics resmi SPIDER apa adanya

**Konflik:** `EVALUATION_ANALYSIS_GUIDE.md` Bagian 1.3(c) langkah 3 minta EX dibandingkan **order-insensitive** (set of tuples) KECUALI gold SQL punya `ORDER BY` eksplisit, yang berarti order jadi relevan. Tapi `eval_exec_match()` resmi SPIDER (`external/spider_eval/evaluation.py` baris 614) yang dipakai `src/metrics/esm_ex_cm.py` SELALU order-sensitive — dicek langsung dari source resmi: `res_map()` membangun `list` per kolom SELECT langsung dari urutan `cursor.fetchall()`, lalu membandingkan `dict` berisi list itu. Tidak ada `sorted()` atau `set()` di mana pun di fungsi itu, dan tidak ada logic yang mengecek `gold['orderBy']`. Guide sendiri mensyaratkan dua hal yang berkontradiksi (1.3 poin b: "reuse parser resmi SPIDER" vs 1.3 poin c langkah 3: order-insensitivity custom) — resmi SPIDER tidak melakukan yang kedua.

**Keputusan (2026-08-08):** tetap pakai `eval_exec_match()` resmi apa adanya.

**Alasan:** angka EX tetap directly comparable dengan "Spider EX" yang dilaporkan paper/sistem lain yang juga pakai evaluator resmi ini.

**Risiko yang diterima (false negative):** query yang secara relasional/isi setara tapi dikembalikan SQLite dengan urutan baris fisik berbeda (mis. tidak ada `ORDER BY` di kedua query, tapi `JOIN`/`WHERE` disusun beda sehingga query planner SQLite memilih execution plan berbeda) bisa ke-score EX=0 walau sebenarnya benar. Sifatnya plan-dependent/non-deterministic (coba direproduksi manual di dua contoh kecil, tidak selalu muncul — SQLite kadang tetap mengembalikan urutan yang sama meski query berbeda struktur). Kemungkinan lebih sering muncul di database Spider asli yang lebih besar/kompleks daripada di toy example.

**Implikasi untuk skripsi ini secara spesifik:** GraphRAG (context dipangkas) dan Baseline (full schema, tanpa pruning) memberi LLM konteks yang sangat berbeda, jadi LLM kemungkinan menghasilkan gaya SQL yang berbeda (struktur JOIN, subquery vs join eksplisit) di antara dua kondisi bahkan untuk pertanyaan yang sama-sama "benar" — risiko false-negative dari isu ini TIDAK NETRAL antar dua kondisi, berpotensi jadi confound saat membandingkan EX_G vs EX_B. Belum ada mitigasi untuk ini.

**Lokasi implementasi:** `src/metrics/esm_ex_cm.py` (`ex = 1 if eval_exec_match(...) else 0`, dengan komentar panjang menjelaskan keputusan ini persis di baris tersebut). `src/utils/sql_execution.py`'s `compare_execution_results()` (yang akan mengimplementasi order-insensitivity literal sesuai guide) SENGAJA dibiarkan stub, tidak dipakai di manapun — jangan wire tanpa instruksi baru yang membalik keputusan ini.

---

## 6. Difficulty label: `"extra"` (konvensi SPIDER), bukan `"extra_hard"` (teks literal guide)

**Konflik:** skema raw_logs di `EVALUATION_ANALYSIS_GUIDE.md` Bagian 2 menulis `"difficulty": "easy|medium|hard|extra_hard"`. Tapi `Evaluator.eval_hardness()` resmi SPIDER (`external/spider_eval/evaluation.py` baris 362-377) — yang dipanggil verbatim oleh `evaluate_single_query()` — return string `"extra"`, bukan `"extra_hard"`.

**Keputusan (2026-08-08):** ikuti konvensi resmi SPIDER, `"extra"`. Tidak ada relabeling di kode.

**Alasan:** ini cuma perbedaan label/penamaan (bukan perbedaan formula/threshold penentuan level kesulitan), dan tetap konsisten dengan cara SPIDER benchmark dilaporkan di literatur.

**Lokasi implementasi:** `src/metrics/esm_ex_cm.py`'s `QueryEvalResult.difficulty` — nilai apa adanya dari `evaluator.eval_hardness()`, tidak di-remap.

---

## 7. CM clause key casing: lowercase snake_case, bukan uppercase

**Konflik:** `EVALUATION_ANALYSIS_GUIDE.md` menulis nama klausa dengan uppercase di teks algoritma (mis. "SELECT, FROM, WHERE, GROUP BY, ORDER BY...") dan di contoh skema raw_logs Bagian 2 (`"cm_per_clause": {"SELECT": 0|1, "FROM": 0|1, ...}`). Implementasi (`CLAUSES` constant di `src/metrics/esm_ex_cm.py`, sudah ada sejak stub awal sebelum sesi ini) pakai lowercase snake_case: `select, from, where, group_by, order_by, having, keywords, union, union_all, intersect, except`.

**Keputusan (2026-08-08):** pertahankan lowercase snake_case yang sudah ada, tidak di-rename ke uppercase.

**Alasan:** konsisten dengan konvensi penamaan key JSON/Python di seluruh codebase (semua field lain di raw_logs — `query_id`, `db_id`, `predicted_sql`, dst. — juga snake_case lowercase, bukan mengikuti casing guide secara literal).

**Lokasi implementasi:** `src/metrics/esm_ex_cm.py`'s `CLAUSES` constant dan `_PARTIAL_SCORE_KEY` mapping.

---

## 8. CM klausa `union_all`: dibiarkan `null`, tidak diaproksimasi

**Konflik:** guide 1.4 minta `union_all` dinilai sebagai klausa terpisah (0 atau 1) sama seperti klausa lain. Parser resmi SPIDER (`process_sql.py`'s `SQL_OPS = ('intersect', 'union', 'except')`) tidak membedakan `UNION` dari `UNION ALL` secara struktural — keduanya di-parse jadi node `union` yang sama, tidak ada informasi apakah `ALL` dipakai.

**Keputusan:** `cm_per_clause["union_all"]` selalu `None` (bukan `0`/`1` hasil approksimasi/tebakan).

**Alasan:** guide sendiri melarang "mengarang formula baru" — memaksa angka 0/1 dari data yang parsernya tidak punya berarti mengarang, bukan menghitung. `None` secara eksplisit menandakan "tidak bisa dihitung dari tooling ini", bukan "selalu gagal" (0) atau "selalu berhasil" (1) yang keduanya salah.

**Lokasi implementasi:** `src/metrics/esm_ex_cm.py`'s `_evaluate_single_query_inner()` dan `aggregate_cm()` (skip `None` saat agregasi, bukan dihitung sebagai 0).

---

## 9. Token Consumption: ints-only, tanpa tokenizer sendiri; TEP epsilon = ±0.05

**Konflik/ambiguitas (a):** stub asli `src/metrics/token_consumption.py` dirancang untuk memuat tokenizer Qwen2.5-Coder-7B-Instruct sendiri (`get_tokenizer()`/`count_tokens()`) dan menghitung `T_in`/`T_out` dari re-encode raw prompt/output **text**. Tapi `generate_sql_with_token_count()` (`src/generation/generation.py` baris 180) sudah menghitung `T_in`/`T_out` sebagai **int** di titik generasi, pakai tokenizer yang SUDAH dimuat (bukan instance baru) — hasilnya sudah tersimpan sebagai int di `data/raw_logs/*.json` (`token_input`/`token_output`) dan `outputs/tables/ablation_results.csv`. Tidak ada raw prompt/output text yang disimpan di raw_logs untuk di-re-tokenize belakangan.

**Keputusan (2026-08-15):** `compute_token_consumption()` di ubah jadi murni aritmetika atas int yang sudah ada (`T = T_in + mu*T_out`), `get_tokenizer()`/`count_tokens()` dihapus total dari `token_consumption.py`.

**Alasan:** memuat tokenizer kedua di `token_consumption.py` akan redundan (satu instance sudah dipakai live di `pipeline.py`/`ablation.py`) dan tidak pernah dipanggil siapa pun di alur nyata — `dim1_efficiency.py` selalu bekerja dari raw_logs yang sudah berisi int, bukan teks mentah.

**Konflik/ambiguitas (b):** guide Bagian 3 Dimensi 1 mendefinisikan tabel interpretasi TEP dengan tiga band (`TEP < 0`, `TEP ≈ 0`, `TEP > 0`) tapi tidak memberi angka eksplisit untuk lebar band "≈ 0" — beda dengan threshold lain di guide (±2% QVT, 80%/65% F1/EX Dimensi 5) yang eksplisit "WAJIB, jangan diubah".

**Keputusan (2026-08-15):** epsilon = ±0.05 sebagai default `compute_tep(..., epsilon=0.05)` di `src/metrics/tep.py`.

**Alasan:** skala kecil, konsisten dengan urutan besaran threshold ±2% QVT di guide Dimensi 4. Dioper sebagai parameter (bukan konstanta buta di dalam fungsi) supaya bisa direvisi peneliti tanpa mengubah signature kalau nanti ada angka lain yang lebih tepat.

**Lokasi implementasi:** `src/metrics/token_consumption.py` (`compute_token_consumption(token_input: int, token_output: int, mu)`, `aggregate_token_consumption()`), `src/metrics/tep.py` (`compute_tep()`, `DEFAULT_EPSILON = 0.05`).

---

## 10. QVT "is_correct": pakai EX, bukan ESM

**Konflik/ambiguitas:** guide 1.6 poin (e) eksplisit minta "benar" per variasi NL question didefinisikan pakai EX ATAU ESM — "tentukan salah satu secara konsisten" — tapi tidak memutuskan yang mana.

**Keputusan (2026-08-15):** EX.

**Alasan:**
1. QVT mengukur *stabilitas jawaban fungsional* terhadap parafrase (guide 1.6 poin (a): "apakah model tetap menghasilkan SQL yang benar" secara hasil, bukan secara struktur persis) — EX cocok dengan tujuan ini, ESM tidak (ESM sensitif ke struktur, bukan hasil).
2. Open item A di file ini (parser resmi SPIDER tidak bisa parse `LEFT JOIN`/`RIGHT JOIN` — `KeyError: 'left'`) akan menghantam ESM/CM lebih parah khusus untuk QVT: parafrase pertanyaan (mis. "siswa yang tidak punya nilai" vs "siswa dengan nilai") kemungkinan besar justru MEMICU LLM memilih idiom SQL berbeda seperti `LEFT JOIN` — variasi yang secara fungsional benar akan otomatis gagal ESM/CM murni karena parser-nya crash, bukan karena SQL-nya salah. EX (`eval_exec_match()`) tidak butuh parsing sama sekali (langsung eksekusi SQLite), jadi tidak kena isu ini.
3. Trade-off yang diterima: EX mewarisi isu order-sensitivity yang sudah didokumentasikan di poin 5 di atas — tapi itu bukan risiko baru, sudah accepted risk untuk EX di seluruh skripsi ini.

**Lokasi implementasi:** `src/metrics/qvt.py` (docstring modul menegaskan kontrak data `"is_correct"` = hasil EX; `compute_qvt_per_query()`/`aggregate_qvt()` sendiri cuma mengonsumsi field itu, tidak menghitung EX/ESM sendiri). **Belum ada** kode yang benar-benar mengisi `data/qvt_variations/*.json` (generator/runner untuk itu belum ditulis) — sesi berikutnya yang membangun generator itu WAJIB memanggil `src.metrics.esm_ex_cm.evaluate_single_query()` dan mengambil field `.ex`, bukan `.esm`, supaya konsisten dengan keputusan ini.

---

## Belum diputuskan / open items

### A. `LEFT JOIN` / `RIGHT JOIN` / `INNER JOIN` tidak didukung parser resmi SPIDER

**Temuan (2026-08-08):** `process_sql.py`'s `parse_from()` cuma mengenali token `join` (bare), tidak ada branch untuk `left`/`right`/`inner`. Predicted SQL yang pakai keyword-keyword itu GAGAL di-parse (`KeyError: 'left'` dkk, sudah diverifikasi langsung), jatuh ke fallback `_empty_sql()` yang sama dengan predicted SQL yang benar-benar rusak → ESM=0 otomatis, EX biasanya ikut 0 (karena `p_val_units` jadi kosong).

**Dampak potensial:** karena Qwen2.5-Coder (atau LLM manapun) sangat mungkin menghasilkan `LEFT JOIN` secara alami (idiom SQL yang sangat umum, kadang justru pilihan yang lebih tepat secara semantik untuk pertanyaan "which X have no Y"), ini berpotensi jadi sumber false-negative yang LEBIH besar dan lebih sistematis daripada isu order-sensitivity di poin 5.

**Catatan teknis:** `INNER JOIN` → `JOIN` aman dinormalisasi sebelum parsing (100% setara secara semantik di SQL standar). `LEFT JOIN`/`RIGHT JOIN` TIDAK aman dinormalisasi ke `JOIN` biasa — keduanya mempertahankan baris yang tidak match dengan `NULL`, beda perilaku dari inner join, jadi normalisasi buta akan mengubah semantik yang diukur, bukan cuma perbaikan kompatibilitas parser.

**Status:** belum diputuskan mau diapakan (dibiarkan sebagai known limitation vs investigasi seberapa sering LLM benar-benar memakai keyword ini di praktik). **Tanyakan ke peneliti sebelum mengambil tindakan apapun di sini.**
