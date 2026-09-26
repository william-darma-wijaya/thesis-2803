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

**Implikasi untuk skripsi ini secara spesifik:** GraphRAG (context dipangkas) dan Baseline (full schema, tanpa pruning) memberi LLM konteks yang sangat berbeda, jadi LLM kemungkinan menghasilkan gaya SQL yang berbeda (struktur JOIN, subquery vs join eksplisit) di antara dua kondisi bahkan untuk pertanyaan yang sama-sama "benar" — risiko false-negative dari isu ini TIDAK NETRAL antar dua kondisi, berpotensi jadi confound saat membandingkan EX_G vs EX_B.

**Mitigasi (2026-09-06, keputusan "Document + affected-query counter"):** confound ini sekarang DIUKUR, bukan dibiarkan tanpa data. `src/utils/sql_execution.py`'s `execute_sql()` + `compare_execution_results()` diimplementasi penuh (multiset comparison via `collections.Counter`, koneksi SQLite read-only) dan dipakai HANYA oleh `_update_confound_diag()` di `pipeline.run_comparison()` — counter read-only yang menghitung, per kondisi (GraphRAG vs Baseline): jumlah query dengan EX resmi = 0 tapi multiset baris hasil match gold, dipecah lagi berdasarkan apakah gold SQL punya `ORDER BY` eksplisit (kalau ya → order memang relevan, BUKAN false negative). Hasil ditulis ke blok "CONFOUND DIAGNOSTICS" di `outputs/tables/comparison_report.txt`. **`ex_result`/`ex` di raw_logs TIDAK berubah** — tetap 100% dari `eval_exec_match()` resmi. Keputusan 2026-08-08 di atas TIDAK dibalik; ini murni instrumentasi threat-to-validity. Lihat poin 20.

**Lokasi implementasi:** `src/metrics/esm_ex_cm.py` (`ex = 1 if eval_exec_match(...) else 0`, dengan komentar panjang menjelaskan keputusan ini persis di baris tersebut). `src/utils/sql_execution.py`'s `compare_execution_results()` — **implemented, tapi TIDAK pernah jadi sumber `ex_result`**; cuma dipakai counter diagnostic di poin 20. Jangan wire ke `esm_ex_cm.py` tanpa instruksi baru yang membalik keputusan ini.

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
2. Open item A di file ini (parser resmi SPIDER tidak bisa parse `LEFT JOIN`/`RIGHT JOIN` — `KeyError: 'left'`) akan menghantam ESM/CM lebih parah khusus untuk QVT: parafrase pertanyaan (mis. "siswa yang tidak punya nilai" vs "siswa dengan nilai") kemungkinan besar justru MEMICU LLM memilih idiom SQL berbeda seperti `LEFT JOIN` — variasi yang secara fungsional benar akan otomatis gagal ESM/CM murni karena parser-nya crash, bukan karena SQL-nya salah.
   **⚠️ Koreksi (2026-08-25, lihat poin 11):** klaim "EX tidak butuh parsing sama sekali, jadi tidak kena isu ini" di draf keputusan ini SALAH untuk cara `eval_exec_match()` dipanggil di `esm_ex_cm.py` — fungsi itu memang mengeksekusi string mentah, TAPI juga memakai `p_sql`/`g_sql` (dict hasil parse) untuk membangun pemetaan kolom hasil eksekusi (`pred['select'][1]`). Jadi EX **juga** ikut rusak oleh parser gap ini sebelum poin 11 di-implementasi (`p_val_units` jadi kosong saat parse gagal → perbandingan hasil eksekusi selalu `False`). Alasan #1 di atas (EX cocok dengan tujuan QVT) tetap berlaku dan keputusan EX-untuk-QVT tidak berubah — tapi setelah poin 11, EX untuk QVT (jalur in-process lewat `evaluate_single_query()`) juga otomatis mendapat fix parsingnya, bukan cuma "kebetulan tidak kena bug".
3. Trade-off yang diterima: EX mewarisi isu order-sensitivity yang sudah didokumentasikan di poin 5 di atas — tapi itu bukan risiko baru, sudah accepted risk untuk EX di seluruh skripsi ini.

**Lokasi implementasi:** `src/metrics/qvt.py` (docstring modul menegaskan kontrak data `"is_correct"` = hasil EX; `compute_qvt_per_query()`/`aggregate_qvt()` sendiri cuma mengonsumsi field itu, tidak menghitung EX/ESM sendiri). **Belum ada** kode yang benar-benar mengisi `data/qvt_variations/*.json` (generator/runner untuk itu belum ditulis) — sesi berikutnya yang membangun generator itu WAJIB memanggil `src.metrics.esm_ex_cm.evaluate_single_query()` dan mengambil field `.ex`, bukan `.esm`, supaya konsisten dengan keputusan ini.

---

## 11. Parser JOIN-keyword gap (open item A): normalisasi teks sebelum parsing, dua tier untuk jalur in-process vs subprocess

**Konflik/ambiguitas:** menyelesaikan open item A (di bawah). `process_sql.py`'s `JOIN_KEYWORDS = ('join', 'on', 'as')` (baris 32) cuma mengenali token `join` polos. Diverifikasi langsung menjalankan `get_sql()`: **`INNER JOIN`, `CROSS JOIN`, `LEFT [OUTER] JOIN`, `RIGHT [OUTER] JOIN`, `FULL [OUTER] JOIN` SEMUA crash** (`KeyError: 'inner'`/`'cross'`/`'left'`/`'right'`/`'full'`) — bukan cuma `LEFT`/`RIGHT` seperti dugaan awal open item A, `INNER JOIN` (join type paling umum di SQL biasa) juga ikut crash.

**Temuan kunci yang mengubah pendekatan:** struktur hasil parse `'from': {'table_units': [...], 'conds': condition}` (`process_sql.py` baris 15) **tidak pernah merekam join type sama sekali**. Jadi ESM/CM tidak kehilangan informasi apapun kalau join-type keyword dinormalisasi jadi `JOIN` polos SEBELUM di-parse — normalisasi teks di titik ini aman untuk *parsing*, beda dengan menormalisasi teks yang akan *dieksekusi* (itu tetap tidak aman untuk `LEFT`/`RIGHT`/`FULL JOIN`, karena join-join itu mempertahankan baris ber-`NULL` yang benar-benar mengubah result set — poin ini konsisten dengan catatan teknis open item A yang lama).

**Koreksi:** klaim lama "EX biasanya ikut 0" di open item A ternyata bukan cuma "biasanya" — **EX SELALU ikut gagal** untuk join type yang crash, karena `eval_exec_match()` (`evaluation.py:614`) memakai `pred['select'][1]` (dict hasil parse) untuk membangun pemetaan kolom hasil eksekusi; kalau parse gagal dan jatuh ke `_empty_sql()` (`"select": [False, []]`), pemetaan itu kosong dan perbandingan hasil eksekusi selalu `False`. Fix di bawah otomatis memulihkan EX juga, bukan cuma ESM/CM.

**Keputusan (2026-08-25):** buat modul baru `src/utils/sql_normalize.py` sebagai **satu-satunya sumber kebenaran** untuk normalisasi ini — `normalize_join_keywords_for_parsing(sql, execution_safe_only)` untuk string langsung, `normalize_sql_file_for_parsing(...)` untuk file (dipakai jalur subprocess). Dua tier, BUKAN satu normalisasi buta untuk semua kasus:
- `execution_safe_only=True` → cuma `INNER`/`CROSS JOIN` dinormalisasi (100% setara hasil eksekusi, aman dipakai bahkan di teks yang benar-benar akan dieksekusi).
- `execution_safe_only=False` (default) → SEMUA join type dinormalisasi. Hanya aman kalau hasilnya CUMA dipakai untuk parsing, TIDAK PERNAH untuk teks yang akan dieksekusi.

Penerapan beda per jalur, karena constraint arsitektur beda:
- **Jalur in-process** (`src/metrics/esm_ex_cm.py`'s `_evaluate_single_query_inner()`) — Python call langsung, jadi raw-string-untuk-eksekusi vs normalized-string-untuk-parsing bisa dipisah bersih sebagai dua argumen berbeda. `get_sql()` di baris ~195/199 menerima salinan ternormalisasi PENUH (`execution_safe_only=False`); `eval_exec_match()` di baris ~251 TETAP menerima `predicted_sql`/`gold_sql` mentah, tidak diubah sama sekali. **Fix penuh, semua join type, termasuk EX.**
- **Jalur subprocess** (`pipeline.py`'s `run_official_evaluation()`, `ablation.py`'s `_run_ablation_evals()`, notebook `notebooks/eval_pipeline.ipynb` Bagian 14) — `evaluation.py` CLI resmi membaca SATU file `--pred`/`--gold` dan memakai string yang SAMA untuk parsing MAUPUN eksekusi (dikonfirmasi `evaluation.py`'s `evaluate()`, baris ~501-547: `p_str = p[0]` di-`get_sql()` DAN dioper mentah ke `eval_exec_match(db, p_str, ...)`) — tidak ada cara memisahkan keduanya tanpa mengubah script resmi. Karena itu:
  - `--etype match` (dikonfirmasi baris ~546: `eval_exec_match` cuma dipanggil kalau `etype in ["all","exec"]`, jadi `match` TIDAK PERNAH mengeksekusi apapun) → file dinormalisasi PENUH, aman.
  - `--etype exec` → file HANYA dinormalisasi `execution_safe_only=True` (INNER/CROSS saja). **`LEFT`/`RIGHT`/`FULL JOIN` TETAP gagal parse di jalur subprocess-exec ini** — residual limitation yang diterima, bukan bug baru. Kalau nanti mau di-fix juga, satu-satunya cara adalah patch `external/spider_eval/process_sql.py` langsung, yang melanggar konvensi "kode resmi apa adanya" — **belum diputuskan, tanyakan ke peneliti dulu** kalau ini jadi prioritas.

**Alasan:** kode resmi SPIDER (`external/spider_eval/`) tetap byte-identical/tidak disentuh (konvensi atribusi di `CLAUDE.md`) — normalisasi selalu terjadi di teks INPUT sebelum masuk ke kode resmi, bukan modifikasi kode resminya. Satu fungsi regex dipakai semua call site (diminta peneliti) supaya tidak ada logic normalisasi yang duplikat/drift antar file.

**Lokasi implementasi:** `src/utils/sql_normalize.py` (baru), `src/metrics/esm_ex_cm.py` (`_evaluate_single_query_inner()`, baris ~195-204), `src/experiments/pipeline.py` (`run_official_evaluation()`), `src/experiments/ablation.py` (`_run_ablation_evals()`), `notebooks/eval_pipeline.ipynb` Bagian 14 (`run_spider_official()`, cell markdown diupdate dengan catatan residual limitation).

---

## 12. Dimensi 2 baris ke-4 ("EX rendah, ESM rendah"): dibaca directional (turun/turun), bukan threshold absolut baru

**Konflik/ambiguitas:** guide Bagian 3 Dimensi 2 mensyaratkan 4 kombinasi klasifikasi ESM×EX (WAJIB, "tabel ini persis"). Tiga baris pertama jelas *directional* — "naik"/"turun" berarti GraphRAG dibanding Baseline (delta). Baris ke-4 ditulis "EX rendah, ESM rendah" — kata "rendah" (bukan "turun"), tanpa angka threshold (beda dari Dimensi 5 yang eksplisit 80%/65%), dan tanpa kejelasan itu nilai absolut milik siapa (GraphRAG? Baseline? keduanya?). Dibaca literal-directional, 3 baris pertama cuma menutup 3 dari 4 kemungkinan arah delta (naik/turun, turun/naik, naik/naik) — kuadran ke-4 (turun/turun) tidak eksplisit ada di tabel manapun kecuali baris ke-4 ini dimaksudkan untuk itu.

**Keputusan (2026-08-26):** baris ke-4 dibaca directional juga — **EX turun DAN ESM turun** (melengkapi kuadran ke-4, pasangan alami baris "EX naik, ESM naik" = "kondisi paling ideal"). TIDAK memperkenalkan threshold absolut baru untuk "rendah".

**Alasan:** membaca semua 4 baris dengan semantik yang sama (directional) menghasilkan klasifikasi 4-kuadran yang lengkap dan internally consistent tanpa perlu mengarang angka threshold baru yang tidak ada di manapun di guide untuk tabel spesifik ini (melanggar aturan anti-halusinasi Bagian 4 poin 1 kalau dipaksakan). Interpretasi threshold-absolut juga berisiko tumpang-tindih dengan baris 1-3 (mis. EX naik tapi nilai absolutnya tetap rendah), yang tidak dijelaskan guide cara resolusinya.

**Catatan implementasi:** label yang ditampilkan ke pengguna tetap teks literal guide, `"EX rendah, ESM rendah"` — bukan diganti jadi `"EX turun, ESM turun"` — supaya tetap traceable ke tabel resmi proposal, walau logika pemicunya (kode) memakai perbandingan delta turun/turun.

**Edge case ditemukan saat implementasi (belum ada di guide):** kalau salah satu delta persis 0 (EX naik tapi ESM sama sekali tidak berubah, dst.), tidak ada satupun dari 4 kombinasi resmi yang cocok. Diputuskan: laporkan sebagai "tidak terklasifikasi" dengan delta mentahnya ditampilkan, BUKAN dipaksakan ke salah satu dari 4 kategori resmi.

**Lokasi implementasi:** `src/dimensions/dim2_structure.py`'s `_classify()`. Juga di kesempatan yang sama, `DIFFICULTY_LEVELS`/grouping-per-difficulty diextract dari `dim1_efficiency.py` (yang sebelumnya sengaja lokal, lihat catatan di file itu) ke `src/utils/raw_logs.py` (`group_by_difficulty()`) karena `dim2_structure.py` butuh pola identik — sesuai rencana "extract saat pemanggil kedua muncul" yang sudah dicatat di `dim1_efficiency.py` sebelumnya.

---

## 13. `union_all` di exception fallback: tetap `None`, bukan `0` (bug fix)

**Konflik/ambiguitas:** `aggregate_cm()`'s docstring dan jalur evaluasi normal (`_evaluate_single_query_inner()`, baris 278) konsisten: `cm_per_clause["union_all"]` SELALU `None` — parser resmi SPIDER tidak bisa membedakan `UNION` dari `UNION ALL` secara struktural (poin 8 di atas), jadi tidak ada satupun jalur kode yang bisa menghasilkan angka 0/1 yang valid untuknya. Tapi outer exception handler di `evaluate_single_query()` (fallback untuk query yang gagal total dievaluasi) sebelumnya menulis `cm_per_clause={clause: 0 for clause in CLAUSES}` — termasuk `union_all: 0`, kontradiksi langsung dengan kontrak "selalu None" itu.

**Ditemukan saat:** investigasi dependency `src/dimensions/dim3_component.py` (2026-08-26) — Dimensi 3 secara spesifik mencari "klausa dengan penurunan skor terbesar", jadi kontaminasi ini bukan cuma soal kerapian data tapi berpotensi langsung menyesatkan kesimpulan Dimensi 3 kalau ada cukup banyak query yang gagal total dievaluasi karena alasan lain (bukan soal `UNION ALL`).

**Keputusan (2026-08-26):** fallback exception sekarang menulis `cm_per_clause={clause: (None if clause == "union_all" else 0) for clause in CLAUSES}` — `union_all` tetap `None` bahkan di jalur kegagalan total, klausa lain tetap `0` (itu tetap valid: query yang gagal total memang gagal juga di klausa-klausa itu).

**Alasan:** `None` dan `0` punya makna yang beda secara fundamental di sini — `0` berarti "sudah dicek, salah", `None` berarti "tidak bisa dicek sama sekali, oleh siapapun, kapanpun". Menulis `0` untuk `union_all` di fallback pura-pura mengukur sesuatu yang sebenarnya tidak pernah benar-benar diukur, dan karena `aggregate_cm()` cuma merata-ratakan nilai non-`None`, tiap query yang gagal total (karena alasan APAPUN, tidak ada hubungannya dengan `UNION ALL`) diam-diam menurunkan skor `union_all` — mengubahnya jadi proxy "berapa banyak query yang crash", bukan indikator kebenaran `UNION ALL`.

**Lokasi implementasi:** `src/metrics/esm_ex_cm.py`'s `evaluate_single_query()`, exception handler (sebelumnya baris ~320-324). Diverifikasi lewat sanity check: `aggregate_cm()` atas campuran hasil normal + hasil fallback tetap melaporkan `union_all: None`, tidak tercemar oleh entri fallback.

---

## 14. Dimensi 5: signature raw_logs + satu kondisi, SLA dua level dengan diagnosis independen

**Konflik/ambiguitas (a) — signature:** stub asli `run_dimension_5(f1_score_sla: float, ex_score: float, precision: float, recall: float)` menerima angka yang SUDAH teragregasi, beda dari pola Dimensi 1-3 (`run_dimension_N(baseline_logs, graphrag_logs)` — agregasi dilakukan DI DALAM fungsi, dari raw_logs mentah).

**Keputusan:** ubah signature jadi `run_dimension_5(logs: List[dict], condition_label: str = "GraphRAG") -> dict` — raw_logs mentah untuk SATU kondisi (bukan sepasang baseline+graphrag seperti Dimensi 1-3), agregasi SLA & EX dilakukan di dalam fungsi.

**Alasan:** guide sendiri menulis Dimensi 5 sebagai diagnostik untuk "kondisi yang dianalisis" (satu kondisi, bukan perbandingan berpasangan seperti Dimensi 1-3) — beda sifat dari dimensi lain, jadi signature-nya secara alami beda juga. Konsisten dengan pola raw_logs-in yang sudah dipakai Dimensi 1-3, mengurangi kerja manual pre-agregasi yang nanti harus dilakukan `run_all_dimensions.py` (masih belum diimplementasi).

**Konflik/ambiguitas (b) — level SLA mana yang dipakai:** guide cuma menulis "F1-Score SLA" tanpa spesifik level (table atau column) — beda dari Dimensi 1.1 sendiri yang eksplisit membedakan dua level ini sebagai granularitas terpisah. Threshold 80% berlaku sama untuk berapa pun level yang dipilih, tapi tingkat keketatan (seberapa sering trigger) bisa sangat beda antara table-level (longgar) vs column-level (ketat).

**Keputusan:** hitung KEDUANYA, jalankan klasifikasi 4-kuadran PENUH (termasuk breakdown precision/recall) secara independen untuk masing-masing level — bukan cuma satu level yang dipakai untuk keputusan akhir. Kalau kedua level menghasilkan diagnosis yang beda, keduanya ditampilkan apa adanya + flag `levels_agree=False`, TIDAK ada aturan resolusi yang dikarang untuk memutuskan mana yang "benar".

**Alasan:** peneliti secara eksplisit meminta dua diagnosis independen (bukan satu level mendominasi), setelah didiskusikan opsi alternatif ("column-level saja yang menentukan, table-level cuma info tambahan"). Guide tidak mendefinisikan cara resolusi kalau dua level tidak sepakat — memaksakan resolusi (mis. "pakai yang lebih ketat") akan jadi aturan baru yang tidak ada dasarnya di guide (melanggar aturan anti-halusinasi Bagian 4 poin 1). Diverifikasi lewat sanity test: skenario di mana GraphRAG dapat tabel yang benar tapi cuma 1 dari 3 kolom yang dibutuhkan menghasilkan table-level F1=100% ("pipeline sehat") vs column-level F1=50% ("jarang terjadi, kemungkinan query sederhana") — dua diagnosis yang sungguh berbeda untuk data yang sama persis, membuktikan skenario ini bukan cuma teoretis.

**Lokasi implementasi:** `src/dimensions/dim5_bottleneck.py` (`_diagnose_level()` dipanggil dua kali dari `run_dimension_5()`, sekali per level, `levels_agree` dihitung dari perbandingan string diagnosis).

---

## 15. Dimensi 6: algoritma greedy step-wise untuk k_final, threshold 2 poin persen per langkah

**Konflik/ambiguitas:** guide Bagian 3 Dimensi 6 minta "baca pola transisi EX vs k" dan eksplisit melarang "cuma ambil EX tertinggi mentah-mentah", tapi tidak memberi angka untuk apa yang dihitung "signifikan" saat EX naik dari satu k ke k berikutnya, dan tidak memberi algoritma presisi untuk menentukan `k_final` dari pola itu — cuma tiga deskripsi kualitatif (diminishing returns, konsisten, context overload).

**Keputusan (2026-08-26):**
1. **Algoritma:** greedy/step-wise, bukan perbandingan ke EX maksimum global. Mulai dari `k` terkecil, terus maju ke `k` berikutnya SELAMA lompatan EX ke situ >= `STEP_THRESHOLD`. Begitu satu langkah gagal signifikan (termasuk kalau EX-nya turun), berhenti di `k` SEBELUM langkah itu — TIDAK lanjut lagi walau ada langkah signifikan lagi setelahnya (diverifikasi lewat sanity test: `{0:50, 1:51, 3:90, 5:91}` tetap berhenti di k=0 meski lompatan 1→3 sangat besar, karena langkah 0→1 sudah gagal signifikan duluan).
2. **`STEP_THRESHOLD = 2.0` poin persen** — dipilih supaya konsisten dengan threshold ΔQVT yang sudah WAJIB di guide Dimensi 4 (±2%), bukan angka baru yang berdiri sendiri tanpa preseden di guide manapun.
3. Pola "EX naik signifikan di semua transisi sampai k terbesar" (belum plateau dalam rentang k yang diuji) diberi label jujur `terus_naik_signifikan` — bukan salah satu dari 3 pola resmi guide, karena guide memang tidak mendeskripsikan skenario ini secara eksplisit.

**Alasan:** algoritma greedy/step-wise ini secara langsung merepresentasikan cara guide menjelaskan pola-polanya ("naik signifikan dari k=0→k=1 LALU stabil" — deskripsi berurutan per-langkah, bukan perbandingan ke titik global), dan secara alami menghasilkan ketiga pola resmi guide sebagai kasus khusus dari SATU aturan yang sama, tanpa perlu tiga pengecekan terpisah yang bisa saling kontradiksi.

**Lokasi implementasi:** `src/dimensions/dim6_ablation.py`'s `run_dimension_6()`. **Catatan:** modul ini murni interpretasi — `ex_per_k` harus diisi dari hasil run nyata `ablation.py` + evaluation resmi, yang belum pernah dijalankan (lihat `RESEARCHER_TODO.md`). Kode ini sendiri sudah diverifikasi lewat fixture sintetis untuk keempat pola (termasuk `single_k` dan guard `ex_per_k` kosong), sama seperti Dimensi 1/2/3/5 sebelumnya.

---

## 16. Validasi terhadap Spider dev set asli (2026-08-26): lulus, satu limitation data upstream ditemukan dan diterima apa adanya

**Konteks:** dataset SPIDER resmi lengkap (1034 dev queries, database SQLite per db_id) ditambahkan ke `data/spider_data/` (di luar git, lihat `.gitignore`). Ini kesempatan pertama untuk memvalidasi `src/metrics/esm_ex_cm.py` dan `src/metrics/sla.py` terhadap data Spider ASLI, bukan cuma sanity test SQLite sintetis (catatan "belum pernah divalidasi terhadap dev set asli" yang sebelumnya ada di beberapa tempat — poin 4, `CLAUDE.md` — SEKARANG SUDAH TIDAK BERLAKU, lihat hasil di bawah).

**Metodologi validasi:** round-trip test — `predicted_sql = gold_sql` untuk seluruh 1034 query dev set, jalankan lewat `evaluate_single_query()` (harus esm=1, ex=1 kalau parser/eval sehat) dan `compute_sla()` self-comparison (harus f1=1.0).

**Hasil:**
- **0 exception, 0 kegagalan ESM** dari 1034 query — parser resmi SPIDER (termasuk fix JOIN-keyword di poin 11) menangani seluruh kompleksitas SQL asli (subquery bersarang, multi-join, GROUP BY/HAVING) tanpa masalah.
- **42 "kegagalan" SLA — BUKAN bug, asumsi test yang salah.** Semuanya query `COUNT(*)`/`SELECT *` tanpa referensi kolom spesifik apapun — `extract_ground_truth_schema()` BENAR mengembalikan set kosong untuk kasus ini, dan `compute_sla()` memang didesain sengaja (keputusan lama, bukan baru) mengembalikan F1=0.0 untuk perbandingan set-kosong-vs-set-kosong (bukan "1.0 by convention"). Test round-trip saya yang salah asumsi, bukan kode yang salah — tidak ada perubahan kode.
- **2 kegagalan EX — ditemukan limitation data asli, BUKAN bug kode kita.** Satu baris di `data/spider_data/database/wta_1/wta_1.sqlite` (`player_id=212305`, kolom `last_name` berisi byte `b'Treyes Albarrac\xe3\x8dN'`, bukan UTF-8 valid — kemungkinan artefak konversi encoding saat dataset resmi dibuat) bikin `sqlite3`'s `fetchall()` throw `UnicodeDecodeError`. `eval_exec_match()` RESMI SPIDER (`external/spider_eval/evaluation.py`) menangkapnya lewat bare `except: return False` di sisi predicted — jadi query APAPUN yang menyentuh baris ini otomatis EX=0, walau SQL-nya benar. Terverifikasi affects persis 2/1034 query (0.19%), keduanya menyentuh tabel `players` di `wta_1` tanpa filter yang mengecualikan baris itu.

**Keputusan:** DIDOKUMENTASIKAN, TIDAK diperbaiki — baik di kode maupun di file database.

**Alasan:** dampaknya sangat kecil (0.19% dev set), dan "perbaikan" apapun (patch `eval_exec_match()` resmi, atau edit langsung byte yang rusak di file `.sqlite`) berarti hasil EX kita tidak lagi 100% comparable dengan paper/sistem lain yang mengevaluasi di dataset resmi SPIDER yang TIDAK dimodifikasi. Prinsip yang sama dengan keputusan EX order-sensitivity di poin 5 — comparability dengan literatur lebih diprioritaskan daripada memperbaiki keterbatasan kecil yang bukan berasal dari kode kita sendiri.

**Lokasi:** tidak ada perubahan kode. Script validasi (`validate_real_spider.py`) dijalankan sekali dari scratchpad session, tidak masuk repo (bukan bagian permanen dari test suite).

---

## 17. Bug fix: `_set_op_clause_match()` mengambil sub-query union/intersect/except SETELAH termutasi eval_exact_match() utama

**Ditemukan:** saat validasi Dimensi 3 terhadap data Spider dev set asli (lihat poin 16) — tabel CM per klausa menunjukkan `intersect`/`except` cuma ~96-97%, bukan 100%, padahal `predicted_sql = gold_sql` persis (round-trip test, harusnya selalu match). Diinvestigasi lebih dalam: dari 40 query dev set asli yang benar-benar pakai `INTERSECT`, **38 (95%) salah skor 0**; dari 31 query yang pakai `EXCEPT`, **29 (94%) salah skor 0** — walau predicted==gold persis.

**Root cause (diverifikasi langsung via reproduksi terisolasi):** `Evaluator.eval_exact_match()` resmi SPIDER memutasi in-place SEMUA list nested di dalam dict yang dioper kepadanya (bukan cuma level teratas seperti `select`/`where` — juga merambat ke sub-dict `union`/`intersect`/`except` yang tersimpan di dalam struktur SQL yang sama). `_set_op_clause_match()` (di `esm_ex_cm.py`) dipanggil SETELAH `eval_exact_match(p_sql, g_sql)` utama, dan sebelumnya mengambil sub-query lewat `pred_sql.get(key)`/`gold_sql.get(key)` — yaitu referensi ke dict yang SAMA yang baru saja termutasi. Perbandingan `eval_exact_match()` rekursif di dalam `_set_op_clause_match()` jadi membandingkan data yang sudah "termakan" sebagian, menghasilkan false-negative sistematis. Dikonfirmasi lewat reproduksi terisolasi: comparing pristine (belum tersentuh) deep copy dari sub-query yang SAMA persis memberi hasil match=True, comparing sub-query yang diambil setelah eval_exact_match() utama memberi hasil match=False -- pembeda satu-satunya adalah timing pengambilan referensi, bukan isi data.

**Keputusan (2026-08-26):** ambil deep copy (`copy.deepcopy`) dari `p_sql`/`g_sql`'s sub-dict `union`/`intersect`/`except` SEBELUM `eval_exact_match()` utama dipanggil (bukan sesudahnya), simpan di variabel lokal (`p_set_ops`/`g_set_ops`), dan oper salinan pristine itu ke `_set_op_clause_match()` — bukan mengambil ulang dari `p_sql`/`g_sql` yang sudah termutasi. `_set_op_clause_match()`'s signature diubah dari `(pred_sql, gold_sql, key, evaluator)` jadi `(pred_sub, gold_sub, evaluator)` supaya caller wajib mengoper sub-query yang sudah diekstrak, bukan dict induk + key (mencegah kesalahan yang sama terulang di masa depan).

**Alasan:** ini murni bug di kode custom thesis (bukan official SPIDER — official evaluator TIDAK punya fungsi untuk klausa union/intersect/except terpisah, `_set_op_clause_match()` seluruhnya ditulis untuk thesis ini, lihat CLAUDE.md "Atribusi kode"), jadi bebas diperbaiki langsung tanpa menyentuh `external/spider_eval/`. Dampaknya nyata: 6.9% dari dev set asli (71/1034 query) memakai UNION/INTERSECT/EXCEPT, dan hampir semuanya salah dilaporkan di CM sebelum fix ini.

**Verifikasi:** setelah fix, seluruh 40 query INTERSECT dan 31 query EXCEPT di dev set asli match sempurna (0 kegagalan, turun dari 38 dan 29). Regression check: skenario LEFT JOIN dan union_all fallback (poin 11, 13) tetap berfungsi normal setelah perubahan ini.

**Lokasi implementasi:** `src/metrics/esm_ex_cm.py` — `_set_op_clause_match()` (signature + docstring), `_evaluate_single_query_inner()` (penambahan `p_set_ops`/`g_set_ops` deep copy sebelum `eval_exact_match()` utama, dan 3 call site union/intersect/except diupdate).

---

## 18. Definisi Baseline: table-level retrieval (FINAL)

**Konflik/ambiguitas:** `CLAUDE.md` + `RESEARCHER_TODO.md` lama mencatat baseline "saat ini table-level retrieval, mungkin diubah ke full-schema bypass — diskusikan dulu". Selama belum diputuskan, tiap run raw_logs/ablation berisiko harus diulang kalau definisi berubah.

**Keputusan (2026-09-06):** Baseline skripsi = **table-level retrieval** (`src/retrieval/baseline.py`: top-k table selection + FK-path expansion + semua kolom tabel terpilih, tanpa pruning). Full-schema bypass (`use_full_schema_bypass` / `--full-schema`) tetap ada sebagai **mode ablation**, BUKAN baseline.

**Alasan:** table-level retrieval adalah kontras yang tepat untuk klaim skripsi (granularitas retrieval: table vs column). Full-schema bypass sebagai baseline utama membuat perbandingan token jadi tidak informatif (bandingan "kirim semua" vs "kirim sedikit" itu trivial).

**Lokasi implementasi:** tidak ada perubahan kode — `src/retrieval/baseline.py` sudah table-level. Dokumen di-update: `CLAUDE.md` ("Hal yang Jangan Diubah Tanpa Diskusi" + TBD list), `context/RESEARCHER_TODO.md` bagian 3 (di-check).

---

## 19. Recall/precision di `retrieval.py`/`baseline.py` = proxy internal, BUKAN SLA

**Konflik/ambiguitas:** `evaluate_schema_linking()` (`src/retrieval/retrieval.py`) dan `evaluate_table_linking()` (`src/retrieval/baseline.py`) menghitung "recall/precision" yang di-print saat run dan dipakai `sweep.py` untuk ranking F6. Tapi keduanya: (1) campur nama table + column jadi SATU set tanpa kualifikasi `table.column` (kolom `name` di dua tabel beda → kolaps jadi satu elemen), (2) ekstrak gold lewat token-matching terhadap known schema names, bukan SQL parser, (3) tidak dipecah table-level vs column-level. Semua itu yang justru diminta guide 1.1 untuk SLA yang benar, dan `src/metrics/sla.py` sudah mengimplementasikannya dengan parser resmi SPIDER.

**Keputusan (2026-09-06, opsi "Document only"):** biarkan fungsi proxy apa adanya (dipakai live di `pipeline.py`/`sweep.py`/`ablation.py` untuk progress + ranking sweep — mengganti perilakunya akan menggeser ranking F6 dan pilihan top-k, jadi TIDAK disentuh sekarang). Tambahkan warning docstring/komentar tegas di kedua fungsi + baris di `CLAUDE.md` bahwa **angka SLA yang dilaporkan di skripsi diambil dari `src/metrics/sla.py`, bukan dari fungsi proxy ini**. Tidak di-rename (ripple ke 4 file).

**Alasan:** proxy cukup sebagai sinyal relatif saat tuning dan konsisten dengan cara `sweep.py` sudah bekerja; risikonya hanya kalau seseorang salah melaporkannya sebagai SLA resmi. Dokumentasi eksplisit menutup risiko itu tanpa mengubah perilaku yang sudah tervalidasi.

**Lokasi implementasi:** komentar blok di `src/retrieval/retrieval.py` (di atas seksi "Schema evaluation"), docstring `evaluate_table_linking()` di `src/retrieval/baseline.py`, baris SLA di tabel "Evaluasi Metrik" `CLAUDE.md`.

---

## 20. Confound diagnostics read-only di `run_comparison()` (tidak mengubah metrik apapun)

**Konteks:** dua confound evaluasi yang sudah didokumentasikan (poin 5 = EX order-sensitivity; poin 11 = normalisasi outer JOIN untuk parsing) butuh data kuantitatif dari run asli, bukan cuma catatan naratif. Peneliti minta "document + affected-query counter", bukan metrik baru / bukan membalik keputusan lama.

**Keputusan (2026-09-06):** `pipeline.run_comparison()` menghitung dan menulis blok **CONFOUND DIAGNOSTICS** ke `outputs/tables/comparison_report.txt`, read-only:
- **EX order-sensitivity (poin 5):** per kondisi, jumlah query `ex_result==0` (order-sensitive resmi) tapi multiset baris hasil eksekusi match gold (order-insensitive), dipecah subset gold punya `ORDER BY`. "Likely false-negative" = tanpa `ORDER BY`. Plus asimetri GraphRAG−Baseline.
- **Outer JOIN (poin 11):** frekuensi `LEFT`/`RIGHT`/`FULL JOIN` di gold + prediksi tiap kondisi; dari prediksi outer-JOIN, berapa yang `ESM=1` tapi `EX=0` (kandidat "normalisasi parsing menutupi beda semantik nyata").

`_update_confound_diag()` punya try/except sendiri (gagal → skip sample itu, tidak ganggu raw_logs). Eksekusi SQL cuma dijalankan untuk sample yang minimal satu arm-nya sudah `EX=0` (biaya near-zero untuk mayoritas).

**Yang TIDAK berubah:** `esm_result`/`ex_result`/`cm_per_clause` di `data/raw_logs/*.json` — semua tetap 100% dari `esm_ex_cm.evaluate_single_query()` (evaluator resmi). Poin 5 tidak dibalik. `compare_execution_results()` diimplementasi tapi TIDAK pernah jadi sumber `ex_result`.

**Lokasi implementasi:** `src/experiments/pipeline.py` (`_new_confound_diag()`, `_update_confound_diag()`, `_format_confound_diag()`, dipanggil di `run_comparison()`), `src/utils/sql_execution.py` (`execute_sql()`, `compare_execution_results()` — diimplementasi penuh, dipakai HANYA di sini).

---

## 21. QVT variations: paraphrase ALAMI dari dev.json, bukan digenerate/disiapkan manual (mengganti keputusan 2026-08-15)

**Konteks:** keputusan 2026-08-15 (dicatat di `RESEARCHER_TODO.md`) bilang grouping variasi NL question untuk QVT (Dimensi 4) dikerjakan manual oleh peneliti di luar coding — waktu itu diasumsikan SPIDER dev set biasa tidak punya variasi NL question untuk SQL yang sama (asumsi ini juga tertulis literal di docstring lama `src/metrics/qvt.py`: "Ini TIDAK datang otomatis dari SPIDER dev set biasa").

**Ditemukan (2026-09-07), setelah dataset SPIDER asli tersedia di `data/spider_data/`:** asumsi itu SALAH. Proses anotasi SPIDER (penulis pertanyaan berbeda + langkah verifikasi paraphrase) menghasilkan banyak SQL yang punya >1 pertanyaan NL berbeda secara alami. Dicek langsung ke `dev.json` (1034 baris): mengelompokkan berdasarkan `(db_id, SQL text)` identik menghasilkan **470 grup dengan tepat 2 pertanyaan NL berbeda per grup** (tidak ada grup dengan 3+), mencakup **940/1034 baris (90.9%)** dev set. 94 baris sisanya SQL-nya unik (cuma 1 pertanyaan), di luar populasi QVT.

Contoh (`concert_singer`, SQL `SELECT count(*) FROM singer`):
- "How many singers do we have?"
- "What is the total number of singers?"

**Keputusan:** pakai paraphrase alami ini sebagai `N_i1, N_i2` untuk QVT (guide 1.6), BUKAN generate parafrase baru (LLM atau manual). Alasan:
1. Ini pertanyaan yang benar-benar ditulis manusia dengan intent yang sama — bukan sintetis (LLM-generated paraphrase punya risiko drift makna, dan menambah beban kerja peneliti tanpa perlu).
2. Cakupan 90.9% dari dev set sudah lebih dari cukup untuk populasi QVT yang representatif.
3. `predicted_sql`/`is_correct` (EX, sesuai poin 10) untuk tiap variasi bisa langsung diambil dari `data/raw_logs/{baseline,graphrag}_log.json` yang SUDAH diproduksi `run_comparison()` untuk keperluan Dimensi 1/2/3/5 — **tidak perlu jalankan pipeline terpisah/tambahan** untuk QVT sama sekali, tinggal join by `query_id`.

**Batasan yang harus disebut di metodologi:**
- `m_i` (jumlah variasi per query) selalu tepat 2, tidak pernah lebih — lebih kasar dari contoh ilustratif guide (m=5). `skor_i` cuma bisa 0, 0.5, atau 1.
- Populasi QVT cuma 940/1034 (90.9%) dev set — 94 query dengan SQL unik tidak pernah dievaluasi variance-nya.
- **Syarat teknis join:** `query_id` di raw_logs ("dev_XXXX") adalah posisi baris di `dev_data` SAAT `run_comparison()` jalan (lihat `pipeline.py` sekitar baris 648-652 & 782) — hanya identik dengan posisi asli di `dev.json` kalau `sample_ratio=1.0` (TANPA subsampling). Builder script mem-verifikasi `len(raw_logs) == len(dev.json)` sebelum join dan berhenti dengan pesan error kalau tidak sama, supaya tidak diam-diam menghasilkan pairing yang salah.

**Lokasi implementasi:** `src/experiments/build_qvt_variations.py` (script baru — `build_groups()` mengelompokkan `dev.json`, `join_with_raw_logs()` menjoin ke raw_logs per kondisi, tulis `data/qvt_variations/{baseline,graphrag}_qvt.json`). Diverifikasi dengan fixture raw_logs sintetis (full 1034 baris, EX acak) di scratchpad — output 470 query x 2 variasi per kondisi sesuai kontrak `src/metrics/qvt.py`, dan dua guard rail (raw_logs belum ada; raw_logs ke-subsample) diuji menghasilkan pesan error yang jelas, bukan crash/silent-wrong-output. Docstring `src/metrics/qvt.py` yang lama (klaim "tidak datang otomatis") sudah diperbaiki.

**Dampak ke `RESEARCHER_TODO.md`:** sub-tugas "grouping" untuk `data/qvt_variations/` sudah selesai secara kode (tinggal dijalankan). Yang masih jadi tanggung jawab peneliti/GPU: menjalankan `pipeline.py --baseline --sample 1.0` di Kaggle (blocker yang sama dengan Dimensi 1/2/3/5), lalu jalankan `build_qvt_variations.py`.

---

## 22. Penghapusan `few_shot_same_db_first` — provably dead code, bukan cuma inert

**Konteks:** `retrieve_few_shot_examples()` (`src/generation/few_shot.py`) punya mode `few_shot_same_db_first=True` yang, kalau aktif: (1) over-fetch pool `few_shot_k * 4` kandidat by cosine similarity dulu, (2) mempromosikan kandidat yang `db_id`-nya sama dengan pertanyaan saat ini ke depan top-k, walau similarity-nya lebih rendah dari kandidat cross-DB yang tergeser. Docstring lama beralasan "Spider train/dev splits share database IDs, so same-DB examples are often the most relevant" — juga ditulis ulang di CLAUDE.md sebagai catatan metodologi yang harus disebut soal bias.

**Ditemukan (2026-09-12), saat menjelaskan alur few-shot retrieval ke peneliti:** asumsi "train/dev share database IDs" itu SALAH. Diverifikasi langsung: `dev.json` (20 database) dan `train_spider.json` (140 database) **disjoint total** — 0 `db_id` overlap. Ini bukan kebetulan di split kita, tapi properti struktural Spider sebagai *cross-domain* benchmark (definisi intinya: model harus generalize ke database yang belum pernah dilihat). Konsekuensinya, cabang `same_db` di kode **provably selalu kosong** untuk evaluasi Spider dev set manapun — bukan "kebetulan nol di run ini", tapi tidak mungkin pernah berisi apapun selama pipeline dievaluasi terhadap dev set standar dengan few-shot dari train set standar.

**Keputusan (2026-09-12):** hapus `few_shot_same_db_first` total (parameter config + logic rerank + over-fetch pool), bukan cuma didokumentasikan sebagai "inert". Alasan:
1. Kode yang tidak mungkin pernah menyala itu dead weight — nambah kompleksitas (pool `k*4`, extra `torch.topk` di kandidat lebih besar) tanpa efek terukur apapun di hasil.
2. Membiarkannya "didokumentasikan sebagai inert" berisiko sesi/peneliti berikutnya lupa konteks ini dan menganggapnya aktif — root cause (Spider cross-domain, disjoint by design) sudah permanen, bukan sesuatu yang bisa berubah nanti kalau dataset di-refresh.
3. Trade-off yang coba dijawab fitur ini (promosikan same-DB walau similarity lebih rendah, demi vocab schema yang identik) TIDAK PERNAH tervalidasi lewat ablation — bukan keputusan berbasis bukti, cuma asumsi desain yang gagal diverifikasi terhadap data asli.

`retrieve_few_shot_examples()` sekarang murni top-k by cosine similarity, tanpa parameter `db_id` (parameter itu jadi tidak terpakai setelah logic same-DB dihapus, jadi ikut dibuang dari signature). Ranking few-shot examples sekarang: 100% cosine similarity pertanyaan, titik.

**Lokasi implementasi:** `src/generation/few_shot.py` (`retrieve_few_shot_examples()` disederhanakan, docstring modul diperbaiki), `src/core/config.py` (field `few_shot_same_db_first` dihapus dari `PipelineConfig`), `src/experiments/pipeline.py` + `src/experiments/ablation.py` (call site disesuaikan, `db_id` tidak lagi dioper ke `retrieve_few_shot_examples()`), `CLAUDE.md` (baris config table, "Few-shot: sumber data & bias", dan "Hal yang Jangan Diubah Tanpa Diskusi" diperbaiki — klaim "train/dev berbagi db_id" dikoreksi jadi "disjoint total").

**Update (2026-09-12, sama hari):** `notebooks/compiled/graphrag_text2sql.ipynb` (versi Kaggle-ready) DISINKRONKAN penuh terhadap `src/` saat ini, bukan cuma untuk `few_shot_same_db_first`. Notebook itu ternyata drift signifikan dari `src/` di banyak tempat (kemungkinan besar belum di-update sejak beberapa refactor terakhir). Yang diperbaiki di 9 cell (config, generation, few_shot, sweep, baseline, pipeline helpers, + 3 cell run):
- `predictions_file` default `"predictions.txt"` → `"outputs/predictions/predictions.txt"`; `bnb_compute_dtype` `torch.bfloat16` → `torch.float16`; `max_new_tokens` `200` → `256`; field `token_output_weight` (hilang total) ditambahkan.
- `_clean_sql()`: guard pemotong teks-setelah-SQL yang hilang ditambahkan (`_cut` regex); fix alias-stripping yang tidak membuang `table.column` prefix dari klausa WHERE/GROUP/ORDER/HAVING/LIMIT yang dipertahankan.
- `generate_sql_with_token_count()` (T_in/T_out) ditambahkan — notebook lama cuma punya `generate_sql()` tanpa hitung token.
- `sweep.py`: metrik heuristik lama (`recall_weighted_f1` ad-hoc) diganti F6 (β=6, arxiv 2501.17174) yang sudah dipakai `src/experiments/sweep.py`; export CSV yang hilang ditambahkan.
- `baseline.py`: `evaluate_table_linking()` diganti ke versi schema-aware (stopword-filtered, sama seperti `evaluate_schema_linking()`) dari versi lama raw-token-overlap; fix FK annotation (`from_col in this_cols`, bukan cuma truthy check) supaya tidak salah arah; fix bug `top_k_tables=0` yang dulu diam-diam fallback ke `top_k=3` alih-alih "semua tabel" (`_all_table_nodes()` ditambahkan).
- `pipeline.py`: `run_official_evaluation()` ditambahkan JOIN-keyword normalization (`normalize_join_keywords_for_parsing`/`normalize_sql_file_for_parsing`, di-port dari `src/utils/sql_normalize.py`) supaya prediksi `INNER`/`CROSS`/`LEFT`/`RIGHT`/`FULL JOIN` tidak bikin parser resmi SPIDER crash saat evaluasi Kaggle (poin 11) -- notebook lama tidak punya normalisasi ini sama sekali.
- **Konsekuensi yang perlu disadari:** `prune_path_nodes()` DIPANGGIL di versi notebook lama (`run_single`/sweep) tapi TIDAK dipanggil di `src/experiments/pipeline.py`/`sweep.py` saat ini (lihat open item terkait di bawah; DIPUTUSKAN di poin 25: fungsinya dihapus). Sync ini bikin notebook ikut TIDAK memanggilnya lagi (match `src/` current) -- artinya retrieval jadi kurang selektif dibanding sebelumnya (bridge node dari path-trace tidak lagi dibuang). Ini regresi nyata, bukan cuma cosmetic, kalau dibandingkan ke perilaku notebook SEBELUM sync ini. Perlu diputuskan terpisah: wire `prune_path_nodes()` balik ke `src/` + notebook, atau terima kondisi ini.
- **TIDAK di-port:** sistem `data/raw_logs/*.json` + evaluasi ESM/EX/CM inline (`src/metrics/esm_ex_cm.py`) + confound diagnostics. Sesuai `requirements.txt` ("Pipeline utama... jalan di Kaggle... metrics/dimensions... dijalankan secara lokal"), notebook ini scope-nya tetap retrieval+generation+official-eval saja. **Catatan:** ini menyisakan pertanyaan terbuka -- `pipeline.py --baseline`'s `run_comparison()` (yang memproduksi `data/raw_logs/*.json`) SECARA NYATA memanggil `src.metrics.esm_ex_cm` inline di environment yang sama dengan LLM inference (jadi harus jalan di Kaggle juga), bertentangan dengan asumsi requirements.txt bahwa metrics selalu lokal. Belum diselesaikan di sini -- lihat open items.

Diverifikasi: seluruh 27 cell notebook (termasuk yang tidak diubah) tetap valid Python (`ast.parse`) kecuali 2 cell magic command (`%pip install`/`!wget`, sudah begitu dari awal), JSON notebook valid, tidak ada sisa referensi `few_shot_same_db_first`.

---

## 23. Dimensi 4: silang-ESM butuh raw_logs terpisah (data QVT sengaja tidak membawa ESM), populasi silang = yang lolos filter wajib

**Konflik/ambiguitas:** guide Bagian 3 Dimensi 4 langkah 6 minta "silangkan dengan ESM (apakah query yang tidak konsisten juga bermasalah di ESM)". Tapi:

1. Data QVT (`data/qvt_variations/*.json`) **tidak punya ESM sama sekali** — field `is_correct` di situ adalah **EX**, bukan ESM (keputusan poin 10, sengaja, supaya definisi "benar" konsisten di seluruh perhitungan QVT). Jadi langkah 6 tidak bisa dijawab dari data QVT saja.
2. Guide tidak mendefinisikan apa arti "tidak konsisten" dan "bermasalah di ESM" secara operasional, dan tidak bilang apakah query yang dibuang filter wajib (semua variasinya salah, guide 1.6(e)) ikut masuk silang atau tidak.
3. `build_qvt_variations.py` versi awal (poin 21) cuma menyimpan **canonical** `query_id` per grup (yang terkecil), membuang `query_id` anggota lainnya — padahal ESM perlu dilihat untuk SETIAP variasi, bukan cuma satu.

**Keputusan (2026-09-14):**

**(a) Signature `run_dimension_4()` diperluas dengan raw_logs opsional:**
```python
run_dimension_4(baseline_qvt_data, graphrag_qvt_data,
                baseline_logs=None, graphrag_logs=None)
```
Dua argumen terakhir HANYA untuk silang-ESM langkah 6. Kalau tidak dioper, `esm_cross` = `None` dan `print_dimension_4()` mencetak blok `[!] DILEWATI` yang menjelaskan kenapa. Langkah guide dilewati secara **terlihat**, bukan diam-diam hilang.

**(b) `build_qvt_variations.py` sekarang menyimpan `query_id` per VARIASI** (tambahan, bukan menggantikan `query_id` canonical di level grup). Field ini tidak dipakai `src/metrics/qvt.py` sama sekali — murni supaya Dimensi 4 bisa melihat `esm_result` tiap variasi di raw_logs. `dim4_robustness.py` melempar `ValueError` dengan pesan eksplisit kalau menemui file qvt_variations versi lama tanpa field ini, alih-alih diam-diam cuma melihat 1 dari m_i variasi.

**(c) Populasi silang-ESM = query yang LOLOS filter wajib QVT (`included=True`) saja.** Query yang semua variasinya salah itu **"konsisten salah"**, bukan "tidak konsisten" — dan memang sudah dibuang dari QVT by definition. Jumlah yang dibuang tetap dilaporkan terpisah (`n_excluded`) supaya tidak tersembunyi.

**(d) Definisi operasional dua sumbu, sengaja dibuat simetris:**
- "tidak konsisten" := `qvt_score < 1.0` (tidak semua variasi benar secara EX)
- "bermasalah di ESM" := tidak semua variasi grup itu punya `esm_result == 1`

**(e) Skala threshold:** `aggregate_qvt()` mengembalikan 0–1, jadi "±2%" dari guide dipakai sebagai `0.02` pada skala itu (2 poin persen), bukan `2.0`. Batas persis mengikuti tabel guide: `ΔQVT ≥ +2%` meningkat, `−2% ≤ ΔQVT < +2%` stabil, `ΔQVT < −2%` menurun — tidak ada celah/overlap.

**Alasan:**

(a) Alternatifnya adalah memaksa data QVT membawa ESM juga. Itu ditolak karena bertentangan dengan poin 10 (satu definisi "benar" untuk seluruh QVT, jangan dicampur) dan akan bikin file qvt_variations punya dua field korektnya yang gampang tertukar. Membaca ESM dari raw_logs — sumber aslinya — lebih jujur dan tidak menduplikasi data. Dibuat opsional supaya Dimensi 4 tetap bisa dijalankan standalone atas file qvt_variations saja (mis. saat cek cepat ΔQVT) tanpa memuat raw_logs penuh.

(c) Memasukkan query yang semua variasinya gagal ke kolom "tidak konsisten" akan salah secara konseptual — variance-nya justru NOL (konsisten gagal) — dan akan mengukur hal yang berbeda dari yang ditanya guide.

(d) Kalau dua sumbu pakai ambang yang beda bentuk (mis. QVT pakai "tidak semua benar" tapi ESM pakai "rata-rata ESM < X%"), tabel 2×2-nya jadi tidak apple-to-apple. Bentuk all-or-nothing yang sama di dua sumbu bikin silangnya bisa dibaca langsung.

**Catatan keterbatasan (dari poin 21, bukan baru):** dengan paraphrase alami dev.json, `m_i` **selalu tepat 2**, jadi `qvt_score` per query hanya bisa bernilai `1.0` (2/2 benar) atau `0.5` (1/2 benar) — query 0/2 dibuang filter. Artinya sumbu "konsisten vs tidak konsisten" di praktiknya adalah biner 2/2 vs 1/2. Kode tidak meng-hardcode `m_i = 2`, tapi interpretasi hasilnya harus sadar batasan ini. Sebutkan di bab metodologi.

**Verifikasi:** 7 test fixture sintetis — QVT hand-computed (termasuk filter wajib: 4 grup → M=3, 1 dibuang), batas threshold ±2% di 7 titik (termasuk persis `+0.02`, `-0.02`, dan `0`), kontingensi 2×2 hand-computed untuk kedua kondisi (memastikan grup yang difilter TIDAK ikut masuk), `esm_cross is None` saat logs tidak dioper, dan 4 error guard (qvt_data kosong, semua query difilter → re-raise dengan label kondisi, format qvt_variations versi lama, variasi hilang di raw_logs → dihitung `n_missing_in_logs`, tidak crash). Plus satu test end-to-end atas `data/spider_data/dev.json` ASLI (1034 baris → 470 grup, semuanya `m_i=2`) lewat `build_groups()` + `join_with_raw_logs()` sungguhan dengan raw_logs sintetis — memastikan kontrak builder ↔ dim4 nyambung.

**Lokasi implementasi:** `src/dimensions/dim4_robustness.py` (implementasi penuh, `run_dimension_4()` + `print_dimension_4()`, menggantikan stub `raise NotImplementedError`), `src/experiments/build_qvt_variations.py` (`join_with_raw_logs()` menambahkan `query_id` per variasi), `src/metrics/qvt.py` (blok format data di docstring diperbarui + catatan bahwa `query_id` dalam variasi tidak dipakai modul itu), `CLAUDE.md` (status Dimensi 4), `context/DIMENSIONS_EXPLAINED.md` (bagian Dimension 4).

---

## 24. Orkestrasi 6 dimensi (`run_all_dimensions.py`): skip-dengan-alasan, gating Dimensi 5 dari Dimensi 2, `ex_per_k` manual untuk Dimensi 6

**Konflik/ambiguitas:** `run_all_dimensions.py` sebelumnya skeleton (`main()` raise `NotImplementedError`). Saat diisi, muncul empat hal yang guide tidak jawab:

1. Setiap dimensi punya prasyarat data BERBEDA (raw_logs, qvt_variations, ex_per_k) dan semuanya belum ada sampai run Kaggle selesai. Apakah file ini crash kalau salah satu belum siap?
2. Guide bikin Dimensi 5 **kondisional** ("hanya dijalankan jika pipeline belum optimal", trigger dari Dimensi 2 kondisi "EX rendah, ESM rendah"), tapi poin 14 sudah memutuskan `run_dimension_5()` sendiri TIDAK mengecek itu — gating-nya tugas pemanggil. Sekarang pemanggilnya ada, jadi gating harus diletakkan di sini.
3. Dimensi 6 butuh `ex_per_k` (EX per nilai k, skala 0–100). Tidak ada file manapun di project yang menyimpannya: `outputs/tables/ablation_results.csv` cuma punya recall/precision/token (lihat `_save_csv()` di `ablation.py`), dan EX per k dihitung oleh Spider `evaluation.py` yang dijalankan `_run_ablation_evals()` sebagai **subprocess** — outputnya ke stdout/log, tidak pernah machine-readable.
4. Dimensi 1 satu-satunya yang butuh `PipelineConfig` (untuk μ), dan `src/core/config.py` `import torch` — yang tidak ada di environment evaluasi lokal.

**Keputusan (2026-09-17):**

**(a) Skip-dengan-alasan, bukan crash.** Tiap dimensi dijalankan dalam wrapper yang mengembalikan `status` ∈ `{"ok", "skipped", "error"}`. `"skipped"` = prasyarat data belum ada (kondisi NORMAL di tahap ini) dengan `reason` yang menyebut file yang kurang + perintah persis untuk memproduksinya. `"error"` = exception tak terduga (bug). Satu dimensi gagal TIDAK membatalkan lima lainnya. **Exit code 1 hanya untuk `"error"`** — "data belum ada" exit 0, supaya file ini aman dijalankan kapan saja cuma untuk melihat status kesiapan.

**(b) Gating Dimensi 5 diletakkan di sini, sesuai guide.** Dimensi 5 dijalankan HANYA kalau `run_dimension_2()` mengembalikan `escalate_to_dimension_5=True` (yaitu kuadran "EX rendah, ESM rendah" per poin 12). Kalau tidak, di-skip dengan alasan yang menyebut kombinasi Dimensi 2 yang terbaca. Flag `--always-dim5` menimpa gating itu. Dimensi 5 dijalankan untuk **dua kondisi** (Baseline dan GraphRAG) secara terpisah — bukan digabung jadi satu diagnosis — karena dimensi itu memang diagnostik satu-kondisi (poin 14).

**(c) `ex_per_k` dioper manual** lewat `--ex-per-k 0=45.2,1=52.1,3=54.0,5=53.8` atau `--ex-per-k-file` (JSON `{"0": 45.2, ...}`). Kalau tidak diberikan, Dimensi 6 di-skip dengan alasan yang menjelaskan kenapa angkanya tidak bisa diambil otomatis. **TIDAK** dibuat parser stdout Spider eval, dan `ablation.py` **TIDAK** diubah untuk menulis kolom EX — dua-duanya perubahan ke jalur eksperimen yang sudah jalan, di luar scope pengisian orkestrator ini (lihat open item D).

**(d) Dimensi 1 di-import LAZY**, terpisah dari lima import dimensi lain di top-level. Kalau `torch` tidak ada, Dimensi 1 di-skip dengan `reason` yang menyebut sebab persisnya (`config.py` baris 5, untuk satu default `bnb_compute_dtype`) + dua remedy (`pip install torch` versi CPU, atau jalankan di Kaggle). **Sengaja TIDAK** dibuat `PipelineConfig` tiruan/shim lokal yang meng-hardcode μ=3 — itu akan bikin sumber μ bercabang dua dan bisa drift dari `config.py` tanpa ada yang sadar (justru persis masalah yang poin 1 selesaikan). Lihat open item C.

**(e) Validasi skema raw_logs saat load.** `load_raw_logs()` memeriksa 12 field wajib guide Bagian 2 per baris dan melempar `ValueError` yang menyebut **path + index baris + query_id + daftar field yang hilang**. Alasan: tanpa ini, raw_logs yang malformed baru meledak jadi `KeyError` mentah di tengah dimensi ke-2 atau ke-3, tanpa petunjuk baris mana yang salah.

**Alasan (a):** file ini adalah satu-satunya tempat peneliti bisa bertanya "apa yang sudah bisa dijalankan sekarang?". Kalau crash di prasyarat pertama yang kurang, ia tidak pernah bisa menjawab pertanyaan itu — dan output "DILEWATI + alasan + perintah remedy" per dimensi justru berguna sebagai checklist kesiapan, bukan cuma penanganan error.

**Output:** `outputs/tables/dimensions_results.json` (semua return value dimensi + status, machine-readable untuk bikin tabel/plot skripsi) dan `outputs/tables/dimensions_report.txt` (teks laporan persis seperti yang dicetak ke stdout). `--no-save` untuk print saja.

**Verifikasi:** 9 test fixture sintetis end-to-end lewat CLI sungguhan (subprocess) — (1) skenario GraphRAG lebih baik: dim 2/3/4/6 OK dan dim5 di-skip dengan alasan "Dimensi 2 tidak escalate"; (2) skenario GraphRAG lebih buruk: dim2 escalate → dim5 JALAN untuk kedua kondisi; (3) `--always-dim5` menimpa gating pada fixture (1) yang sama; (4) `qvt_variations` hilang → cuma dim4 di-skip, dim2/dim3 tetap OK; (5) raw_logs malformed (2 field dihapus di baris ke-7) → pesan menyebut `baris ke-7` + `ex_result` + `cm_per_clause`, exit 0; (6) `--only 2 3` menghasilkan tepat 2 key; (7) `--ex-per-k-file` + validasi parsing `--ex-per-k`; (8) `--only 9` → error argparse bersih (bukan traceback) + file output tetap tertulis; (9) laporan penuh dicetak dan diperiksa manual. Dimensi 1 ter-skip di semua test karena `torch` memang tidak ada di environment lokal — itu jalur (d) yang bekerja sesuai rancangan, bukan kegagalan test.

**Lokasi implementasi:** `src/experiments/run_all_dimensions.py` (seluruh file ditulis ulang dari skeleton), `CLAUDE.md` (status orkestrasi + CLI quick reference).

---

## 25. Penghapusan `prune_path_nodes()` — dead code, dihapus (2026-09-19)

**Keputusan peneliti:** path pruning tidak dipakai lagi, fungsinya dihapus dari `src/retrieval/retrieval.py`.

**Latar belakang:** `prune_path_nodes()` (buang node perantara path-trace yang bukan direct hit/PK/FK) sudah tidak dipanggil oleh `run_single()`, `sweep.py`, `ablation.py` maupun notebook sejak `log.md` mencatat pemanggilnya dicabut. Sejak itu fungsinya cuma tersisa sebagai definisi tanpa pemanggil, sementara dokumentasi (`CLAUDE.md`, komentar `sweep.py`) masih menyebut "path pruning" seolah bagian pipeline — menyesatkan untuk bab metodologi.

**Yang berubah:** fungsi dihapus (grep: tidak ada pemanggil tersisa di `src/`); `CLAUDE.md` (deskripsi GraphRAG, diagram alur GraphRAG jadi 7 tahap, struktur file) dan komentar `sweep.py` disesuaikan. **Perilaku pipeline TIDAK berubah** karena fungsi itu memang tidak dijalankan.

**Yang perlu diingat di metodologi:** GraphRAG = two-stage retrieval + graph traversal (shortest path antar kolom terpilih; node jalur masuk konteks apa adanya). Selektivitas berasal dari `top_k_tables`/`top_k_columns`, bukan dari pruning. Node perantara dari traversal ikut dikirim ke LLM, jadi `avg_precision` proxy bisa lebih rendah dari desain awal.

**Notebook:** `notebooks/eval_pipeline.ipynb` (dulu di root, dipindah ke `notebooks/` 2026-09-26 dan versi lama di `notebooks/` ditimpa) ditulis ulang 2026-09-19 mengikuti `pipeline.py` saat ini — import `src.*`, unpack `run_single`/`run_single_baseline` 7/6 nilai, evaluasi in-process + `raw_logs`, confound diagnostics, normalisasi JOIN untuk cross-check Spider, walk-through per tahap. Ini menggantikan catatan "TIDAK di-port" di poin sebelumnya untuk notebook root tersebut.

## 26. Integrasi pipeline -> metrics -> 6 dimensi (`run_dimension_analysis()`), 2026-09-19

**Keputusan:** lapisan analisis dipasang ke akhir `run_comparison()` (`pipeline.py --baseline`) dan ke `eval_pipeline.ipynb` lewat SATU fungsi, `pipeline.run_dimension_analysis()`. Fungsi ini hanya mengorkestrasi (`build_qvt_files()` + `run_all()` + `save_dimension_outputs()`); logikanya tetap di `build_qvt_variations.py` dan `run_all_dimensions.py`, jadi CLI, pipeline, dan notebook tidak punya jalur berbeda yang bisa drift.

**Aturan yang dipasang:**
- `full_dev` (= `sample_ratio >= 1.0`) menentukan Dimensi 4: hanya pada run penuh QVT dibangun ulang dan Dimensi 4 dijalankan. Pada subsample Dimensi 4 dikeluarkan dari daftar (bukan sekadar di-skip), supaya `data/qvt_variations/` sisa run penuh sebelumnya tidak dipasangkan diam-diam dengan raw_logs yang berbeda.
- Analisis dibungkus `try/except` non-fatal di `run_comparison()`: jam-jam generation sudah tersimpan di raw_logs; kegagalan analisis tidak boleh terlihat seperti run gagal (jalankan ulang `run_all_dimensions.py`).
- Flag CLI baru: `--no-dimensions`, `--ex-per-k`, `--always-dim5`. `EX_PER_K` untuk Dimensi 6 tetap manual (open item D belum berubah).
- Refactor kecil pendukung: `build_qvt_variations.build_qvt_files()` (dipisah dari `main()`, perilaku CLI sama) dan `run_all_dimensions.save_dimension_outputs()`.

**Yang perlu diingat:** `run_comparison()` sejak awal menulis raw_logs ke `data/raw_logs/` tanpa memedulikan `sample_ratio` (menimpa run penuh sebelumnya bila `--sample < 1.0`); ini perilaku lama yang tidak diubah di sini. Notebook memakai `outputs/trial/` untuk run trial.

**Verifikasi (lokal, tanpa torch):** `run_dimension_analysis()` diuji dengan fixture 1034 query (run penuh: QVT terbentuk dan Dimensi 2/3/4/5/6 `ok`; subsample: Dimensi 4 tidak muncul walau ada QVT lama; run penuh dengan raw_logs subsample: Dimensi 4 `skipped`, tanpa crash). Dimensi 1 ter-skip lokal karena `torch` (open item C), akan jalan di Kaggle. Pipeline penuh belum pernah dijalankan end-to-end.

## Belum diputuskan / open items

### A. `LEFT JOIN` / `RIGHT JOIN` / `INNER JOIN` tidak didukung parser resmi SPIDER

✅ **Diselesaikan (2026-08-25) — lihat poin 11 di atas.** Ditemukan bahwa `INNER`/`CROSS JOIN` juga crash (bukan cuma `LEFT`/`RIGHT`), dan EX (bukan cuma ESM/CM) juga ikut rusak. Semua dinormalisasi ke `JOIN` polos HANYA untuk salinan teks yang di-parse — jalur in-process (`esm_ex_cm.py`) dapat fix penuh untuk semua join type; jalur subprocess (`evaluation.py` CLI) dapat fix penuh untuk `--etype match` tapi cuma `INNER`/`CROSS` untuk `--etype exec` (residual limitation, lihat poin 11 untuk alasan arsitekturnya). Teks asli di bawah dipertahankan sebagai catatan historis.

**Temuan (2026-08-08):** `process_sql.py`'s `parse_from()` cuma mengenali token `join` (bare), tidak ada branch untuk `left`/`right`/`inner`. Predicted SQL yang pakai keyword-keyword itu GAGAL di-parse (`KeyError: 'left'` dkk, sudah diverifikasi langsung), jatuh ke fallback `_empty_sql()` yang sama dengan predicted SQL yang benar-benar rusak → ESM=0 otomatis, EX biasanya ikut 0 (karena `p_val_units` jadi kosong).

**Dampak potensial:** karena Qwen2.5-Coder (atau LLM manapun) sangat mungkin menghasilkan `LEFT JOIN` secara alami (idiom SQL yang sangat umum, kadang justru pilihan yang lebih tepat secara semantik untuk pertanyaan "which X have no Y"), ini berpotensi jadi sumber false-negative yang LEBIH besar dan lebih sistematis daripada isu order-sensitivity di poin 5.

**Catatan teknis:** `INNER JOIN` → `JOIN` aman dinormalisasi sebelum parsing (100% setara secara semantik di SQL standar). `LEFT JOIN`/`RIGHT JOIN` TIDAK aman dinormalisasi ke `JOIN` biasa — keduanya mempertahankan baris yang tidak match dengan `NULL`, beda perilaku dari inner join, jadi normalisasi buta akan mengubah semantik yang diukur, bukan cuma perbaikan kompatibilitas parser.

**Status (historis, sebelum poin 11):** belum diputuskan mau diapakan (dibiarkan sebagai known limitation vs investigasi seberapa sering LLM benar-benar memakai keyword ini di praktik).

### B. `requirements.txt` vs `pipeline.py --baseline`: kontradiksi soal "metrics jalan di mana"

**Ditemukan (2026-09-12),** saat sync `notebooks/compiled/graphrag_text2sql.ipynb` (poin 22). `requirements.txt` menyatakan eksplisit: *"Requirements untuk evaluation pipeline (src/metrics/, src/dimensions/) — dijalankan secara lokal, bukan di Kaggle. Pipeline utama (src/core, src/retrieval, src/generation, src/experiments) jalan di Kaggle Notebook..."* — implikasinya `src/metrics/` tidak pernah perlu importable di lingkungan Kaggle.

**Tapi** `src/experiments/pipeline.py`'s `run_comparison()` (dipanggil via `python src/experiments/pipeline.py --baseline`, yang memproduksi `data/raw_logs/{baseline,graphrag}_log.json` — satu-satunya sumber data buat Dimensi 1/2/3/5 dan `build_qvt_variations.py`) memanggil `src.metrics.esm_ex_cm` (`evaluate_single_query`, `build_kmaps`, `build_schema_for_db`) **inline, di proses yang sama** dengan retrieval+generation LLM (yang wajib jalan di GPU/Kaggle). Ini berarti `src/metrics/esm_ex_cm.py` (dan transitif: `external/spider_eval/evaluation.py`+`process_sql.py`) **harus** importable di lingkungan Kaggle supaya `pipeline.py --baseline` bisa jalan sama sekali — bertentangan langsung dengan klaim `requirements.txt` di atas.

**Belum diputuskan:** apakah `requirements.txt`-nya yang perlu diperbarui (akui `src/metrics/esm_ex_cm.py` minimal harus ada di Kaggle juga untuk jalur `--baseline`), atau `run_comparison()`-nya yang perlu dipisah (pisahkan step retrieval+generation dari step evaluasi ESM/EX/CM+raw_logs jadi dua proses/dua environment berbeda, sesuai niat awal requirements.txt). Belum ada dampak konkret ke hasil manapun sejauh ini — cuma inkonsistensi dokumentasi/arsitektur yang perlu diklarifikasi peneliti sebelum run Kaggle sungguhan.

### C. `src/core/config.py` mengikat SELURUH lapisan dimensi ke `torch` demi satu default dtype

**Ditemukan (2026-09-17),** saat mengisi `run_all_dimensions.py` (poin 24).

`src/core/config.py` baris 5 `import torch`, dipakai HANYA untuk satu default field: `bnb_compute_dtype: torch.dtype = torch.float16`. Tapi `src/dimensions/dim1_efficiency.py` `import PipelineConfig` dari situ (untuk μ = `token_output_weight`), jadi **Dimensi 1 tidak bisa jalan di environment tanpa torch** — padahal `requirements.txt` menyatakan lapisan `src/metrics/` + `src/dimensions/` justru dirancang jalan LOKAL, bukan di Kaggle. Lima dimensi lain tidak terpengaruh (tidak ada yang butuh `cfg`).

Efek nyata: dijalankan lokal apa adanya sekarang, `run_all_dimensions.py` melaporkan `Dimensi 1: DILEWATI (torch tidak ada)` — dan Dimensi 1 itu yang menjawab RM1, hasil headline skripsi.

**Opsi:**
1. `pip install torch` versi CPU di environment lokal (paling cepat, tidak menyentuh kode; ~200MB untuk satu konstanta dtype).
2. Tambahkan `torch` ke `requirements.txt` — mengakui lapisan evaluasi memang butuh torch, konsisten dengan opsi 1.
3. Decouple `config.py` dari torch: simpan dtype sebagai string (`"float16"`) dan resolve ke `torch.dtype` di `generation.py` saat load model. Paling bersih secara arsitektur, TAPI menyentuh jalur GPU yang sudah jalan (`load_model_and_tokenizer()`) — perlu diuji ulang di Kaggle sebelum dipercaya.
4. Biarkan; jalankan Dimensi 1 di Kaggle bersama pipeline, lima dimensi lain lokal.

**Belum diputuskan.** Terkait erat dengan open item B (kontradiksi "metrics jalan di mana") — sebaiknya diputuskan bersamaan, karena dua-duanya soal batas environment lokal vs Kaggle untuk lapisan evaluasi.

### D. EX per k untuk Dimensi 6 tidak tersimpan machine-readable

**Ditemukan (2026-09-17),** saat mengisi `run_all_dimensions.py` (poin 24).

`run_dimension_6()` butuh `ex_per_k` = `{0: EX, 1: EX, 3: EX, 5: EX}` (skala 0–100). Tidak ada file di project yang menyimpannya:
- `outputs/tables/ablation_results.csv` (ditulis `_save_csv()` di `ablation.py`) punya `avg_recall`, `avg_precision`, `avg_prompt_tokens`, `avg_output_tokens`, `avg_token_consumption`, `n_samples` — **tidak ada kolom EX/ESM**.
- EX per k dihitung `_run_ablation_evals()` dengan memanggil Spider `evaluation.py` sebagai **subprocess** per file prediksi; hasilnya ke stdout/log, tidak di-parse dan tidak disimpan.

Untuk sekarang `run_all_dimensions.py` menerima angkanya manual (`--ex-per-k` / `--ex-per-k-file`) dan men-skip Dimensi 6 dengan penjelasan kalau tidak diberikan (poin 24c).

**Belum diputuskan:** (1) tambahkan kolom `ex`/`esm` ke `ablation_results.csv` dengan cara meng-capture + parse stdout `evaluation.py` di `_run_ablation_evals()`; atau (2) pindahkan evaluasi ablation ke `esm_ex_cm.evaluate_single_query()` in-process (seperti yang sudah dilakukan `run_comparison()` di `pipeline.py`) supaya EX per k langsung tersedia sebagai angka, sekaligus menutup gap "`ablation.py` belum memproduksi raw_logs" yang sudah tercatat di CLAUDE.md; atau (3) biarkan manual — peneliti copy angka dari output Spider eval sekali saja setelah run ablation. Opsi 2 paling menyelesaikan akar masalah tapi menyentuh `ablation.py` yang jalan di Kaggle, jadi butuh konfirmasi peneliti dulu.
