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
- **Jalur subprocess** (`pipeline.py`'s `run_official_evaluation()`, `ablation.py`'s `_run_ablation_evals()`, notebook `eval_pipeline.ipynb` Bagian 10) — `evaluation.py` CLI resmi membaca SATU file `--pred`/`--gold` dan memakai string yang SAMA untuk parsing MAUPUN eksekusi (dikonfirmasi `evaluation.py`'s `evaluate()`, baris ~501-547: `p_str = p[0]` di-`get_sql()` DAN dioper mentah ke `eval_exec_match(db, p_str, ...)`) — tidak ada cara memisahkan keduanya tanpa mengubah script resmi. Karena itu:
  - `--etype match` (dikonfirmasi baris ~546: `eval_exec_match` cuma dipanggil kalau `etype in ["all","exec"]`, jadi `match` TIDAK PERNAH mengeksekusi apapun) → file dinormalisasi PENUH, aman.
  - `--etype exec` → file HANYA dinormalisasi `execution_safe_only=True` (INNER/CROSS saja). **`LEFT`/`RIGHT`/`FULL JOIN` TETAP gagal parse di jalur subprocess-exec ini** — residual limitation yang diterima, bukan bug baru. Kalau nanti mau di-fix juga, satu-satunya cara adalah patch `external/spider_eval/process_sql.py` langsung, yang melanggar konvensi "kode resmi apa adanya" — **belum diputuskan, tanyakan ke peneliti dulu** kalau ini jadi prioritas.

**Alasan:** kode resmi SPIDER (`external/spider_eval/`) tetap byte-identical/tidak disentuh (konvensi atribusi di `CLAUDE.md`) — normalisasi selalu terjadi di teks INPUT sebelum masuk ke kode resmi, bukan modifikasi kode resminya. Satu fungsi regex dipakai semua call site (diminta peneliti) supaya tidak ada logic normalisasi yang duplikat/drift antar file.

**Lokasi implementasi:** `src/utils/sql_normalize.py` (baru), `src/metrics/esm_ex_cm.py` (`_evaluate_single_query_inner()`, baris ~195-204), `src/experiments/pipeline.py` (`run_official_evaluation()`), `src/experiments/ablation.py` (`_run_ablation_evals()`), `notebooks/eval_pipeline.ipynb` Bagian 10 (`run_spider_eval()`, cell markdown diupdate dengan catatan residual limitation).

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

## Belum diputuskan / open items

### A. `LEFT JOIN` / `RIGHT JOIN` / `INNER JOIN` tidak didukung parser resmi SPIDER

✅ **Diselesaikan (2026-08-25) — lihat poin 11 di atas.** Ditemukan bahwa `INNER`/`CROSS JOIN` juga crash (bukan cuma `LEFT`/`RIGHT`), dan EX (bukan cuma ESM/CM) juga ikut rusak. Semua dinormalisasi ke `JOIN` polos HANYA untuk salinan teks yang di-parse — jalur in-process (`esm_ex_cm.py`) dapat fix penuh untuk semua join type; jalur subprocess (`evaluation.py` CLI) dapat fix penuh untuk `--etype match` tapi cuma `INNER`/`CROSS` untuk `--etype exec` (residual limitation, lihat poin 11 untuk alasan arsitekturnya). Teks asli di bawah dipertahankan sebagai catatan historis.

**Temuan (2026-08-08):** `process_sql.py`'s `parse_from()` cuma mengenali token `join` (bare), tidak ada branch untuk `left`/`right`/`inner`. Predicted SQL yang pakai keyword-keyword itu GAGAL di-parse (`KeyError: 'left'` dkk, sudah diverifikasi langsung), jatuh ke fallback `_empty_sql()` yang sama dengan predicted SQL yang benar-benar rusak → ESM=0 otomatis, EX biasanya ikut 0 (karena `p_val_units` jadi kosong).

**Dampak potensial:** karena Qwen2.5-Coder (atau LLM manapun) sangat mungkin menghasilkan `LEFT JOIN` secara alami (idiom SQL yang sangat umum, kadang justru pilihan yang lebih tepat secara semantik untuk pertanyaan "which X have no Y"), ini berpotensi jadi sumber false-negative yang LEBIH besar dan lebih sistematis daripada isu order-sensitivity di poin 5.

**Catatan teknis:** `INNER JOIN` → `JOIN` aman dinormalisasi sebelum parsing (100% setara secara semantik di SQL standar). `LEFT JOIN`/`RIGHT JOIN` TIDAK aman dinormalisasi ke `JOIN` biasa — keduanya mempertahankan baris yang tidak match dengan `NULL`, beda perilaku dari inner join, jadi normalisasi buta akan mengubah semantik yang diukur, bukan cuma perbaikan kompatibilitas parser.

**Status (historis, sebelum poin 11):** belum diputuskan mau diapakan (dibiarkan sebagai known limitation vs investigasi seberapa sering LLM benar-benar memakai keyword ini di praktik).
