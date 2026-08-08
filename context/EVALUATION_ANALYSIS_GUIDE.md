# Panduan Implementasi Analisis Evaluasi — Skripsi Text-to-SQL (GraphRAG vs Baseline RAG)

> **Tujuan file ini:** Menjadi *single source of truth* bagi Claude Code saat mengimplementasikan pipeline evaluasi dan analisis. Semua formula, threshold, dan tabel interpretasi di bawah ini diambil **langsung dari proposal prethesis (Bab 2.9 dan 3.8)**. Jangan mengubah formula, threshold, atau urutan logika tanpa konfirmasi eksplisit dari peneliti — ini untuk mencegah halusinasi/asumsi yang menyimpang dari proposal.

---

## 0. Konteks Riset

- **Perbandingan:** Baseline RAG (vector-only retrieval) vs Graph-Based RAG (GraphRAG dengan schema-to-graph + graph traversal)
- **Dataset:** SPIDER dev set (SQLite), stratifikasi difficulty: Easy, Medium, Hard, Extra Hard
- **Model:** Qwen2.5-Coder-7B-Instruct (via Ollama)
- **Rumusan Masalah:**
  - **RM1:** Apakah GraphRAG berhasil menekan konsumsi token tanpa menurunkan performa sistem?
  - **RM2:** Bagaimana dampak pemangkasan token terhadap kualitas SQL dan ketahanan sistem?

## 0.1 Dua Kondisi yang Selalu Dibandingkan

Setiap dimensi analisis membandingkan **dua kondisi (G vs B)**:
- **B (Baseline)** = konfigurasi tanpa GraphRAG
- **G (GraphRAG)** = pipeline GraphRAG penuh

Skenario tambahan yang **harus tersedia sebagai data mentah** sebelum analisis dimulai (dari subbab 3.8.1):
1. **Skenario 1** — Full GraphRAG pipeline (graph construction, two-stage retrieval, graph path tracing, few-shot retrieval, post-processing) → ini adalah G
2. **Skenario 2** — Ablation jumlah few-shot, k ∈ {0, 1, 3, 5}, untuk menentukan k final yang dipakai seragam di G dan B (lihat Dimensi 6). *(Belum jadi prioritas saat ini — masih tahap trial-error pembuatan metrics dan kode per dimensi analisis)*
3. **Skenario 3** — Full schema bypass (retrieval langsung level tabel, tanpa graph) → ini adalah B

---

## 1. Definisi Metrik & Formula Wajib

> Setiap metrik di bawah dijelaskan dengan format yang sama: **(a) apa yang diukur & kenapa**, **(b) input yang dibutuhkan**, **(c) algoritma step-by-step**, **(d) contoh konkret**, **(e) hal yang sering salah diimplementasi**. Ikuti urutan langkah persis seperti tertulis.

---

### 1.1 Schema Linking Accuracy (SLA)

**(a) Apa & Kenapa**
SLA mengukur seberapa akurat pipeline retrieval (baik baseline vector-search maupun GraphRAG) berhasil menemukan tabel dan kolom yang **benar-benar dibutuhkan** untuk menjawab sebuah pertanyaan natural language, SEBELUM proses generation SQL dilakukan. Ini adalah metrik diagnostik di tahap retrieval, bukan di tahap generation. Kalau schema linking gagal (elemen yang dibutuhkan tidak ter-retrieve), LLM tidak akan pernah bisa menyusun SQL yang benar — walaupun LLM-nya sendiri sangat pintar.

**(b) Input yang dibutuhkan**
- Gold SQL (untuk di-parse jadi ground truth schema)
- Daftar schema lengkap tiap database (`db_schema[db_id]` → list semua `table.column` yang valid)
- Output retrieval dari pipeline (predicted schema): list `table.column` yang berhasil di-retrieve untuk query tersebut

**(c) Algoritma step-by-step**
1. Untuk setiap query di dev set, parse gold SQL untuk ekstrak semua `table.column` yang dipakai (SELECT, WHERE, JOIN, GROUP BY, ORDER BY, dsb). Ini jadi **ground truth schema set**.
2. Validasi tiap elemen ground truth: cek apakah `table.column` tersebut benar-benar ada di `db_schema[db_id]` (case-insensitive). Buang yang tidak valid (kemungkinan hasil parsing error).
3. Ambil predicted schema (hasil retrieval pipeline) untuk query yang sama. Validasi juga terhadap `db_schema[db_id]`.
4. Hitung di **dua level granularitas terpisah**:
   - **Table-level**: bandingkan hanya bagian nama tabel (`table`) dari kedua set
   - **Column-level**: bandingkan pasangan penuh `table.column`
5. Untuk tiap level, hitung per query:
   - `TP` (true positive) = irisan antara ground truth set dan predicted set
   - `Precision = |TP| / |predicted set|`
   - `Recall = |TP| / |ground truth set|`
   - `F1 = 2 · (Precision · Recall) / (Precision + Recall)`
6. Agregasi across semua query: laporkan **mean Precision, mean Recall, mean F1** (bukan cuma di-sum lalu dibagi total elemen — pakai rata-rata per-query, kecuali proposal secara eksplisit minta micro-average).
7. **Fokus pelaporan utama ada di Recall**, karena elemen schema yang gagal ditemukan jauh lebih fatal daripada elemen berlebih (dampaknya langsung ke tahap generation).

**(d) Contoh konkret**
Gold SQL butuh: `{students.name, students.age, enrollments.student_id}`
Predicted (hasil retrieval): `{students.name, students.age, students.gpa, enrollments.student_id}`
- TP = `{students.name, students.age, enrollments.student_id}` → 3 elemen
- Precision = 3/4 = 0.75 (ada 1 elemen ekstra: `students.gpa`)
- Recall = 3/3 = 1.0 (semua ground truth ketemu)
- F1 = 2 · (0.75 · 1.0) / (0.75 + 1.0) = 0.857

**(e) Kesalahan umum yang harus dihindari**
- Jangan bandingkan string schema secara case-sensitive — selalu `.lower()` dulu.
- Jangan lupa validasi ground truth terhadap `db_schema` — hasil parser SQL kadang menangkap alias atau subquery temp table yang bukan schema asli.
- Jangan campur table-level dan column-level jadi satu angka — proposal minta dua granularitas terpisah.

---

### 1.2 Exact Set Match (ESM)

**(a) Apa & Kenapa**
ESM mengevaluasi apakah SQL yang dihasilkan model **secara struktural setara** dengan gold SQL — tapi tidak seketat text-matching mentah. ESM dipilih sebagai metrik utama karena LLM sering menulis query yang valid secara logika tapi dengan urutan klausa/kolom berbeda dari ground truth; text-matching mentah akan salah menganggap itu sebagai kesalahan (false negative).

**(b) Input yang dibutuhkan**
- Predicted SQL (raw, sudah melalui post-processing)
- Gold SQL
- SQL parser yang bisa memecah query jadi komponen kanonik (proposal mengacu pada pendekatan Scholak et al., 2021 — kalau belum ada parser custom, gunakan library SQL parsing yang bisa breakdown ke clause-level, misalnya `sqlglot` atau parser resmi evaluasi SPIDER)

**(c) Algoritma step-by-step**
1. Parse predicted SQL dan gold SQL ke representasi kanonik/ternormalisasi (bukan string mentah).
2. Pecah masing-masing jadi komponen: `SELECT, FROM, WHERE, GROUP BY, ORDER BY, HAVING, LIMIT, INTERSECT, EXCEPT, UNION`.
3. Untuk tiap komponen, bandingkan sebagai **set** (bukan list berurutan) — artinya `SELECT a, b` dan `SELECT b, a` dianggap SAMA.
4. **Nilai literal DIABAIKAN** dalam ESM (beda dengan EX) — fokus hanya ke struktur logika, bukan angka/string spesifik di WHERE.
5. Query dinyatakan **match (1)** hanya jika **SEMUA komponen** yang ada di gold SQL cocok dengan predicted SQL. Kalau ada satu saja komponen yang tidak cocok → **0** (all-or-nothing di level query, tapi partial di level komponen — komponen-level ini yang dipakai untuk CM di 1.4).
6. `ESM = (jumlah query dengan match=1) / (total query dievaluasi) × 100%`

**(d) Contoh konkret**
Gold: `SELECT name, age FROM students WHERE age > 20`
Predicted: `SELECT age, name FROM students WHERE age > 18`
- SELECT: set `{name, age}` vs `{age, name}` → **match** (urutan diabaikan)
- FROM: `{students}` vs `{students}` → **match**
- WHERE: nilai literal `20` vs `18` **diabaikan** karena ESM tidak sensitif ke literal, tapi strukturnya (`age > <value>`) **match**
- Kesimpulan: ESM = 1 untuk query ini

**(e) Kesalahan umum yang harus dihindari**
- Jangan implementasi ESM sebagai string equality — ini paling sering jadi sumber bug (banyak false negative).
- Jangan ikutkan nilai literal dalam perbandingan — itu tugas EX, bukan ESM.
- Ingat: ESM itu all-or-nothing per query (semua komponen harus cocok), beda dengan CM yang melaporkan skor per komponen.

---

### 1.3 Execution Accuracy (EX)

**(a) Apa & Kenapa**
EX mengukur **kebenaran fungsional/semantic**: apakah hasil eksekusi predicted SQL di database sama dengan hasil eksekusi gold SQL, TERLEPAS dari perbedaan struktur query. Ini melengkapi ESM — query yang strukturnya beda tapi hasilnya identik tetap dihitung benar di EX.

**(b) Input yang dibutuhkan**
- Predicted SQL, gold SQL
- Database SQLite yang sesuai (`db_id`) untuk dieksekusi — **wajib SQLite**, sesuai spesifikasi resmi SPIDER, bukan PostgreSQL/Supabase.

**(c) Algoritma step-by-step**
1. Untuk tiap query, jalankan predicted SQL pada SQLite database `db_id` yang bersangkutan → dapatkan `result_predicted` (set/list of rows).
2. Jalankan juga gold SQL pada database yang sama → `result_gold`.
3. Bandingkan kedua hasil eksekusi:
   - Kalau predicted SQL error saat dieksekusi (syntax error, kolom tidak ada, dsb) → otomatis **0** (gagal), TANGKAP exception, jangan biarkan pipeline crash.
   - Kalau berhasil dieksekusi, bandingkan isi hasil (bukan cuma jumlah baris — isi datanya harus sama, biasanya dibandingkan sebagai set of tuples supaya urutan baris tidak masalah, kecuali ada `ORDER BY` eksplisit di gold SQL yang berarti urutan relevan).
   - Match → `1`, tidak match → `0`.
4. `EX = (jumlah query dengan hasil eksekusi identik) / (total query) × 100%`

**(d) Contoh konkret**
Gold: `SELECT name FROM students WHERE age > 20`
Predicted: `SELECT students.name FROM students WHERE students.age > 20`
- Struktur berbeda (ada prefix `students.`) → ESM mungkin gagal kalau normalizer tidak menangani alias dengan baik
- Tapi hasil eksekusi identik → **EX = 1**

**(e) Kesalahan umum yang harus dihindari**
- **Selalu bungkus eksekusi query dengan try-except** — predicted SQL dari LLM sangat mungkin error/invalid, dan itu harus dihitung sebagai EX=0, bukan bikin script crash.
- Jangan lupa: EX **sensitif terhadap nilai literal** (beda dari ESM) — karena yang dibandingkan adalah hasil eksekusi aktual.
- Hati-hati false positive: dua query dengan logika berbeda bisa menghasilkan output yang sama kebetulan (misalnya sama-sama return tabel kosong). Ini adalah keterbatasan EX yang sudah diakui di proposal (subbab 2.9.3), bukan bug yang perlu "diperbaiki" — cukup dicatat sebagai limitation.

---

### 1.4 Component Match (CM)

**(a) Apa & Kenapa**
CM adalah versi **granular** dari ESM — alih-alih memberi skor all-or-nothing per query, CM melaporkan akurasi **per klausa SQL secara independen**. Ini dipakai untuk debugging: klausa mana yang paling sering jadi sumber kesalahan (misalnya apakah model sering salah di WHERE tapi selalu benar di SELECT).

**(b) Input yang dibutuhkan**
- Sama seperti ESM: predicted SQL dan gold SQL yang sudah di-parse ke representasi kanonik (bisa reuse parser yang sama dengan ESM, tinggal ambil hasil perbandingan per-komponen sebelum di-AND-kan jadi keputusan all-or-nothing)

**(c) Algoritma step-by-step**
1. Parse kedua query ke komponen kanonik (sama seperti langkah ESM poin 1-2).
2. Untuk **setiap klausa** (`SELECT, FROM, WHERE, GROUP BY, ORDER BY, HAVING, KEYWORDS [DISTINCT, LIMIT], UNION, UNION ALL, INTERSECT, EXCEPT`), bandingkan secara terpisah sebagai set — hasilnya `1` (match) atau `0` (tidak match) PER klausa PER query.
   - Kalau suatu klausa tidak muncul di gold SQL maupun predicted SQL (misal tidak ada GROUP BY di keduanya), perlakukan sebagai **match otomatis (1)** untuk klausa itu — jangan exclude dari perhitungan, kecuali diputuskan lain oleh peneliti.
   - Kalau klausa hanya muncul di salah satu (gold ada GROUP BY, predicted tidak) → `0`.
3. Agregasi: untuk tiap klausa, hitung `akurasi_klausa = (jumlah query dengan klausa itu match) / (total query)`.
4. Output akhir adalah **tabel skor per klausa**, bukan satu angka gabungan. Jangan rata-ratakan semua klausa jadi satu skor tunggal kecuali diminta eksplisit — tujuan CM adalah granularitas.

**(d) Contoh konkret**
Dari 100 query dev set, misal hasil CM baseline:
```
SELECT: 92%   FROM: 95%   WHERE: 68%   GROUP BY: 74%   ORDER BY: 88%
```
→ Interpretasi: model paling banyak salah di WHERE dan GROUP BY, konsisten dengan temuan Yu et al. (2019) di paper SPIDER original.

**(e) Kesalahan umum yang harus dihindari**
- Jangan gabungkan semua klausa jadi satu skor rata-rata sebagai output utama — itu menghilangkan tujuan diagnostik CM.
- Pastikan kebijakan untuk klausa yang tidak muncul di kedua query (match otomatis) diterapkan konsisten di semua query, supaya perbandingan antar klausa adil.

---

### 1.5 Token Consumption (T) dan Token Elasticity of Performance (TEP)

**(a) Apa & Kenapa**
Token Consumption mengukur **biaya komputasi** per query (representasi realistis dari biaya API/inference). TEP adalah metrik custom (diadaptasi dari konsep elastisitas ekonomi) yang mengukur **seberapa efisien** perubahan token consumption diterjemahkan menjadi perubahan performa (EX). Ini adalah metrik novelty utama di riset ini — jangan sampai formulanya salah implementasi.

**(b) Input yang dibutuhkan**
- Untuk tiap query: jumlah token input (`T_in`) dan token output (`T_out`), dihitung pakai tokenizer **Qwen2.5-Coder-7B-Instruct** (bukan tokenizer lain seperti tiktoken/GPT tokenizer — hasilnya akan salah kalau beda tokenizer).
- `T_in` = seluruh isi prompt yang dikirim ke LLM: schema context (dalam bentuk DDL), few-shot examples, dan pertanyaan natural language.
- `T_out` = output mentah hasil generation LLM, **sebelum** post-processing (sebelum alias simplification, CAST removal, dsb dari subbab 3.7.2).
- EX aggregate untuk kondisi Baseline (`EX_B`) dan GraphRAG (`EX_G`) — dari hasil Dimensi 2/metrik 1.3.

**(c) Algoritma step-by-step — Token Consumption**
1. Untuk tiap query, hitung `T_in` = jumlah token dari prompt lengkap yang benar-benar dikirim ke LLM (pakai tokenizer Qwen2.5-Coder-7B-Instruct, method `.encode()` lalu `len()`).
2. Hitung `T_out` = jumlah token dari raw output LLM (sebelum post-processing).
3. Hitung `T = T_in + μ · T_out`, dengan `μ = 3` (default, jangan diubah kecuali peneliti eksplisit minta).
4. Simpan `T` per query. Agregasi: `mean(T)` dan `median(T)` untuk populasi query yang dievaluasi (lihat Dimensi 1 di Bagian 3 untuk breakdown per difficulty).

**(c) Algoritma step-by-step — TEP**
1. Pastikan sudah ada: `T_B` (rata-rata token consumption baseline), `T_G` (rata-rata token consumption GraphRAG), `EX_B` (execution accuracy baseline, dalam skala 0-1 atau %, **konsisten dengan skala EX yang dipakai di seluruh laporan**), `EX_G`.
2. Hitung `ΔEX_G = EX_G − EX_B`
3. Hitung `ΔT_G = T_G − T_B`
4. Hitung `TEP_G = (ΔEX_G / EX_B) / (ΔT_G / T_B)`
5. Interpretasikan pakai tabel di Bagian 3 Dimensi 1 (TEP<0 ideal, TEP≈0 netral, TEP>0 ada trade-off).

**(d) Contoh konkret**
`EX_B = 60%`, `EX_G = 68%`, `T_B = 2000 token`, `T_G = 1400 token`
- `ΔEX_G = 68 − 60 = 8`
- `ΔT_G = 1400 − 2000 = −600`
- `TEP_G = (8/60) / (−600/2000) = 0.1333 / (−0.3) = −0.444`
- TEP < 0 → skenario ideal: GraphRAG lebih akurat DAN lebih hemat token.

**(e) Kesalahan umum yang harus dihindari**
- **Jangan pakai tokenizer selain Qwen2.5-Coder-7B-Instruct** — angka token akan salah dan tidak sebanding dengan biaya inference aktual.
- Jangan hitung `T_out` dari hasil SETELAH post-processing — itu bukan token yang benar-benar dikeluarkan LLM.
- Pastikan `EX_B` dan `EX_G` dalam skala yang sama (kalau EX dalam persen, pastikan keduanya persen; kalau desimal 0-1, pastikan konsisten) — kalau tercampur, hasil TEP akan salah total.
- `μ = 3` adalah default dari proposal (subbab 3.8.2.5) — jangan diganti nilai lain tanpa instruksi eksplisit.

---

### 1.6 Query Variance Testing (QVT)

**(a) Apa & Kenapa**
QVT mengukur **robustness/konsistensi** sistem: apakah model tetap menghasilkan SQL yang benar ketika pertanyaan yang SAMA secara makna ditulis dengan berbagai variasi (sinonim, gaya formal/informal, struktur kalimat berbeda). Berbeda dari ESM/EX yang mengukur ketepatan hasil, QVT mengukur stabilitas prediksi terhadap variasi input.

**(b) Input yang dibutuhkan**
- Untuk tiap gold SQL `Q_i`, kumpulan variasi pertanyaan NL: `N_i1, N_i2, ..., N_im` (variasi ini harus disiapkan/dibuat terpisah — TIDAK datang otomatis dari SPIDER dev set biasa, kemungkinan perlu digenerate atau pakai dataset variasi tambahan)
- Hasil prediksi pipeline untuk **setiap** variasi pertanyaan tersebut: `F(N_ij)`

**(c) Algoritma step-by-step**
1. Untuk tiap gold SQL `Q_i`, kumpulkan semua variasi NL question `N_i1...N_im` beserta hasil prediksi SQL `F(N_ij)` dari pipeline.
2. **Filter wajib**: buang `Q_i` dari perhitungan kalau **SEMUA** variasi `N_ij` gagal diprediksi benar (tidak ada satupun `F(N_ij) = Q_i`). Hanya `Q_i` dengan minimal 1 prediksi benar yang dimasukkan ke perhitungan QVT (aturan ini dari Li et al., 2024, WAJIB diikuti, jangan hitung semua Q_i tanpa filter).
3. Untuk tiap `Q_i` yang lolos filter, hitung skor level-1: `skor_i = (jumlah j dengan F(N_ij) = Q_i) / m_i` (proporsi variasi yang berhasil diprediksi benar untuk query itu). "Benar" di sini idealnya diukur pakai EX atau ESM — tentukan salah satu secara konsisten dan dokumentasikan pilihannya.
4. Hitung level-2 (skor akhir): `QVT = mean(skor_i)` untuk semua `Q_i` yang lolos filter di langkah 2. Formula lengkap:
```
QVT = (1/M) · Σ_{i=1}^{M} [ skor_i ]
```
dengan `M` = jumlah `Q_i` yang lolos filter (BUKAN total semua Q_i di dataset).
5. Untuk analisis Dimensi 4, hitung `QVT_B` dan `QVT_G` dengan cara yang sama, lalu `ΔQVT = QVT_G − QVT_B`.

**(d) Contoh konkret**
Gold SQL `Q_1` punya 5 variasi NL question. Hasil prediksi: 3 dari 5 variasi menghasilkan SQL yang benar (match dengan `Q_1`).
- `skor_1 = 3/5 = 0.6`
- `Q_1` lolos filter (karena ada minimal 1 yang benar)
- Kalau `Q_2` punya 4 variasi dan SEMUA gagal → `Q_2` **dibuang** dari perhitungan QVT sama sekali (tidak dihitung sebagai `skor_2 = 0`)

**(e) Kesalahan umum yang harus dihindari**
- **Jangan lupa filter di langkah 2** — ini kesalahan paling umum. Kalau semua `Q_i` dimasukkan tanpa filter, QVT akan under-estimate dan tidak sesuai definisi proposal.
- Pastikan variasi NL question sudah disiapkan sebagai data terpisah sebelum mulai menghitung QVT — ini bukan sesuatu yang otomatis ada di SPIDER dev set standar.
- Definisikan dengan jelas kriteria "benar" (`F(N_ij) = Q_i`) — apakah pakai EX atau ESM — dan pakai definisi yang sama untuk seluruh perhitungan QVT (jangan campur).

---

### 1.6 Query Variance Testing (QVT)
- Setiap gold SQL `Q_i` punya beberapa variasi NL question `N_i1 ... N_im`
- **Syarat inklusi:** hanya `Q_i` dengan minimal 1 variasi NL yang berhasil diprediksi benar yang dimasukkan ke evaluasi QVT
- Formula:
```
QVT = (1/M) · Σ_{i=1}^{M} [ (Σ_{j=1}^{m_i} 1(F(N_ij) = Q_i)) / m_i ]
```
- Dua layer: (1) persentase variasi benar per query SQL, (2) rata-rata across semua query SQL

---

## 2. Struktur Data Mentah yang Dibutuhkan (Prasyarat Sebelum Analisis)

Sebelum implementasi 6 dimensi analisis, pastikan pipeline sudah menghasilkan **raw log per query** dengan field minimal berikut (per kondisi B dan G, per query di dev set):

```json
{
  "query_id": "...",
  "db_id": "...",
  "difficulty": "easy|medium|hard|extra_hard",
  "gold_sql": "...",
  "predicted_sql": "...",
  "gold_schema": ["table.column", ...],
  "predicted_schema": ["table.column", ...],
  "token_input": int,
  "token_output": int,
  "esm_result": 0 | 1,
  "ex_result": 0 | 1,
  "cm_per_clause": {"SELECT": 0|1, "FROM": 0|1, "WHERE": 0|1, ...}
}
```

Untuk QVT, dibutuhkan file terpisah berisi variasi NL question per gold SQL beserta hasil prediksi masing-masing variasi.

**Jangan mulai Dimensi 1–6 sebelum data ini lengkap untuk kedua kondisi (B dan G).**

---

## 3. Alur Berpikir & Urutan Implementasi per Dimensi Analisis

> Urutan actual eksekusi eksperimen: **Dimensi 6 (Ablation Few-Shot) dijalankan PALING AWAL** untuk menentukan k final, baru setelah itu Dimensi 1–5 dijalankan dengan k yang sudah fix. Tapi karena user menanyakan "1 per 1" sesuai nomor dimensi di Tabel 3.9, panduan di bawah tetap diurutkan 1→6, dengan catatan dependency di Dimensi 6.

### 🔹 Dimensi 1 — Efisiensi Token dan Performa
**Menjawab:** RM1 — Apakah GraphRAG menekan konsumsi token tanpa menurunkan performa?
**Metrik:** Token Consumption, EX, TEP

**Alur implementasi:**
1. Hitung `T` per query untuk kondisi B dan G (formula 1.5), pakai tokenizer Qwen2.5-Coder-7B-Instruct, `μ=3`
2. Agregasi Token Consumption: tampilkan **mean dan median** (median penting karena distribusi token biasanya skewed oleh query kompleks), pecah juga per difficulty level (Easy/Medium/Hard/Extra Hard)
3. Hitung EX untuk kondisi B dan G dengan cara yang sama (agregat + per difficulty)
4. Setelah Token Consumption dan EX baseline vs GraphRAG sudah tampil berdampingan, **baru** hitung TEP menggunakan formula di 1.5
5. Interpretasikan TEP menggunakan tabel berikut (WAJIB pakai tabel ini persis, jangan buat interpretasi sendiri):

| Nilai TEP | Interpretasi |
|---|---|
| TEP < 0 | Skenario ideal: GraphRAG meningkatkan EX **dan** menurunkan Token Consumption dibanding baseline |
| TEP ≈ 0 | Efisiensi token tercapai tanpa trade-off penurunan EX yang signifikan |
| TEP > 0 | Ada trade-off: penurunan token disertai penurunan EX |

**Output yang harus ditampilkan:** tabel Token Consumption (mean, median, per difficulty) → tabel EX (per difficulty) → nilai TEP tunggal + interpretasi.

---

### 🔹 Dimensi 2 — Akurasi Struktur SQL
**Menjawab:** RM2 — Dampak pemangkasan token terhadap kebenaran struktural SQL
**Metrik:** ESM, EX (dibaca **berpasangan**, bukan terpisah)

**Alur implementasi:**
1. Hitung ESM dan EX untuk B dan G (agregat + per difficulty)
2. Tampilkan keduanya berdampingan dalam satu tabel per kondisi
3. Klasifikasikan hasil ke 4 kombinasi berikut (tabel wajib, sudah final dari proposal):

| Kombinasi | Interpretasi |
|---|---|
| EX naik, ESM turun | Penghematan token tidak merusak logika semantic meski struktur berbeda dari ground truth |
| ESM naik, EX turun | Struktur SQL ditiru tepat tapi gagal eksekusi → indikasi kehilangan detail nilai literal |
| EX naik, ESM naik | Kondisi paling ideal — konteks ringkas GraphRAG presisi |
| EX rendah, ESM rendah | Kegagalan menyeluruh → lanjut ke Dimensi 5 (Bottleneck Analysis) |

**Catatan penting:** Jangan menyimpulkan superioritas GraphRAG hanya dari salah satu metrik — laporan HARUS membaca EX dan ESM sebagai pasangan sesuai tabel di atas.

---

### 🔹 Dimensi 3 — Komponen SQL
**Menjawab:** RM2 — Klausa SQL mana yang paling sering error?
**Metrik:** CM (per klausa)

**Alur implementasi:**
1. Hitung skor CM per klausa (`SELECT, FROM, WHERE, GROUP BY, ORDER BY, HAVING, KEYWORDS, UNION, UNION ALL, INTERSECT, EXCEPT`) untuk B dan G
2. Tampilkan sebagai tabel perbandingan per-klausa (bukan satu angka agregat)
3. Identifikasi klausa dengan penurunan skor terbesar antara B → G
4. Interpretasi kualitatif (ikuti logika proposal, jangan generalisasi berlebihan):
   - Penurunan terbesar di **WHERE** → dampak pemangkasan konteks terasa di logika filtering
   - Penurunan terbesar di **JOIN/FROM** → GraphRAG kehilangan informasi relasi antar tabel
5. Gunakan hasil ini untuk mendukung diagnosis di Dimensi 5 (apakah error bersumber dari retrieval atau reasoning)

---

### 🔹 Dimensi 4 — Robustness dan Konsistensi Query
**Menjawab:** RM2 — Apakah sistem tetap stabil saat pertanyaan diparafrase?
**Metrik:** QVT, ESM (berpasangan)

**Alur implementasi:**
1. Jalankan pipeline B dan G secara independen untuk setiap variasi NL question
2. Filter: hanya sertakan gold SQL yang punya ≥1 variasi terprediksi benar (syarat wajib QVT, subbab 2.9.6)
3. Hitung QVT untuk B dan G memakai formula di 1.6
4. Hitung `ΔQVT = QVT_G − QVT_B`
5. Interpretasikan pakai tabel berikut (threshold ±2%, WAJIB, jangan diubah — alasannya: variasi antar metode SOTA di literatur hanya 3-4 poin persen, dan model di riset ini tidak fine-tuned sehingga lebih rentan noise):

| Selisih QVT | Interpretasi |
|---|---|
| ΔQVT ≥ +2% | GraphRAG meningkatkan konsistensi query secara meaningful |
| −2% ≤ ΔQVT < +2% | Konsistensi relatif stabil, pemangkasan token tidak merusak robustness |
| ΔQVT < −2% | GraphRAG menurunkan konsistensi — indikasi pemangkasan berlebih |

6. Silangkan dengan ESM (apakah query yang tidak konsisten juga bermasalah di ESM) untuk analisis lebih lengkap

---

### 🔹 Dimensi 5 — Bottleneck Retrieval vs Generation
**Menjawab:** Di mana sumber kegagalan utama pipeline?
**Metrik:** F1-Score SLA, EX (berpasangan) — **hanya dijalankan jika pipeline belum optimal** (trigger dari Dimensi 2, kondisi "EX rendah, ESM rendah")

**Alur implementasi:**
1. Hitung F1-Score SLA (dari precision & recall di Dimensi/Metrik 1.1) dan EX untuk kondisi yang dianalisis
2. Terapkan threshold (WAJIB, sudah final dari proposal, jangan diganti):
   - F1-Score SLA threshold: **80%**
   - EX threshold: **65%**
3. Gunakan tabel diagnostik berikut:

| F1-Score (Retrieval) | EX (Generation) | Interpretasi |
|---|---|---|
| ≥ 80% | ≥ 65% | Pipeline sehat, retrieval & generation optimal bersamaan |
| ≥ 80% | < 65% | Kegagalan di tahap **generation** — schema relevan berhasil didapat, LLM gagal menyusun SQL |
| < 80% | < 65% | Kegagalan di tahap **retrieval** — schema gagal diambil sejak awal |
| < 80% | ≥ 65% | Jarang terjadi — kemungkinan query sederhana atau LLM berhasil menalar dari schema parsial |

4. Jika F1-Score rendah, pecah lagi jadi precision vs recall secara terpisah:
   - **Recall rendah = lebih fatal** (elemen schema hilang → SQL tidak bisa disusun benar)
   - **Precision belum sempurna = masih bisa ditoleransi** (noise/JOIN redundan, tapi tidak mematikan kemampuan LLM)

---

### 🔹 Dimensi 6 — Ablation Few-Shot
**Menjawab:** Berapa jumlah contoh (k) optimal sebelum diminishing returns?
**Metrik:** EX vs k

**⚠️ Catatan dependency (untuk nanti):** Idealnya dimensi ini dijalankan **sebelum** Dimensi 1–5, karena tujuannya menentukan **satu nilai k tetap** yang dipakai seragam di kondisi Baseline dan GraphRAG (supaya perbandingan apple-to-apple). Belum perlu diterapkan selama masih tahap trial-error pembuatan metrics dan kode per dimensi.

**Alur implementasi:**
1. Ambil 20% dari **train set** SPIDER (bukan dev set), pakai **stratified sampling** berdasarkan difficulty level
2. Jalankan pipeline dengan k ∈ {0, 1, 3, 5}
   - k=0 = zero-shot, jadi baseline internal
3. Hitung EX untuk tiap nilai k
4. Plot/tabelkan EX vs k, baca **pola transisi** antar nilai k (bukan mencari titik optimal presisi):
   - Jika EX naik signifikan dari k=0→k=1 lalu stabil → sistem sensitif hanya pada contoh pertama (diminishing returns)
   - Jika perubahan EX kecil & konsisten di semua transisi → sistem tidak terlalu bergantung pada jumlah contoh
   - Jika EX turun di k besar → indikasi context overload / over-prompting
5. Tentukan k final berdasarkan pola ini, lalu gunakan k tersebut secara tetap untuk seluruh eksperimen di Dimensi 1–5

---

## 4. Aturan Umum Implementasi (Anti-Halusinasi)

1. **Jangan mengarang formula baru.** Semua rumus di file ini adalah final dari proposal — kalau ada kebutuhan metrik tambahan, tanyakan ke peneliti dulu, jangan asumsikan.
2. **Jangan mengubah threshold** (TEP, ±2% QVT, 80% F1-SLA, 65% EX) tanpa instruksi eksplisit.
3. **Selalu laporkan mean DAN median** untuk Token Consumption (distribusi cenderung skewed).
4. **Selalu pecah hasil per difficulty level** (Easy/Medium/Hard/Extra Hard) di samping angka agregat, untuk semua metrik utama (EX, ESM, Token Consumption).
5. **Jangan menyimpulkan superioritas satu metode hanya dari satu metrik.** Ikuti urutan dimensi: efisiensi (D1) → struktur (D2) → komponen (D3) → robustness (D4) → bottleneck (D5, kondisional) → ablation (D6, prasyarat).
6. **Urutan output per dimensi harus eksplisit**: tampilkan metrik mentah dulu (tabel angka B vs G), baru metrik turunan (TEP/ΔQVT), baru interpretasi berdasarkan tabel yang sudah ditentukan.
7. Semua evaluasi ESM/EX/CM dilakukan di atas **SPIDER dev set** dengan **SQLite** sebagai database engine.
8. Tokenizer untuk Token Consumption **harus** tokenizer Qwen2.5-Coder-7B-Instruct — jangan pakai tiktoken atau tokenizer lain sebagai pengganti.

---

## 5. Referensi Subbab Proposal (untuk traceability)

| Dimensi | Subbab Metrik | Subbab Analisis |
|---|---|---|
| 1. Efisiensi Token & Performa | 2.9.5, 3.8.2.5 | 3.8.3.1 |
| 2. Akurasi Struktur SQL | 2.9.2–2.9.3, 3.8.2.2–3.8.2.3 | 3.8.3.2 |
| 3. Komponen SQL | 2.9.4, 3.8.2.4 | 3.8.3.3 |
| 4. Robustness & Konsistensi | 2.9.6, 3.8.2.6 | 3.8.3.4 |
| 5. Bottleneck Retrieval vs Generation | 2.9.1, 3.8.2.1 | 3.8.3.5 |
| 6. Ablation Few-Shot | 3.8.1 (skenario 2) | 3.8.3.6 |

Kalau Claude Code menemukan ambiguitas yang tidak tercakup file ini, **tanyakan ke peneliti** — jangan berasumsi.
