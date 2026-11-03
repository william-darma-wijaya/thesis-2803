# Desain Penelitian: Skenario Pengujian & Penerapan Metrik Evaluasi

> Sumber: Proposal Skripsi — BAB 3, Subbab 3.8.1 dan 3.8.2

---

## Tujuan Evaluasi

Evaluasi sistem dilakukan untuk mengukur **kualitas**, **efisiensi**, dan **stabilitas** dari *query* SQL yang dihasilkan oleh *pipeline* GraphRAG Text-to-SQL secara kuantitatif:

- **Kualitas *query*** → ESM, EX, CM. ESM digunakan sebagai pengembangan dari pendekatan EM pada *benchmark* SPIDER untuk mengurangi sensitivitas terhadap perbedaan *syntax* non-*semantic*.
- **Efisiensi sistem** → Token Consumption dan TEP.
- **Stabilitas *generation query*** → QVT, untuk mengukur konsistensi *output* antar inferensi.

---

## 3.8.1 Skenario Pengujian

Pengujian menggunakan **development set** dataset SPIDER yang berisi pasangan pertanyaan *natural language* dan *gold* SQL pada berbagai *relational database* lintas domain. Dirancang dalam tiga skenario:

### Skenario 1 — Full GraphRAG Pipeline

Seluruh komponen sistem diaktifkan:
- *Graph construction*
- *Two-stage schema retrieval*
- *Graph path tracing*
- *Retrieval few-shot examples*
- *Post-processing* SQL

**Tujuan**: merepresentasikan kondisi operasional utama sistem yang diusulkan.

### Skenario 2 — Ablation Study Few-Shot Examples

Variasi jumlah contoh: **k = 0, 1, 3, 5** pada *subset development set*.

**Tujuan**: menganalisis pengaruh jumlah contoh terhadap kualitas *query* SQL yang dihasilkan. Kondisi k=0 merepresentasikan pendekatan *zero-shot* dan digunakan sebagai *baseline* internal untuk membandingkan efektivitas *few-shot retrieval*.

### Skenario 3 — Full Schema Bypass

*Retrieval* berbasis *graph* dinonaktifkan; proses *retrieval* dilakukan langsung pada level tabel tanpa memanfaatkan *graph*. Konteks *schema* tetap dipilih berdasarkan relevansi tabel, namun tanpa mempertimbangkan hubungan relasional antar *node* dalam *graph schema*.

**Tujuan**: mengukur kontribusi mekanisme GraphRAG terhadap kualitas *retrieval* dan performa akhir *generation* SQL.

### Ringkasan Skenario

| Skenario | Graph Retrieval | Few-Shot | Tujuan Utama |
|---|---|---|---|
| 1. Full GraphRAG | ✅ Aktif | ✅ Aktif | Kondisi operasional sistem usulan |
| 2. Ablation Few-Shot | ✅ Aktif | Variasi k = 0,1,3,5 | Menentukan jumlah contoh optimal |
| 3. Full Schema Bypass | ❌ Nonaktif (table-level) | ✅ Aktif | Mengukur kontribusi GraphRAG |

---

## 3.8.2 Penerapan Metrik Evaluasi

Lima metrik evaluasi kuantitatif (diuraikan teoritis pada Bab 2) diterapkan secara konkret pada pengujian *pipeline* GraphRAG Text-to-SQL.

### 3.8.2.1 Schema Linking Accuracy (SLA)

**Peran dalam riset**: metrik **diagnostik** untuk mengevaluasi seberapa akurat *pipeline* GraphRAG mengidentifikasi tabel dan kolom yang dibutuhkan **sebelum** proses *generation* SQL dilakukan.

**Implementasi**:
- Evaluasi pada dua tingkat granularitas: **tingkat tabel** dan **tingkat kolom**
- Elemen *schema* hasil prediksi *pipeline* dibandingkan dengan elemen *schema ground truth* dari *query* referensi pada **dev set** SPIDER
- Skor yang dilaporkan: *precision*, *recall*, dan *F1-score*
- **Penekanan khusus pada *recall***, mengingat kegagalan menemukan elemen *schema* relevan berdampak lebih fatal dibandingkan kelebihan elemen yang tidak diperlukan

Hasil SLA dianalisis pada seluruh skenario pengujian untuk mengidentifikasi apakah kesalahan *generation query* berakar dari tahap *retrieval schema linking* atau dari tahap *generation* itu sendiri.

---

### 3.8.2.2 Exact Set Match (ESM)

**Peran dalam riset**: **metrik utama**, karena mampu memberikan penilaian yang lebih adil terhadap *output query* LLM yang sering memiliki urutan penulisan berbeda dari *gold* SQL.

**Implementasi**:
- Setiap *query* didekomposisi menjadi komponen utama: `SELECT`, `FROM`, `WHERE`, `GROUP BY`, `ORDER BY`, `HAVING`, `LIMIT`, `INTERSECT`, `EXCEPT`, `UNION`
- Seluruh *query* pada **dev set** SPIDER digunakan sebagai data *testing*
- Skor ESM = proporsi *query* yang seluruh komponennya dinyatakan sesuai terhadap total *query* yang dievaluasi

---

### 3.8.2.3 Execution Accuracy (EX)

**Peran dalam riset**: **metrik pelengkap** terhadap ESM, untuk menangkap kemampuan model menghasilkan *query* yang benar secara **fungsional**. *Query* dengan perbedaan struktur namun *output* identik tetap dianggap benar.

**Implementasi**:
- *Database engine*: **SQLite**, sesuai spesifikasi resmi *benchmark* SPIDER
- Membandingkan hasil eksekusi *query* prediksi dengan *query gold* pada *database*

---

### 3.8.2.4 Component Match (CM)

**Peran dalam riset**: memberikan analisis yang lebih rinci dibandingkan ESM yang menilai *query* secara keseluruhan — mengevaluasi akurasi setiap klausa SQL secara terpisah.

**Komponen yang dievaluasi**:
- `SELECT`, `FROM`, `WHERE`, `GROUP BY`, `ORDER BY`, `HAVING`
- `KEYWORDS` (termasuk `DISTINCT` dan `LIMIT`)
- Operasi himpunan: `UNION`, `UNION ALL`, `INTERSECT`, `EXCEPT`

Hasil CM dianalisis pada seluruh skenario pengujian untuk melihat apakah penambahan komponen seperti GraphRAG atau *few-shot examples* memberikan peningkatan performa secara **merata pada seluruh klausa**, atau hanya pada bagian tertentu saja.

---

### 3.8.2.5 Token Consumption dan TEP

**Peran dalam riset**: menganalisis *trade-off* antara konsumsi token dengan peningkatan performa yang diperoleh dari penambahan konteks hasil *retrieval* GraphRAG dan *few-shot examples*.

**Implementasi Token Consumption**:

| Aspek | Spesifikasi |
|---|---|
| Tokenizer | Qwen2.5-Coder-7B-Instruct (sama dengan yang digunakan pada *generation*) |
| Token input | Seluruh isi *prompt*: *schema context* (DDL), *few-shot examples*, pertanyaan *natural language* |
| Token output | Hasil *generation* LLM **sebelum** *post-processing* |
| Faktor pengali output (α) | **3** (default) |

Faktor pengali α = 3 ditetapkan sesuai rasio harga yang umum diterapkan pada layanan API LLM komersial, di mana token output umumnya memiliki biaya lebih tinggi dibandingkan token input.

**Implementasi TEP**:
- Sistem yang diuji (G) = *pipeline* **GraphRAG penuh**
- *Baseline* (B) = konfigurasi **tanpa GraphRAG**
- Nilai TEP menjelaskan seberapa efisien tambahan token yang dikonsumsi sistem dalam menghasilkan peningkatan akurasi

---

### 3.8.2.6 Query Variance Testing (QVT)

**Peran dalam riset**: mengevaluasi *robustness pipeline* GraphRAG terhadap perubahan formulasi pertanyaan yang umum terjadi dalam penggunaan nyata.

**Implementasi**:
- *Pipeline* dijalankan **secara independen** untuk setiap variasi pertanyaan
- Skor QVT dihitung menggunakan rumus pada subbab 2.9.6
- Sesuai ketentuan Li et al. (2024): **hanya kelompok *query* yang memiliki setidaknya satu prediksi benar** yang disertakan dalam perhitungan skor QVT

Kriteria inklusi ini memastikan evaluasi difokuskan pada stabilitas model dalam mempertahankan prediksi yang konsisten terhadap variasi input.

---

## Peta Metrik ke Tahapan Pipeline

```
Pertanyaan NL
     ↓
[ Two-Stage Semantic Retrieval ]  ←── diukur oleh: SLA (precision/recall/F1)
     ↓
[ Graph Traversal ]
     ↓
[ Few-Shot Retrieval ]            ←── diukur oleh: Ablation k = 0,1,3,5
     ↓
[ Structured Prompt ]             ←── diukur oleh: Token Consumption (input)
     ↓
[ Qwen2.5-Coder-7B-Instruct ]     ←── diukur oleh: Token Consumption (output)
     ↓
[ Post-processing SQL ]
     ↓
Generated SQL                     ←── diukur oleh: ESM, EX, CM, QVT, TEP
```
