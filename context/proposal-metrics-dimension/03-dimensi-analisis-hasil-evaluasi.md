# Dimensi Analisis Hasil Evaluasi

> Sumber: Proposal Skripsi — BAB 3, Subbab 3.8.3

---

## Prinsip Analisis

Analisis hasil evaluasi bertujuan menguraikan bagaimana setiap metrik dipahami, diinterpretasi, dan **dihubungkan satu sama lain** untuk menjawab rumusan masalah. Analisis tidak menilai semua metrik secara terpisah, karena **angka tinggi pada satu metrik saja belum cukup untuk menjamin keunggulan suatu konfigurasi**.

| Rumusan Masalah | Fokus | Metrik Bertumpu |
|---|---|---|
| **RM-1** | Efektivitas penghematan token GraphRAG tanpa menurunkan performa | Token Consumption, TEP, EX |
| **RM-2** | Dampak pemangkasan token terhadap kualitas SQL dan ketahanan sistem | ESM, CM, QVT |

---

## Tabel 3.9 — Pemetaan Dimensi Analisis, Metrik Evaluasi, dan Rumusan Masalah

| Dimensi Analisis | Metrik yang Dianalisa | Pertanyaan yang Dijawab |
|---|---|---|
| **Efisiensi Token dan Performa** | Token Consumption, EX, TEP | Apakah GraphRAG berhasil menekan konsumsi token tanpa menurunkan performa sistem? |
| **Akurasi Struktur SQL** | ESM, EX | Bagaimana dampak pemangkasan token terhadap kebenaran struktural SQL yang dihasilkan? |
| **Komponen SQL** | CM | Bagian komponen SQL mana yang paling sering mengalami kesalahan? |
| **Robustness dan Konsistensi Query** | QVT, ESM | Apakah sistem tetap stabil dan konsisten ketika pertanyaan diparafrase? |
| **Bottleneck Retrieval vs Generation** | Schema Linking Accuracy, EX, CM | Di mana sumber kegagalan utama *pipeline*: pengambilan *schema* atau penalaran LLM? |
| **Ablation Few-Shot** | Jumlah contoh (k), EX | Berapa jumlah contoh optimal sebelum terjadi *diminishing returns*? |

---

## 3.8.3.1 Analisis Trade-off Efisiensi Token dan Performa

**Menjawab**: Rumusan Masalah 1

Analisis difokuskan pada hubungan antara **Token Consumption**, **EX**, dan **TEP**.

- Penurunan Token Consumption pada GraphRAG diinterpretasikan sebagai kemampuan metode mengurangi konteks *database schema* yang tidak relevan selama *generation* SQL.
- Nilai EX digunakan untuk mengukur dampak pengurangan konteks tersebut terhadap akurasi eksekusi *query*.
- TEP mengintegrasikan keduanya: mengukur perubahan performa relatif terhadap perubahan konsumsi token.

### Logika Tanda TEP

Karena GraphRAG diharapkan menggunakan **lebih sedikit** token dibandingkan *baseline*, nilai perubahan token (penyebut TEP) akan **bernilai negatif**. Konsekuensinya:

- Jika EX **meningkat** → pembilang positif, penyebut negatif → **TEP < 0** (ideal)
- Jika EX **menurun** → pembilang negatif, penyebut negatif → **TEP > 0** (ada *trade-off*)

### Tabel 3.10 — Interpretasi TEP pada Kondisi Penghematan Token

| Nilai TEP | Interpretasi Hasil |
|---|---|
| **TEP < 0** | Skenario ideal: GraphRAG berhasil meningkatkan akurasi eksekusi **sekaligus** menurunkan jumlah konsumsi token dibandingkan *baseline*. |
| **TEP ≈ 0** | Skenario efisiensi token: tidak ada *trade-off* penurunan EX yang signifikan. |
| **TEP > 0** | Terdapat *trade-off* antara penurunan token yang dikonsumsi dengan akurasi eksekusi yang diukur dengan EX. |

### Prosedur Interpretasi

1. Baca kondisi Token Consumption dan EX pada **baseline** → tetapkan titik referensi performa sistem.
2. Baca kondisi yang sama pada **GraphRAG** → bandingkan.
3. Evaluasi nilai TEP menggunakan referensi Tabel 3.10 → tentukan ada/tidaknya *trade-off*.

---

## 3.8.3.2 Analisis Akurasi Generation SQL

**Menjawab**: Rumusan Masalah 2

Evaluasi dilakukan dengan membaca **EX dan ESM secara berpasangan**, karena keduanya mengukur dimensi kebenaran yang berbeda dan saling menutupi kelemahan masing-masing.

### Karakteristik Komplementer

| | ESM | EX |
|---|---|---|
| **Mekanisme** | *Parsing* kedua *query* ke struktur ternormalisasi, lalu bandingkan kecocokan klausa (Scholak et al., 2021) | Bandingkan hasil eksekusi *query* prediksi vs *ground truth* langsung pada *database* (Yu et al., 2019) |
| **Sensitivitas nilai literal** | Tidak sensitif | Sensitif |
| **Kelemahan** | Menurun ketika LLM menghasilkan formulasi SQL berbeda dari *ground truth* meskipun maknanya ekuivalen | Rentan *false positive* — *query* dengan logika berbeda dapat menghasilkan keluaran sama secara kebetulan |
| **Bias** | Cenderung **meremehkan** akurasi model | Cenderung **melebih-lebihkan** akurasi model |

> Shi et al. (2025) mencatat bahwa metrik berbasis EM cenderung meremehkan akurasi model, sementara EX cenderung melebih-lebihkannya.

Dengan membaca keduanya secara berpasangan, dapat dibedakan apakah perubahan performa berasal dari **kehilangan informasi literal**, **perbedaan formulasi SQL yang tidak substansial**, atau **kegagalan struktural yang lebih mendasar**.

### Tabel 3.11 — Interpretasi Hasil EX dan ESM

| Kombinasi Metrik | Interpretasi Hasil |
|---|---|
| **EX meningkat, ESM menurun** | Penghematan token tidak merusak logika *semantic* SQL meskipun strukturnya berbeda dengan *ground truth*. |
| **ESM meningkat, EX menurun** | Struktur SQL berhasil ditiru dengan tepat, tetapi gagal pada saat eksekusi. Mengindikasikan penghematan token menyebabkan sistem kehilangan detail nilai literal. |
| **EX meningkat, ESM meningkat** | Kondisi paling ideal: konteks ringkas yang dihasilkan GraphRAG terbukti presisi sehingga SQL dapat ditiru dengan benar dan menghasilkan *output database* yang tepat. |
| **EX rendah, ESM rendah** | Kegagalan menyeluruh — perlu ditelusuri lebih lanjut dengan analisis *Bottleneck* (3.8.3.5). |

### Prosedur Interpretasi

1. Baca kondisi EX dan ESM pada sistem **baseline** → pahami akurasi tanpa konfigurasi penghematan token.
2. Baca hasil **GraphRAG** → bandingkan dengan *baseline* → kategorikan ke dalam Tabel 3.11.

---

## 3.8.3.3 Analisis Komponen SQL

**Menjawab**: Rumusan Masalah 2

Menggunakan **CM** untuk mengidentifikasi bagian *query* SQL yang menjadi sumber kesalahan. Dengan membandingkan skor CM antara *baseline* dan GraphRAG, dapat diketahui komponen mana yang paling terdampak oleh pemangkasan konteks.

### Contoh Pola Interpretasi

| Pola yang Diamati | Interpretasi |
|---|---|
| Skor CM pada klausa **WHERE** lebih rendah dibanding klausa lain | Dampak pemangkasan token dirasakan pada **logika *filtering*** |
| Penurunan terbesar terjadi pada klausa **JOIN** | GraphRAG kehilangan informasi penting mengenai **hubungan antar tabel** saat membangun *query* yang lebih kompleks |

Hasil analisis ini digunakan untuk menentukan apakah kesalahan lebih disebabkan oleh **keterbatasan proses *retrieval*** atau **kemampuan penalaran model**.

---

## 3.8.3.4 Analisis Robustness dan Konsistensi Query

**Menjawab**: Rumusan Masalah 2

Dilakukan menggunakan metrik **QVT dan ESM secara berpasangan**, dengan pendekatan berbasis **selisih relatif (ΔQVT)** antara konfigurasi GraphRAG dan *baseline*.

### Justifikasi Pendekatan ΔQVT dan Threshold ±2%

1. *Benchmark* QVT yang tersedia di literatur hanya mencakup model berukuran besar yang telah melalui *fine-tuning*, sehingga tidak dapat dibandingkan langsung dengan model dalam penelitian ini.
2. Perbedaan QVT antar metode SOTA hanya berkisar **3–4 poin persentase**, sehingga perubahan sebesar **2 poin** ke arah manapun sudah mewakili sekitar separuh dari total variasi yang diamati dan layak dianggap *meaningful* (B. Li et al., 2024).
3. Model dalam penelitian ini **tidak melalui *fine-tuning***, sehingga variasi skor QVT lebih rentan terhadap *noise* dibandingkan model SOTA referensi. *Threshold* ±2% memberikan batas konservatif untuk menghindari interpretasi berlebih terhadap perubahan yang tidak substansial.

### Tabel 3.12 — Interpretasi Nilai QVT

| Selisih QVT | Interpretasi Hasil |
|---|---|
| **ΔQVT ≥ +2%** | GraphRAG meningkatkan konsistensi *query* secara *meaningful*. |
| **−2% ≤ ΔQVT < +2%** | Konsistensi relatif stabil; pemangkasan token tidak merusak *robustness*. |
| **ΔQVT < −2%** | GraphRAG menurunkan konsistensi — indikasi pemangkasan berlebih. |

---

## 3.8.3.5 Analisis Bottleneck Retrieval dan Generation

**Tujuan**: mengidentifikasi sumber kegagalan sistem apabila *pipeline* belum optimal — apakah kegagalan terletak pada tahap ***retrieval*** (gagal mengambil *schema* yang tepat) atau pada tahap ***generation*** (LLM gagal menyusun SQL yang benar meski *schema* sudah sesuai).

**Dua metrik tolok ukur**:
- **F1-Score SLA** → menilai kualitas *retrieval*
- **EX** → menilai kualitas *generation* LLM

Pendekatan diagnostik berbasis dua metrik ini sejalan dengan temuan Glass et al. (2025) yang menunjukkan bahwa kualitas *schema linking* berkorelasi langsung dengan EX pada *pipeline* Text-to-SQL.

### Penetapan Threshold

| Metrik | Threshold | Dasar Penetapan |
|---|---|---|
| **F1-Score SLA** | **80%** | Glass et al. (2025): EX meningkat signifikan ketika F1-Score *schema linking* mencapai 80% ke atas |
| **EX** | **65%** | Performa tipikal *code-focused* 7B model pada *benchmark* SPIDER (D. Gao et al., 2023; B. Li et al., 2024) — Deepseek-Coder-7B dan CodeLlama-7B mencapai EX ~66–80% dengan *supervised fine-tuning*. Digunakan sebagai referensi estimasi mengingat Qwen2.5-Coder-7B-Instruct merupakan model dengan kemampuan *coding* sebanding. |

### Tabel 3.13 — Diagnostik Sumber Kegagalan Pipeline

| F1-Score (Retrieval) | EX (Generation) | Interpretasi Hasil |
|---|---|---|
| **F1 ≥ 80%** | **EX ≥ 65%** | *Pipeline* beroperasi secara sehat — *retrieval* dan *generation* bekerja optimal secara bersamaan. |
| **F1 ≥ 80%** | **EX < 65%** | Titik kegagalan pada tahap **generation**. *Schema* yang relevan berhasil dikumpulkan, tetapi LLM gagal menyusun SQL dengan benar dari *schema* tersebut. |
| **F1 < 80%** | **EX < 65%** | Titik kegagalan pada tahap **retrieval**. Sistem gagal mengambil *schema* yang diperlukan sejak awal sehingga LLM tidak memiliki informasi yang cukup untuk menjawab *query*. |
| **F1 < 80%** | **EX ≥ 65%** | Skenario jarang terjadi. Kemungkinan: *query* sangat sederhana sehingga tidak memerlukan banyak konteks *schema*, atau LLM berhasil menalar dari *schema* parsial yang tersedia. |

### Pemeriksaan Lanjutan: Precision vs Recall

Saat F1-Score SLA rendah, **precision dan recall perlu diperiksa secara terpisah** karena keduanya tidak memiliki bobot yang setara:

- **Recall rendah** → jauh lebih merugikan. Elemen *schema* yang tidak terjangkau membuat *query* tidak bisa disusun secara benar. Ketika *recall schema linking* tidak mencapai 100%, EX turun ke kisaran **15–40%** tergantung variasi dataset.
- **Precision belum sempurna** → masih dapat ditoleransi. Elemen *schema* berlebih memang menambah *noise* dan berpotensi memunculkan JOIN redundan, tetapi tidak langsung mematikan kemampuan LLM menghasilkan SQL yang sesuai (Glass et al., 2025; Zhang et al., 2023).

> **Kesimpulan**: ***recall* yang cukup tinggi merupakan syarat minimum untuk *pipeline* Text-to-SQL, bukan *precision* yang sempurna.**

---

## 3.8.3.6 Analisis Ablation Few-Shot

**Tujuan**: menguji seberapa sensitif performa sistem Text-to-SQL terhadap penambahan jumlah contoh *in-context* (k).

### Konfigurasi Pengujian

| Aspek | Spesifikasi |
|---|---|
| Nilai k | k ∈ {0, 1, 3, 5} — mengacu pada standar evaluasi *few-shot* penelitian Text-to-SQL sebelumnya (D. Gao et al., 2023) |
| Data uji | **20% dari train set SPIDER** |
| Metode sampling | *Stratified sampling* berdasarkan tingkat kesulitan *query* untuk menjaga representativitas distribusi |
| Posisi dalam alur | Dilakukan **sebelum** evaluasi utama (3.8.3.1–3.8.3.5) |

Analisis ini bertujuan menentukan **satu nilai k yang digunakan secara seragam pada kedua *pipeline***, sehingga perbandingan antara *baseline* dan GraphRAG bersifat *apple-to-apple*.

### Fokus Analisis

Analisis difokuskan pada **pola sensitivitas performa** terhadap penambahan jumlah contoh, bukan pada pencarian titik optimal yang presisi. Shi et al. (2025) mencatat bahwa **kualitas contoh demonstrasi lebih menentukan dibandingkan jumlahnya**, sehingga pemahaman terhadap pola sensitivitas lebih informatif daripada sekadar mencari k optimal.

### Pola Interpretasi Perubahan EX

| Pola yang Diamati | Interpretasi | Dukungan Literatur |
|---|---|---|
| EX naik signifikan dari k=0 ke k=1, lalu relatif stabil | Sistem **sensitif hanya pada contoh pertama** — konsisten dengan pola *diminishing returns* | D. Gao et al. (2023) pada berbagai model di *benchmark* SPIDER |
| Perubahan EX konsisten kecil pada semua transisi | Sistem **tidak terlalu bergantung** pada jumlah contoh | — |
| EX **menurun** pada k yang lebih tinggi | Terjadi ***context overload*** | Tang et al. (2025): *over-prompting* melalui terlalu banyak *few-shot examples* tidak selalu berbanding lurus dengan peningkatan performa; Liu et al. (2024): model cenderung mengabaikan informasi di bagian tengah *prompt* yang panjang (*lost in the middle*) |

---

## Urutan Eksekusi Analisis

```
[ 3.8.3.6 ] Ablation Few-Shot  →  tentukan nilai k seragam
                 ↓
[ 3.8.3.1 ] Trade-off Token & Performa    (RM-1)
[ 3.8.3.2 ] Akurasi Generation SQL        (RM-2)
[ 3.8.3.3 ] Komponen SQL                  (RM-2)
[ 3.8.3.4 ] Robustness & Konsistensi      (RM-2)
                 ↓
[ 3.8.3.5 ] Bottleneck Retrieval vs Generation
            └─ dipicu bila EX & ESM sama-sama rendah
```
