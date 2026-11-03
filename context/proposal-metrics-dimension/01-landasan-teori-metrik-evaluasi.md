# Landasan Teori: Evaluasi Sistem Text-to-SQL

> Sumber: Proposal Skripsi — BAB 2, Subbab 2.9

---

## 2.9.1 Schema Linking Accuracy (SLA)

SLA merupakan metrik evaluasi yang berfokus pada tahap *retrieval* dengan *schema linking*, yaitu proses pemetaan entitas berdasarkan pertanyaan *natural language* ke elemen *database schema* yang relevan, baik berupa tabel maupun kolom. SLA menilai seberapa tepat sistem mengidentifikasi bagian *schema* yang benar-benar dibutuhkan untuk menjawab pertanyaan.

Cao et al. (2024) menunjukkan bahwa *schema linking*, yang sering dianggap sebagai komponen kecil dalam model Text-to-SQL, sebenarnya memegang peranan penting — jika dilakukan dengan baik, model dengan struktur sederhana pun dapat menghasilkan performa yang baik pada dataset SPIDER.

### Granularitas Evaluasi

Evaluasi SLA dilakukan pada dua tingkat:
- **Tingkat tabel**
- **Tingkat kolom**

Pada masing-masing tingkat, prediksi elemen *schema* yang dihasilkan oleh *pipeline* dibandingkan dengan *schema ground truth* yang digunakan pada *query* referensi.

### Rumus

**Precision** — proporsi elemen *schema* yang diproduksi model yang benar-benar relevan:

$$\text{precision} = \frac{TP}{TP + FP} \tag{5}$$

**Recall** — proporsi elemen *schema* relevan pada *ground truth* yang berhasil ditemukan model:

$$\text{recall} = \frac{TP}{TP + FN} \tag{6}$$

**F1-Score** — rata-rata harmonik dari *precision* dan *recall*:

$$F1 = \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}} \tag{7}$$

Dengan:
- **TP** (*true positive*): elemen *schema* yang diprediksi benar dan relevan
- **FP** (*false positive*): elemen yang diprediksi relevan namun sebenarnya tidak digunakan
- **FN** (*false negative*): elemen yang gagal diprediksi

### Trade-off Precision vs Recall

Dalam konteks *schema linking*, kegagalan menemukan elemen *schema* yang relevan (*recall* rendah) umumnya memberikan dampak yang lebih fatal dibandingkan kelebihan elemen yang tidak relevan (*precision* rendah) (Glass et al., 2025). Hal ini disebabkan elemen yang hilang membuat *query* mustahil dihasilkan secara benar, sementara elemen berlebih masih dapat ditoleransi pada tahap *generation*.

Tantangan utama SLA adalah adanya *trade-off* antara *precision* dan *recall*. Karena ambiguitas pada *natural language* dan kemiripan *semantic* antar tabel kandidat, peningkatan *recall* biasanya disertai meningkatnya redundansi yang tercermin lewat nilai *precision* yang menurun. Pengambilan tabel atau kolom yang berlebihan dapat memunculkan *join* antar tabel yang redundan dan berpotensi menurunkan kualitas hasil *generation*.

Sasaran ideal *schema linking*: menemukan seluruh elemen *schema* yang relevan sekaligus menghindari pengambilan elemen yang tidak diperlukan — menjadikan SLA sebagai alat diagnostik untuk menelusuri sumber kesalahan *pipeline* Text-to-SQL pada tingkat *retrieval* (Glass et al., 2025; Nahid et al., 2025).

---

## 2.9.2 Exact Set Match (ESM)

ESM merupakan pengembangan dari metode *Exact Match* (EM) konvensional, menawarkan fleksibilitas lebih tinggi dalam mengevaluasi *query* SQL hasil prediksi model.

| Aspek | EM | ESM |
|---|---|---|
| Metode perbandingan | *String-to-string comparison* berbasis karakter | Kesesuaian struktur komponen *query* |
| Sensitivitas urutan | Sensitif | Tidak sensitif terhadap urutan penulisan non-substantif |
| Sensitivitas nilai literal | Sensitif | Tidak sensitif |

### Mekanisme

1. *Query* SQL hasil prediksi dan *query ground truth* diproses melalui tahap *parsing* ke dalam struktur data yang terstandarisasi/dinormalisasi (Scholak et al., 2021).
2. Dalam bentuk kanonik ini, sistem memvalidasi apakah setiap klausa SQL (SELECT, WHERE, GROUP BY, dll.) telah sesuai dengan referensi kunci jawaban, tanpa terikat pada urutan penulisan teks yang tidak substantif (L. Shi et al., 2025).

### Kelebihan

Kemampuan mengabaikan perbedaan minor yang tidak memengaruhi logika *query*. ESM tidak sensitif terhadap nilai literal — model tidak langsung dianggap salah hanya karena perbedaan penulisan angka atau *string*, asalkan struktur logika pembentuk *query* sudah tepat. Hal ini memberikan penilaian yang lebih adil terhadap kemampuan model dalam memahami *database schema*.

---

## 2.9.3 Execution Accuracy (EX)

EX merupakan metode evaluasi yang mengukur kesamaan hasil eksekusi *query* SQL dari model dengan hasil eksekusi *ground truth* SQL pada *database* (Mohammadjafari et al., 2025). *Query* prediksi dianggap benar apabila menghasilkan keluaran yang sama dengan *ground truth*, meskipun struktur *query* berbeda.

Berbeda dengan EM maupun CM yang berfokus pada kesamaan struktur *query*, EX berorientasi pada kesamaan hasil akhir dari proses eksekusi.

### Rumus

Hasil evaluasi setiap *query* direpresentasikan secara biner — benar (1) atau salah (0):

$$EX = \frac{\text{Jumlah Hasil Eksekusi Benar}}{\text{Total Seluruh Query Pengujian}} \times 100\% \tag{8}$$

### Karakteristik dan Kelemahan

- **False positive**: terjadi ketika *query* prediksi dan *ground truth* menghasilkan keluaran yang sama (misalnya nilai NULL atau tabel kosong) meskipun secara *semantic* kedua *query* memiliki makna berbeda (Yu et al., 2019).
- **Sensitif terhadap nilai literal**: berbeda dengan ESM dan CM, EX pada evaluasi SPIDER harus mempertimbangkan nilai literal/angka pada pertanyaan pengguna. Perbedaan nilai literal dapat memengaruhi hasil eksekusi secara langsung walaupun struktur *query* sudah sesuai.

---

## 2.9.4 Component Match (CM)

CM melakukan evaluasi terhadap setiap komponen penyusun *query* SQL secara terpisah, sehingga evaluasi menjadi lebih fleksibel dan rinci dibandingkan ESM yang menilai *query* secara keseluruhan (Yu et al., 2019).

Menurut Mohammadjafari et al. (2025), pada CM kesamaan komponen SQL yang digunakan lebih penting dibandingkan urutan penulisannya.

### Komponen yang Dievaluasi

| No | Komponen | Deskripsi |
|---|---|---|
| a | Pemilihan tabel | Ketepatan tabel *database* yang digunakan pada *query* |
| b | Pemilihan kolom | Kesesuaian kolom yang dipilih atau digunakan pada *query* |
| c | Pemilihan operasi | Penggunaan operasi logika maupun klausa SQL (SELECT, FROM, WHERE, JOIN, GROUP BY) |

### Mekanisme

CM memecah *query* SQL hasil prediksi dan *ground truth query* menjadi beberapa komponen penyusun, kemudian membandingkan masing-masing komponen secara terpisah. Jika elemen-elemen penting telah sesuai dengan *ground truth*, *query* dapat dianggap benar meskipun urutan penulisannya berbeda atau struktur *query* belum optimal.

Pendekatan ini memungkinkan analisis kesalahan yang lebih mendalam karena setiap bagian *query* dievaluasi secara individual, sehingga membantu mengidentifikasi kelemahan spesifik pada sistem Text-to-SQL termasuk kesalahan prediksi maupun halusinasi LLM (Gurawa & Dharmik, 2025).

> **Temuan empiris**: Yu et al. (2019) menunjukkan bahwa berdasarkan hasil evaluasi pada *test set* SPIDER, klausa yang paling sering mengalami kesalahan prediksi adalah **WHERE** dan **GROUP BY**.

---

## 2.9.5 Token Consumption dan Token Elasticity of Performance (TEP)

Menurut Zhu et al. (2025), di tengah banyaknya sistem Text-to-SQL canggih yang meraih skor tinggi, masih terdapat permasalahan borosnya token. Sistem-sistem ini sering menggunakan *pipeline* rumit untuk memproses semua jenis pertanyaan secara merata, padahal banyak pertanyaan yang sebenarnya dapat diselesaikan dengan mudah.

### Token Consumption

Dihitung dari total jumlah token input (teks yang dimasukkan ke LLM) dan token output (teks hasil keluaran LLM). Token output dikalikan terlebih dahulu dengan faktor tertentu karena pada banyak API LLM, token output lebih mahal dibandingkan token input.

$$T = T_{in} + \mu \cdot T_{out} \tag{9}$$

Dengan:
- $T_{in}$ = jumlah token yang dimasukkan
- $T_{out}$ = jumlah token yang dikeluarkan
- $\mu$ = faktor pengali/bobot untuk token yang dikeluarkan

### Token Elasticity of Performance (TEP)

Zhu et al. (2025) mengambil konsep "elastisitas" dari ilmu ekonomi — perubahan performa diukur relatif terhadap jumlah kenaikan token yang digunakan. TEP membandingkan sistem Text-to-SQL yang lebih canggih dengan metode *baseline* yang lebih sederhana:

$$TEP_G = \frac{\Delta EX_G / EX_B}{\Delta T_G / T_B} \tag{10}$$

Dengan:
- $\Delta EX_G = EX_G - EX_B$ (perubahan performa pada metode yang diuji)
- $EX_B$ = performa dari metode *baseline*
- $\Delta T_G = T_G - T_B$ (perubahan konsumsi token pada metode yang diuji)
- $T_B$ = konsumsi token dari metode *baseline*

TEP berguna untuk mengukur keseimbangan antara performa metode Text-to-SQL dan biaya yang dikeluarkan. Semakin tinggi skor TEP, semakin efisien biaya yang harus dikeluarkan untuk metode Text-to-SQL yang diuji.

---

## 2.9.6 Query Variance Testing (QVT)

QVT merupakan metrik evaluasi yang pertama kali diperkenalkan pada *framework* NL2SQL360 untuk mengukur tingkat stabilitas dan *robustness* sistem Text-to-SQL ketika menerima berbagai variasi pertanyaan *natural language* yang memiliki makna dan target *query* SQL yang sama (B. Li et al., 2024).

Berbeda dengan ESM dan EX yang berfokus pada ketepatan hasil *query*, QVT berfokus pada kemampuan model mempertahankan konsistensi prediksi ketika terjadi perubahan gaya bahasa pada pertanyaan pengguna. Variasi tersebut dapat berupa:
- Penggunaan sinonim
- Perubahan struktur kalimat
- Gaya bahasa formal maupun informal
- Perbedaan panjang kalimat

### Notasi

Setiap *ground truth query* SQL memiliki beberapa pasangan pertanyaan *natural language* yang berbeda:

$$(N_1, Q_1), (N_2, Q_1), \ldots, (N_m, Q_1) \tag{11}$$

Dengan:
- $N$ = pertanyaan *natural language*
- $Q_i$ = *query* SQL ke-i

### Kriteria Inklusi

Li et al. (2024) menetapkan bahwa tidak semua *query* SQL dapat dimasukkan ke dalam *test set* QVT. Sebuah *query* SQL hanya dapat digunakan apabila **terdapat minimal satu pasangan *natural language* yang berhasil diprediksi dengan benar** oleh model. Apabila seluruh variasi pertanyaan gagal diprediksi secara akurat, *query* tersebut tidak dimasukkan ke dalam evaluasi QVT.

### Rumus

$$QVT = \frac{1}{M} \sum_{i=1}^{M} \left( \frac{\sum_{j=1}^{m_i} \mathbf{1}(F(N_{ij}) = Q_i)}{m_i} \right) \tag{12}$$

Dengan:
- $M$ = jumlah total *query* SQL
- $m_i$ = jumlah variasi *natural language* untuk SQL ke-i
- $N_{ij}$ = variasi pertanyaan ke-j
- $Q_i$ = *ground truth query* SQL
- $F(N_{ij})$ = *query* SQL hasil prediksi model
- $\mathbf{1}(\cdot)$ = fungsi indikator (1 jika prediksi sesuai *ground truth*, 0 jika tidak)

### Dua Lapisan Evaluasi

1. **Tingkat *query* SQL** — menghitung persentase variasi pertanyaan *natural language* yang berhasil diprediksi dengan benar untuk setiap *query* SQL.
2. **Tingkat keseluruhan dataset** — menghitung rata-rata seluruh skor yang diperoleh pada lapisan pertama.

### Relevansi

Pada penggunaan nyata, pengguna tidak selalu memberikan pertanyaan dalam format yang konsisten. Variasi penulisan, penggunaan sinonim, kesalahan pengetikan, maupun perubahan gaya bahasa dapat memengaruhi performa model.

> **Temuan empiris**: Tidak terdapat pemenang mutlak antara pendekatan PLM dan LLM dalam evaluasi QVT. Namun, model yang telah melalui proses *fine-tuning* umumnya memiliki nilai QVT lebih tinggi karena lebih selaras dengan distribusi data yang spesifik terhadap *task* Text-to-SQL (B. Li et al., 2024).

---

## Ringkasan Metrik

| Metrik | Fokus Evaluasi | Sensitif Nilai Literal | Tahapan Pipeline |
|---|---|---|---|
| **SLA** | Ketepatan pemetaan entitas ke elemen *schema* | — | *Retrieval* |
| **ESM** | Kesesuaian struktur komponen *query* | Tidak | *Generation* |
| **EX** | Kesamaan hasil eksekusi *query* | Ya | *Generation* |
| **CM** | Akurasi per-komponen SQL | Tidak | *Generation* |
| **Token Consumption / TEP** | Efisiensi penggunaan token vs performa | — | End-to-end |
| **QVT** | Konsistensi output terhadap variasi input | — | End-to-end |
