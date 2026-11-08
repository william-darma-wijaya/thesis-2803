# Researcher To-Do — Hal yang Harus Disiapkan Peneliti

> **Tujuan file ini:** daftar hal-hal yang jadi tanggung jawab **peneliti** (bukan
> sesuatu yang bisa diselesaikan Claude Code sendiri lewat coding) — data yang
> harus disiapkan/dikumpulkan manual, eksperimen yang harus dijalankan (butuh
> GPU/Kaggle), dan keputusan yang perlu didiskusikan dulu sebelum implementasi
> lanjut. **Setiap kali peneliti bertanya "hal apa yang belum kita siapkan?" atau
> sejenisnya, baca file ini dulu** sebagai acuan tambahan, di luar TBD list yang
> sudah ada di `CLAUDE.md`.
>
> Checklist di bawah dikelompokkan per kategori. Centang manual (`- [x]`) kalau
> sudah selesai, dan pindahkan catatan konteks/keputusan terkait ke
> `IMPLEMENTATION_DECISIONS.md` supaya tidak dobel.

---

## 1. Data yang harus disiapkan

- [x] **`data/qvt_variations/*.json`** — SELESAI SECARA KODE 2026-09-07 (lihat
  `IMPLEMENTATION_DECISIONS.md` poin 21) — **membalikkan keputusan 2026-08-15**
  di bawah ini. Ternyata `dev.json` SPIDER sendiri sudah punya ~470 SQL dengan
  2 paraphrase NL question ditulis manusia (proses anotasi), mencakup 90.9%
  dev set — tidak perlu grouping manual ataupun generate parafrase baru.
  `src/experiments/build_qvt_variations.py` mengerjakan groupingnya otomatis
  dari `data/spider_data/dev.json` + join ke `data/raw_logs/*.json` yang sudah
  ada (tidak perlu run pipeline terpisah untuk QVT).
  **Yang masih jadi blocker:** hanya `data/raw_logs/*.json` itu sendiri harus
  berasal dari `pipeline.py --baseline --sample 1.0` di Kaggle (item #2 di
  bawah) — begitu itu ada, tinggal jalankan:
  ```bash
  python src/experiments/build_qvt_variations.py
  ```
  <details><summary>Keputusan lama 2026-08-15 (sudah tidak berlaku, dibiarkan untuk riwayat)</summary>

  Format kontrak data (masih berlaku, dipakai `build_qvt_variations.py` —
  ditambah field `query_id` per variasi sejak 2026-09-14, poin 23):
  ```json
  {
    "query_id": "dev_0001",
    "gold_sql": "...",
    "variations": [
      {"query_id": "dev_0001", "nl_question": "...", "predicted_sql": "...", "is_correct": 0 | 1}
    ]
  }
  ```
  `is_correct` pakai **EX** (`IMPLEMENTATION_DECISIONS.md` poin 10).
  `query_id` di dalam variasi dipakai Dimensi 4 untuk silang-ESM ke raw_logs.
  </details>

## 2. Eksperimen yang harus dijalankan (menentukan parameter TBD)

Ablation bertahap (`IMPLEMENTATION_DECISIONS.md` poin 28) — urutannya wajib,
tiap tahap memakai pemenang tahap sebelumnya. Setelah tiap tahap, **tulis
pemenangnya ke `config.py`** (atau oper lewat flag `--tables-pct` dkk).

- [ ] **Tahap 1 — threshold tabel & kolom** (SLA; EX zero-shot butuh GPU, poin 35, `RUN_EX = False` untuk tanpa LLM): sejak poin 30/32
  tidak lagi top-k%. GraphRAG dan Baseline sama-sama memilih tabel by
  `table_similarity_threshold` (default 0.60), kolom GraphRAG by
  `semantic_similarity_threshold` (0.70). Dituning di
  `notebooks/graphrag_sla_trial.ipynb` (jalankan ulang G1-G7 dengan
  `max_ngram=2`). Baseline belum punya trial sendiri: ukur SLA tabel Baseline
  di beberapa threshold. `sweep.py` hanya referensi (`best_config()` = `{}`).
  **Update 2026-11-03 (poin 33):** notebook trial tidak lagi memilih threshold
  otomatis. Baca rekap bagian 9 + tabel GraphRAG vs Baseline di T yang sama
  (bagian 9b: reduksi kolom, token schema, token prompt, SLA), lalu **isi
  `BEST_T` / `BEST_K` sendiri** dan tulis ke `config.py`. Baseline memakai T yang
  sama (tidak perlu trial Baseline terpisah).
- [x] **Open item E** — DIPUTUSKAN 2026-11-03 (poin 34): ablation di dev set
  (contoh few-shot tetap dari train), sampel 20% stratified per difficulty
  (`SAMPLE_RATIO = 0.20` di notebook trial, `--sample 0.2` di `ablation.py` /
  `ablation_fewshot.ipynb`). Keterbatasan "dipilih dan dilaporkan di dev" WAJIB
  ditulis di metodologi (teks poin 34). Hasil trial/ablation lama (sampel acak)
  harus dijalankan ulang.
- [ ] **Cari nilai α** dari pricing Qwen (atau model terdekat) → isi
  `token_output_weight` di `config.py` (satu-satunya tempat), plus sumber/tanggal
  untuk disclaimer TEP.
- [x] ~~**Sebelum tahap 2 — A/B seleksi few-shot**~~ — TIDAK dijalankan (poin 33):
  `skeleton` dipakai langsung. Catatan lama dibiarkan untuk riwayat.
  (GPU, `IMPLEMENTATION_DECISIONS.md`
  poin 29, branch `few-shot/skeleton`): `question` vs `skeleton` di k=3, full
  dev set, dengan top-k pemenang tahap 1. Hasil cek tanpa LLM sudah positif
  (`outputs/tables/skeleton_retrieval_check.json`); yang ini memutuskan dengan EX.
  ```bash
  python src/experiments/ablation.py --mode graphrag --sample 1.0 --k-values 3 --few-shot-selection question
  python src/experiments/ablation.py --mode graphrag --sample 1.0 --k-values 3 --few-shot-selection skeleton
  python src/experiments/ablation.py --mode baseline --sample 1.0 --k-values 3 --few-shot-selection question
  python src/experiments/ablation.py --mode baseline --sample 1.0 --k-values 3 --few-shot-selection skeleton
  python src/skeleton/compare_ab.py
  ```
  Pemenangnya → `few_shot_selection` di `config.py`, lalu tahap 2 dijalankan
  dengan selection itu (run k=3 di atas bisa dipakai ulang).
- [ ] **Tahap 2 — `ablation.py`** (Run 1 — Baseline, Run 2 — GraphRAG, GPU):
  pilih `few_shot_k` by EX (satu k untuk kedua pipeline; script mencetak
  rekomendasi "shared" setelah dua mode selesai). Juga menghasilkan
  `outputs/tables/ablation_ex_per_k_graphrag.json` untuk Dimensi 6.
  ```bash
  python src/experiments/ablation.py --mode baseline --sample 0.2
  python src/experiments/ablation.py --mode graphrag --sample 0.2
  ```
- [ ] **Tahap 3 — perbandingan model** (GPU, dijalankan TERAKHIR): Qwen2.5-Coder
  3B / 7B / 14B (`llm_model_frame`, 1.5B dibuang — poin 33) dengan best config
  tahap 1+2, kedua pipeline.
  ```bash
  python src/experiments/pipeline.py --baseline --skip-sweep --models all --sample 1.0
  ```
  Catatan: 14B 4-bit butuh ~9 GB VRAM + BGE-M3; di T4 16 GB masih muat tapi
  jalankan per model (`--models <nama>`) kalau sesi Kaggle mepet waktu.

Catatan: tahap 2 & 3 butuh GPU (torch/LLM tidak jalan di environment dev
lokal) — jalankan di Kaggle.

## 3. Keputusan yang perlu didiskusikan

- [ ] **Baseline DAIL-SQL** (dikerjakan nanti, `IMPLEMENTATION_DECISIONS.md` poin 37 dan open item F). GraphRAG 1-Stage dan 2-Stage adalah metode peneliti; baseline skripsi adalah DAIL-SQL. Perlu diputuskan cara menjalankannya, syarat kesebandingan, metrik yang dibandingkan, dan pembanding TEP.

- [x] **Definisi final Baseline** — DIPUTUSKAN 2026-09-06: **table-level
  retrieval** (`src/retrieval/baseline.py`) adalah baseline skripsi, FINAL.
  Full-schema bypass (`--full-schema`) tetap ada sebagai mode ablation, bukan
  baseline. Dicatat di `IMPLEMENTATION_DECISIONS.md` poin 18 + `CLAUDE.md`.
- [x] **JOIN-keyword di parser resmi SPIDER** — SELESAI 2026-08-25
  (`IMPLEMENTATION_DECISIONS.md` poin 11, `src/utils/sql_normalize.py`):
  `INNER`/`CROSS`/`LEFT`/`RIGHT`/`FULL JOIN` semua dinormalisasi ke `JOIN`
  sebelum parsing. Jalur in-process (`esm_ex_cm.py`) fix penuh untuk semua
  join type termasuk EX; jalur subprocess `--etype exec` cuma `INNER`/`CROSS`
  (residual limitation, lihat poin 11). Diagnostic tambahan (2026-09-06,
  `IMPLEMENTATION_DECISIONS.md` poin 20): blok CONFOUND DIAGNOSTICS di
  `comparison_report.txt` menghitung prediksi outer-JOIN yang `ESM=1` tapi
  `EX=0` — kandidat di mana normalisasi parsing menutupi beda semantik nyata.
  **Yang mungkin masih perlu keputusan peneliti** (setelah run Kaggle asli):
  apakah angka norm-masked itu cukup besar untuk perlu mitigasi tambahan.

## 4. Validasi yang belum dilakukan

- [ ] **Raw-logs pipeline (`run_comparison()`) belum pernah dijalankan
  terhadap Spider dev set asli / di Kaggle.** Baru divalidasi lewat sanity
  test terhadap database SQLite sintetis (ESM/EX/CM/gold_schema semua benar
  di situ), tapi belum pernah menghasilkan `data/raw_logs/*.json` yang nyata
  dari dev set sungguhan.

---

## Status lain yang masih TBD (lihat `CLAUDE.md` untuk daftar lengkap)

`CLAUDE.md` root punya bagian "Hal yang Masih TBD" sendiri (nilai final
`top_k_*_pct`/`few_shot_k`/`llm_model`, dll) — file ini melengkapi, bukan
menggantikan, daftar itu. Kalau ada TBD baru yang murni soal kode/implementasi
(bukan sesuatu yang perlu disiapkan/diputuskan peneliti di luar coding), catat
di `CLAUDE.md` atau `IMPLEMENTATION_DECISIONS.md`, bukan di sini.
