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

- [ ] **Tahap 1 — `sweep.py`** (tanpa LLM, cepat): pilih `top_k_tables_pct`,
  `top_k_columns_pct`, `baseline_top_k_tables_pct` by SLA (saat ini
  placeholder `0.6`, TBD).
  ```bash
  python src/experiments/sweep.py --sample 0.2
  ```
  Baca juga `avg_cols_sent` vs `avg_cols_candidate` di `sweep_results.csv`:
  kalau semua persentase kolom memberi hasil sama, threshold 0.35 yang
  memotong — pertimbangkan `--thresholds 0.0 0.3 0.35 0.5`.
- [ ] **Tahap 2 — `ablation.py`** (Run 1 — Baseline, Run 2 — GraphRAG, GPU):
  pilih `few_shot_k` by EX (satu k untuk kedua pipeline; script mencetak
  rekomendasi "shared" setelah dua mode selesai). Juga menghasilkan
  `outputs/tables/ablation_ex_per_k_graphrag.json` untuk Dimensi 6.
  ```bash
  python src/experiments/ablation.py --mode baseline --sample 1.0
  python src/experiments/ablation.py --mode graphrag --sample 1.0
  ```
- [ ] **Tahap 3 — perbandingan model** (GPU): semua model di
  `llm_model_frame` dengan config tahap 1+2, lalu tentukan `llm_model` final.
  ```bash
  python src/experiments/pipeline.py --baseline --skip-sweep --models all --sample 1.0
  ```
  Catatan: 14B 4-bit butuh ~9 GB VRAM + BGE-M3; di T4 16 GB masih muat tapi
  jalankan per model (`--models <nama>`) kalau sesi Kaggle mepet waktu.

Catatan: tahap 2 & 3 butuh GPU (torch/LLM tidak jalan di environment dev
lokal) — jalankan di Kaggle.

## 3. Keputusan yang perlu didiskusikan

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
