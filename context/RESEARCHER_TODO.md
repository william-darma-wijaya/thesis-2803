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

- [ ] **`data/qvt_variations/*.json`** — dataset variasi NL question per gold SQL
  untuk metrik QVT (Dimensi 4).
  **Keputusan (2026-08-15): grouping dilakukan sendiri oleh peneliti**, bukan
  digenerate otomatis oleh Claude Code — peneliti yang menentukan cara
  mengelompokkan variasi pertanyaan ke `query_id` gold SQL yang sama.
  Format kontrak data (lihat `src/metrics/qvt.py`):
  ```json
  {
    "query_id": "dev_0001",
    "gold_sql": "...",
    "variations": [
      {"nl_question": "...", "predicted_sql": "...", "is_correct": 0 | 1}
    ]
  }
  ```
  - `query_id` harus cocok dengan konvensi `dev_{i+1:04d}` yang dipakai
    `pipeline.py` (lihat `src/experiments/pipeline.py:605`), supaya nanti bisa
    disilangkan dengan hasil ESM di raw_logs (Dimensi 4 butuh QVT × ESM
    berpasangan).
  - `predicted_sql` dan `is_correct` (pakai **EX**, sudah diputuskan — lihat
    `IMPLEMENTATION_DECISIONS.md` poin 10) baru bisa diisi setelah tiap variasi
    dijalankan lewat pipeline + dievaluasi. Kalau peneliti ingin bagian
    run-pipeline-dan-hitung-EX ini dibantu lewat script (bukan cuma
    grouping-nya), bilang saja — itu bagian terpisah yang belum ada
    generator/runner-nya sama sekali di `src/experiments/`.

## 2. Eksperimen yang harus dijalankan (menentukan parameter TBD)

- [ ] **`sweep.py`** — jalankan untuk menentukan nilai final `top_k_tables`,
  `top_k_columns`, `semantic_similarity_threshold` di `config.py` (saat ini
  masih nilai default placeholder, ditandai TBD).
  ```bash
  python src/experiments/sweep.py --sample 0.2
  ```
- [ ] **`ablation.py`** (Run 1 — Baseline, Run 2 — GraphRAG) — belum pernah
  dijalankan sama sekali (bukan cuma "belum final", tapi belum ada output di
  `outputs/tables/ablation_results.csv` sama sekali per saat catatan ini
  ditulis). Perlu dijalankan untuk menentukan `few_shot_k` final dan supaya
  Dimensi 1 & 6 punya data nyata untuk dianalisis.
  ```bash
  python src/experiments/ablation.py --mode baseline --k-values 0 1 3 5 --sample 1.0
  python src/experiments/ablation.py --mode graphrag --k-values 0 1 3 5 --sample 1.0
  ```
  Catatan: butuh GPU (torch tidak tersedia di environment dev lokal) — jalankan
  di Kaggle.

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
`top_k_tables`/`top_k_columns`/`few_shot_k`, dll) — file ini melengkapi, bukan
menggantikan, daftar itu. Kalau ada TBD baru yang murni soal kode/implementasi
(bukan sesuatu yang perlu disiapkan/diputuskan peneliti di luar coding), catat
di `CLAUDE.md` atau `IMPLEMENTATION_DECISIONS.md`, bukan di sini.
