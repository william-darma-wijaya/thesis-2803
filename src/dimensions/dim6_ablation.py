"""
dim6_ablation.py — Dimensi 6: Ablation Few-Shot

Referensi: EVALUATION_ANALYSIS_GUIDE.md Bagian 3, "Dimensi 6".

STATUS: BELUM PRIORITAS saat ini (lihat guide Bagian 0.1 dan catatan dependency
di sini) — masih tahap trial-error pembuatan metrics & kode per dimensi.
Stub ini disiapkan supaya strukturnya konsisten dengan dimensi lain, tapi
implementasi bisa ditunda.

Kalau nanti dikerjakan:
- Pakai 20% train set SPIDER (BUKAN dev set), stratified sampling per difficulty
- k in {0, 1, 3, 5}
- Hitung EX untuk tiap k
- Baca pola transisi EX vs k (bukan cari titik optimal presisi matematis)
- Tentukan k final untuk dipakai seragam di Dimensi 1-5
"""

from typing import Dict, List


def run_dimension_6(ex_per_k: Dict[int, float]) -> dict:
    """
    ex_per_k: {0: EX_value, 1: EX_value, 3: EX_value, 5: EX_value}

    TODO (nanti, belum prioritas):
    1. Baca pola transisi antar nilai k berurutan
    2. Kalau EX naik signifikan k=0->k=1 lalu stabil -> diminishing returns
    3. Kalau EX turun di k besar -> indikasi context overload
    4. Tentukan k_final berdasarkan pola (bukan cuma ambil EX tertinggi mentah-mentah)
    5. Return dict berisi tabel EX vs k + k_final + alasan
    """
    raise NotImplementedError
