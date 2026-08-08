"""
dim3_component.py — Dimensi 3: Analisis Komponen SQL

Referensi: EVALUATION_ANALYSIS_GUIDE.md Bagian 3, "Dimensi 3".
Metrik: CM per klausa (baseline vs graphrag).
"""

from typing import List, Dict


def run_dimension_3(baseline_logs: List[dict], graphrag_logs: List[dict]) -> dict:
    """
    TODO:
    1. Agregasi CM per klausa untuk baseline & graphrag
       (pakai src.metrics.esm_ex_cm.aggregate_cm())
    2. Tampilkan sebagai tabel perbandingan per klausa (bukan satu angka)
    3. Identifikasi klausa dengan penurunan skor terbesar baseline -> graphrag
    4. Interpretasi kualitatif ringan (ikuti pola di guide, jangan generalisasi
       berlebihan):
       - penurunan besar di WHERE -> dampak pemangkasan konteks di filtering
       - penurunan besar di FROM/JOIN -> kehilangan info relasi antar tabel
    5. Return dict berisi tabel per klausa + klausa dengan penurunan terbesar
       (untuk dipakai lagi sebagai input diagnostik Dimensi 5)
    """
    raise NotImplementedError
