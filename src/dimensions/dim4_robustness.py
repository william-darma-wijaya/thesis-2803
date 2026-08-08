"""
dim4_robustness.py — Dimensi 4: Robustness & Konsistensi Query

Referensi: EVALUATION_ANALYSIS_GUIDE.md Bagian 3, "Dimensi 4".
Metrik: QVT, ESM (berpasangan). Threshold delta QVT: ±2% (WAJIB, jangan diubah).

PRASYARAT: data/qvt_variations/ harus sudah terisi (lihat src.metrics.qvt).
"""

from typing import List, Dict


def run_dimension_4(baseline_qvt_data: List[dict], graphrag_qvt_data: List[dict]) -> dict:
    """
    TODO:
    1. Hitung QVT_B dan QVT_G (pakai src.metrics.qvt.aggregate_qvt(),
       ingat filter wajib di compute_qvt_per_query())
    2. delta_qvt = QVT_G - QVT_B
    3. Interpretasi pakai threshold ±2% (WAJIB, dari guide Bagian 3 Dimensi 4):
       - delta_qvt >= +2%          -> "GraphRAG meningkatkan konsistensi secara meaningful"
       - -2% <= delta_qvt < +2%    -> "Konsistensi relatif stabil"
       - delta_qvt < -2%           -> "GraphRAG menurunkan konsistensi, indikasi pemangkasan berlebih"
    4. Silangkan dengan ESM: apakah query yang tidak konsisten (score QVT rendah)
       juga bermasalah di ESM -> analisis tambahan
    5. Return dict berisi QVT_B, QVT_G, delta_qvt, interpretasi, dan silang-ESM
    """
    raise NotImplementedError
