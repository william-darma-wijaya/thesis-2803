"""
dim5_bottleneck.py — Dimensi 5: Bottleneck Retrieval vs Generation

Referensi: EVALUATION_ANALYSIS_GUIDE.md Bagian 3, "Dimensi 5".
Metrik: F1-Score SLA, EX (berpasangan). HANYA dijalankan kalau Dimensi 2
menunjukkan kondisi "EX rendah, ESM rendah" (kegagalan menyeluruh).

Threshold WAJIB (jangan diubah):
- F1-Score SLA: 80%
- EX: 65%
"""

from typing import Dict

F1_THRESHOLD = 0.80
EX_THRESHOLD = 0.65


def run_dimension_5(f1_score_sla: float, ex_score: float, precision: float, recall: float) -> dict:
    """
    TODO:
    1. Bandingkan f1_score_sla vs F1_THRESHOLD dan ex_score vs EX_THRESHOLD
    2. Klasifikasikan pakai tabel diagnostik (WAJIB, dari guide Bagian 3 Dimensi 5):
       - F1 >= 80% & EX >= 65%  -> "Pipeline sehat, retrieval & generation optimal"
       - F1 >= 80% & EX < 65%   -> "Kegagalan di tahap generation"
       - F1 < 80%  & EX < 65%   -> "Kegagalan di tahap retrieval"
       - F1 < 80%  & EX >= 65%  -> "Jarang terjadi, kemungkinan query sederhana"
    3. Kalau F1 rendah, pecah lagi jadi precision vs recall:
       - recall rendah -> lebih fatal (elemen schema hilang)
       - precision belum sempurna -> masih bisa ditoleransi (noise)
    4. Return dict berisi klasifikasi + breakdown precision/recall + catatan
    """
    raise NotImplementedError
