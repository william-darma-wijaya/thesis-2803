"""
dim1_efficiency.py — Dimensi 1: Efisiensi Token & Performa

Referensi: EVALUATION_ANALYSIS_GUIDE.md Bagian 3, "Dimensi 1".
Menjawab RM1.

INPUT: raw_logs baseline & graphrag yang SUDAH berisi field token_input,
token_output, ex_result per query (hasil dari src.metrics.token_consumption
dan src.metrics.esm_ex_cm).

Urutan output WAJIB sesuai guide (jangan diacak):
1. Tabel Token Consumption (mean, median, per difficulty) — baseline vs graphrag
2. Tabel EX (agregat + per difficulty) — baseline vs graphrag
3. Nilai TEP tunggal + interpretasi

RESOLVED: μ=3 dari guide menang, `PipelineConfig.token_output_weight` di
src/core/config.py sudah diupdate ke 3.0 (sebelumnya 1.0 — lihat catatan di
src/metrics/token_consumption.py). Saat implementasi, pass `cfg.token_output_weight`
ke compute_token_consumption(), jangan hardcode ulang di sini.
"""

from typing import List, Dict


def run_dimension_1(baseline_logs: List[dict], graphrag_logs: List[dict]) -> dict:
    """
    TODO:
    1. Hitung token consumption T per query untuk baseline & graphrag
       (pakai src.metrics.token_consumption.compute_token_consumption,
        kalau belum dihitung saat logging)
    2. Agregasi mean & median, overall dan per difficulty level
       -> simpan sebagai table_token_consumption
    3. Agregasi EX, overall dan per difficulty level
       -> simpan sebagai table_ex
    4. Ambil EX_B, EX_G, T_B, T_G (rata-rata overall) -> panggil
       src.metrics.tep.compute_tep()
    5. Return dict berisi ketiga output di atas + interpretasi TEP,
       dalam urutan yang sama dengan guide (token dulu, EX, baru TEP)
    """
    raise NotImplementedError
