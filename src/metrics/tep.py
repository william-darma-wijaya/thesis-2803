"""
tep.py — Token Elasticity of Performance

Referensi: EVALUATION_ANALYSIS_GUIDE.md Bagian 1.5, sub-bagian "TEP".

Formula (WAJIB, jangan diubah):
    TEP_G = (delta_EX_G / EX_B) / (delta_T_G / T_B)
    delta_EX_G = EX_G - EX_B
    delta_T_G  = T_G - T_B

PENTING: EX_B dan EX_G harus dalam SKALA YANG SAMA (persen semua, atau desimal
0-1 semua) — lihat guide 1.5 poin (e). Kalau tercampur, hasil TEP salah total.
"""

from dataclasses import dataclass


@dataclass
class TEPResult:
    delta_ex: float
    delta_t: float
    tep: float
    interpretation: str  # diisi otomatis berdasarkan tabel di guide Bagian 3 Dimensi 1


def compute_tep(ex_baseline: float, ex_graphrag: float, t_baseline: float, t_graphrag: float) -> TEPResult:
    """
    Hitung TEP dan interpretasinya.

    TODO:
    - delta_ex = ex_graphrag - ex_baseline
    - delta_t = t_graphrag - t_baseline
    - tep = (delta_ex / ex_baseline) / (delta_t / t_baseline)
    - Interpretasi (WAJIB pakai tabel ini persis, lihat guide Bagian 3 Dimensi 1):
        tep < 0        -> "Skenario ideal: EX naik, Token Consumption turun"
        tep ~ 0 (misal -0.05 <= tep <= 0.05, threshold ini BELUM ditentukan
                  eksplisit di proposal - konfirmasi ke peneliti sebelum dipakai)
                       -> "Efisiensi token tercapai tanpa trade-off EX signifikan"
        tep > 0        -> "Trade-off: penurunan token disertai penurunan EX"
    - Return TEPResult
    """
    raise NotImplementedError
