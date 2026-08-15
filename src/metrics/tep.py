"""
tep.py — Token Elasticity of Performance

Referensi: EVALUATION_ANALYSIS_GUIDE.md Bagian 1.5, sub-bagian "TEP", dan
Bagian 3 "Dimensi 1" (tabel interpretasi WAJIB).

Formula (WAJIB, jangan diubah):
    TEP_G = (delta_EX_G / EX_B) / (delta_T_G / T_B)
    delta_EX_G = EX_G - EX_B
    delta_T_G  = T_G - T_B

PENTING: EX_B dan EX_G harus dalam SKALA YANG SAMA (persen semua, atau desimal
0-1 semua) — lihat guide 1.5 poin (e). Kalau tercampur, hasil TEP salah total.

EPSILON band untuk "TEP ≈ 0": guide tidak memberi angka eksplisit (beda dengan
threshold ±2% QVT atau 80%/65% F1/EX Dimensi 5 yang eksplisit "WAJIB, jangan
diubah"). Peneliti mengonfirmasi (2026-08-15) memakai ±0.05 sebagai default —
lihat context/IMPLEMENTATION_DECISIONS.md poin 9. Nilai ini dioper sebagai
parameter, bukan hardcode buta, supaya bisa direvisi tanpa mengubah signature.
"""

from dataclasses import dataclass

DEFAULT_EPSILON = 0.05  # lebar band "TEP ≈ 0" — lihat catatan EPSILON di atas


@dataclass
class TEPResult:
    delta_ex: float
    delta_t: float
    tep: float
    interpretation: str  # diisi otomatis berdasarkan tabel di guide Bagian 3 Dimensi 1


def compute_tep(
    ex_baseline: float,
    ex_graphrag: float,
    t_baseline: float,
    t_graphrag: float,
    epsilon: float = DEFAULT_EPSILON,
) -> TEPResult:
    """
    Hitung TEP dan interpretasinya.

    Melempar ValueError kalau ex_baseline atau t_baseline/delta_t adalah 0 —
    formula membagi dengan nilai-nilai itu, dan guide tidak mendefinisikan
    kebijakan untuk kasus itu (bukan kasus yang "diformulakan ulang", cuma
    dijaga eksplisit daripada diam-diam menghasilkan inf/nan/ZeroDivisionError
    yang membingungkan).
    """
    if ex_baseline == 0:
        raise ValueError("ex_baseline tidak boleh 0 — TEP membagi dengan EX_B (delta_EX_G / EX_B)")

    delta_ex = ex_graphrag - ex_baseline
    delta_t = t_graphrag - t_baseline

    if delta_t == 0:
        raise ValueError(
            "delta_t (T_G - T_B) = 0 — Token Consumption baseline dan GraphRAG identik, "
            "TEP tidak terdefinisi (pembagian dengan nol)"
        )
    if t_baseline == 0:
        raise ValueError("t_baseline tidak boleh 0 — TEP membagi dengan T_B (delta_T_G / T_B)")

    tep = (delta_ex / ex_baseline) / (delta_t / t_baseline)

    # Tabel interpretasi WAJIB dari guide Bagian 3 Dimensi 1 — dipakai persis,
    # jangan ubah teksnya.
    if -epsilon <= tep <= epsilon:
        interpretation = "Efisiensi token tercapai tanpa trade-off penurunan EX yang signifikan"
    elif tep < 0:
        interpretation = "Skenario ideal: GraphRAG meningkatkan EX dan menurunkan Token Consumption dibanding baseline"
    else:
        interpretation = "Ada trade-off: penurunan token disertai penurunan EX"

    return TEPResult(delta_ex=delta_ex, delta_t=delta_t, tep=tep, interpretation=interpretation)
