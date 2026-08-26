"""
dim2_structure.py — Dimensi 2: Akurasi Struktur SQL

Referensi: EVALUATION_ANALYSIS_GUIDE.md Bagian 3, "Dimensi 2".
Menjawab RM2. Metrik: ESM, EX (dibaca BERPASANGAN, bukan terpisah).

Tabel interpretasi WAJIB (jangan buat sendiri):
| EX naik, ESM turun | EX naik, ESM naik | ESM naik, EX turun | keduanya rendah -> lanjut Dimensi 5 |

KEPUTUSAN (2026-08-26, lihat context/IMPLEMENTATION_DECISIONS.md poin 12):
guide menulis baris ke-4 sebagai "EX rendah, ESM rendah" -- beda kata dari
"naik"/"turun" di 3 baris lain (yang jelas directional: GraphRAG dibanding
Baseline), tanpa angka threshold ATAU acuan siapa yang "rendah". Peneliti
mengonfirmasi baris ke-4 dibaca sebagai directional juga -- EX TURUN dan ESM
TURUN (melengkapi kuadran ke-4 dari 4 kemungkinan arah delta, pasangan alami
baris "EX naik, ESM naik" = "kondisi paling ideal"). TIDAK memakai threshold
absolut baru. Label yang ditampilkan tetap teks literal guide ("EX rendah, ESM
rendah") untuk traceability ke tabel resmi, walau logika pemicunya delta
turun/turun.
"""

from typing import List, Optional

from src.metrics.esm_ex_cm import QueryEvalResult, aggregate_esm, aggregate_ex
from src.utils.raw_logs import DIFFICULTY_LEVELS, group_by_difficulty

# Teks interpretasi WAJIB persis dari guide Bagian 3 Dimensi 2 -- jangan diubah.
_INTERPRETATIONS = {
    "ex_naik_esm_turun": (
        "EX naik, ESM turun",
        "Penghematan token tidak merusak logika semantic meski struktur berbeda dari ground truth",
    ),
    "esm_naik_ex_turun": (
        "ESM naik, EX turun",
        "Struktur SQL ditiru tepat tapi gagal eksekusi -> indikasi kehilangan detail nilai literal",
    ),
    "keduanya_naik": (
        "EX naik, ESM naik",
        "Kondisi paling ideal -- konteks ringkas GraphRAG presisi",
    ),
    "keduanya_turun": (
        "EX rendah, ESM rendah",
        "Kegagalan menyeluruh -> lanjut ke Dimensi 5 (Bottleneck Analysis)",
    ),
}


def _to_query_eval_results(logs: List[dict]) -> List[QueryEvalResult]:
    return [
        QueryEvalResult(
            esm=e["esm_result"], ex=e["ex_result"],
            cm_per_clause=e["cm_per_clause"], difficulty=e["difficulty"],
        )
        for e in logs
    ]


def _esm_ex_stats(logs: List[dict]) -> dict:
    """{"esm": ..., "ex": ...} (0-100, None kalau logs kosong)."""
    if not logs:
        return {"esm": None, "ex": None}
    results = _to_query_eval_results(logs)
    return {"esm": aggregate_esm(results), "ex": aggregate_ex(results)}


def _condition_stats(logs: List[dict]) -> dict:
    groups = group_by_difficulty(logs)
    return {
        "overall": _esm_ex_stats(logs),
        "by_difficulty": {lvl: _esm_ex_stats(g) for lvl, g in groups.items()},
    }


def _classify(delta_ex: float, delta_esm: float) -> dict:
    """
    Klasifikasi 4-kuadran berdasarkan arah delta (GraphRAG - Baseline), lihat
    catatan KEPUTUSAN di docstring modul. delta persis 0 di salah satu metrik
    tidak cocok kuadran manapun -- dilaporkan apa adanya, bukan dipaksakan ke
    salah satu dari 4 kategori resmi (guide tidak mendefinisikan kasus ini).
    """
    if delta_ex > 0 and delta_esm < 0:
        key = "ex_naik_esm_turun"
    elif delta_esm > 0 and delta_ex < 0:
        key = "esm_naik_ex_turun"
    elif delta_ex > 0 and delta_esm > 0:
        key = "keduanya_naik"
    elif delta_ex < 0 and delta_esm < 0:
        key = "keduanya_turun"
    else:
        return {
            "combination": None,
            "interpretation": (
                f"Tidak terklasifikasi ke salah satu dari 4 kombinasi resmi guide "
                f"(delta_EX={delta_ex:.4f}, delta_ESM={delta_esm:.4f} -- salah satu "
                f"persis 0, guide tidak mendefinisikan kasus ini)."
            ),
            "escalate_to_dimension_5": False,
        }

    label, interpretation = _INTERPRETATIONS[key]
    return {
        "combination": label,
        "interpretation": interpretation,
        "escalate_to_dimension_5": key == "keduanya_turun",
    }


def run_dimension_2(baseline_logs: List[dict], graphrag_logs: List[dict]) -> dict:
    """
    Dimensi 2 -- Akurasi Struktur SQL (RM2). Guide Bagian 3, "Dimensi 2".
    ESM & EX SELALU dibaca berpasangan -- klasifikasi dihitung dari agregat
    OVERALL saja (bukan per difficulty), sama seperti TEP di Dimensi 1 hanya
    dari agregat overall. Angka per-difficulty tetap dilaporkan (guide langkah
    1-2), tapi tidak diklasifikasi ulang per level -- guide tidak meminta itu.
    """
    if not baseline_logs or not graphrag_logs:
        raise ValueError(
            "baseline_logs/graphrag_logs kosong -- Dimensi 2 butuh minimal satu "
            "query di kedua kondisi untuk klasifikasi ESM/EX berpasangan."
        )

    baseline = _condition_stats(baseline_logs)
    graphrag = _condition_stats(graphrag_logs)

    delta_ex = graphrag["overall"]["ex"] - baseline["overall"]["ex"]
    delta_esm = graphrag["overall"]["esm"] - baseline["overall"]["esm"]
    classification = _classify(delta_ex, delta_esm)
    classification["delta_ex"] = delta_ex
    classification["delta_esm"] = delta_esm

    return {
        "esm_ex": {"baseline": baseline, "graphrag": graphrag},
        "classification": classification,
    }


def print_dimension_2(result: dict) -> None:
    """Cetak hasil run_dimension_2(): tabel ESM/EX berdampingan dulu, baru klasifikasi."""
    stats = result["esm_ex"]
    cls = result["classification"]

    def _fmt(v: Optional[float]) -> str:
        return f"{v:.2f}" if v is not None else "N/A"

    print("=" * 78)
    print("ESM & EX (%) -- dibaca berpasangan")
    print("=" * 78)
    print(f"{'':10} {'Baseline ESM':>13} {'Baseline EX':>13} {'GraphRAG ESM':>13} {'GraphRAG EX':>13}")
    b_o, g_o = stats["baseline"]["overall"], stats["graphrag"]["overall"]
    print(f"{'overall':10} {_fmt(b_o['esm']):>13} {_fmt(b_o['ex']):>13} {_fmt(g_o['esm']):>13} {_fmt(g_o['ex']):>13}")
    for lvl in DIFFICULTY_LEVELS:
        b, g = stats["baseline"]["by_difficulty"][lvl], stats["graphrag"]["by_difficulty"][lvl]
        print(f"{lvl:10} {_fmt(b['esm']):>13} {_fmt(b['ex']):>13} {_fmt(g['esm']):>13} {_fmt(g['ex']):>13}")

    print("\n" + "=" * 78)
    print("Klasifikasi (overall, GraphRAG vs Baseline)")
    print("=" * 78)
    print(f"delta_EX = {cls['delta_ex']:.4f}   delta_ESM = {cls['delta_esm']:.4f}")
    print(f"Kombinasi: {cls['combination'] or '(tidak terklasifikasi)'}")
    print(f"Interpretasi: {cls['interpretation']}")
    if cls["escalate_to_dimension_5"]:
        print("-> Lanjutkan ke Dimensi 5 (Bottleneck Analysis).")
