"""
dim3_component.py — Dimensi 3: Analisis Komponen SQL

Referensi: EVALUATION_ANALYSIS_GUIDE.md Bagian 3, "Dimensi 3". Menjawab RM2.
Metrik: CM per klausa (baseline vs graphrag) -- guide TIDAK minta breakdown
per difficulty di sini (beda dari Dimensi 1/2 yang eksplisit minta itu), cuma
tabel flat per klausa.
"""

from typing import Dict, List, Optional

from src.metrics.esm_ex_cm import CLAUSES, QueryEvalResult, aggregate_cm

# Guide cuma kasih 2 contoh interpretasi kualitatif (teks WAJIB persis) --
# klausa lain SENGAJA tidak dikasih interpretasi buatan sendiri, guide
# eksplisit warning "jangan generalisasi berlebihan" (Bagian 3 Dimensi 3
# poin 4).
_QUALITATIVE_NOTES = {
    "where": "Dampak pemangkasan konteks terasa di logika filtering.",
    "from": "GraphRAG kehilangan informasi relasi antar tabel.",
}


def _to_query_eval_results(logs: List[dict]) -> List[QueryEvalResult]:
    return [
        QueryEvalResult(
            esm=e["esm_result"], ex=e["ex_result"],
            cm_per_clause=e["cm_per_clause"], difficulty=e["difficulty"],
        )
        for e in logs
    ]


def run_dimension_3(baseline_logs: List[dict], graphrag_logs: List[dict]) -> dict:
    """
    Dimensi 3 -- Analisis Komponen SQL (RM2). Guide Bagian 3, "Dimensi 3".
    Flat table per klausa -- guide TIDAK minta breakdown per difficulty di sini.
    """
    if not baseline_logs or not graphrag_logs:
        raise ValueError(
            "baseline_logs/graphrag_logs kosong -- Dimensi 3 butuh minimal satu "
            "query di kedua kondisi untuk agregasi CM per klausa."
        )

    baseline_cm = aggregate_cm(_to_query_eval_results(baseline_logs))
    graphrag_cm = aggregate_cm(_to_query_eval_results(graphrag_logs))

    deltas: Dict[str, Optional[float]] = {}
    for clause in CLAUSES:
        b, g = baseline_cm[clause], graphrag_cm[clause]
        deltas[clause] = (g - b) if (b is not None and g is not None) else None

    computable = {c: d for c, d in deltas.items() if d is not None}
    if computable:
        clause = min(computable, key=computable.get)  # most negative = biggest drop
        delta = computable[clause]
        is_decrease = delta < 0
        if not is_decrease:
            interpretation = (
                f"Tidak ada klausa yang benar-benar menurun dari Baseline ke GraphRAG "
                f"-- delta terkecil ada di '{clause}' ({delta:+.2f} poin), tapi itu "
                f"tetap kenaikan/tidak berubah, bukan penurunan."
            )
        else:
            note = _QUALITATIVE_NOTES.get(clause)
            interpretation = f"Penurunan terbesar ada di '{clause}' ({delta:+.2f} poin)."
            interpretation += f" {note}" if note else (
                " Guide tidak menyediakan interpretasi kualitatif khusus untuk "
                "klausa ini di luar WHERE/FROM."
            )
    else:
        clause, delta, is_decrease = None, None, False
        interpretation = "Tidak ada klausa yang bisa dibandingkan (semua delta None)."

    return {
        "cm_per_clause": {"baseline": baseline_cm, "graphrag": graphrag_cm, "delta": deltas},
        "biggest_drop": {
            "clause": clause, "delta": delta,
            "is_actual_decrease": is_decrease, "interpretation": interpretation,
        },
    }


def print_dimension_3(result: dict) -> None:
    """Cetak tabel per-klausa dulu, baru klausa dengan penurunan terbesar + interpretasi."""
    cm = result["cm_per_clause"]
    drop = result["biggest_drop"]

    def _fmt(v: Optional[float]) -> str:
        return f"{v:.2f}" if v is not None else "N/A"

    print("=" * 70)
    print("Component Match (CM) per klausa (%)")
    print("=" * 70)
    print(f"{'clause':12} {'Baseline':>10} {'GraphRAG':>10} {'Delta':>10}")
    for clause in CLAUSES:
        b, g, d = cm["baseline"][clause], cm["graphrag"][clause], cm["delta"][clause]
        d_str = f"{d:+.2f}" if d is not None else "N/A"
        print(f"{clause:12} {_fmt(b):>10} {_fmt(g):>10} {d_str:>10}")

    print("\n" + "=" * 70)
    print("Klausa dengan penurunan terbesar")
    print("=" * 70)
    print(f"Klausa: {drop['clause'] or '(tidak ada)'}")
    print(f"Interpretasi: {drop['interpretation']}")
