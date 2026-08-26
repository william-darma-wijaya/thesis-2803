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
src/metrics/token_consumption.py). `run_dimension_1()` menerima `cfg` penuh
(bukan cuma `mu: float`) supaya konsisten dengan cara pipeline.py/ablation.py/
generation.py semuanya mengoper `PipelineConfig` utuh, bukan field yang
diekstrak satu-satu.

Modul ini murni orkestrasi/agregasi -- tidak menghitung apapun yang belum ada
implementasinya di src/metrics/. `aggregate_token_consumption()` sengaja tidak
tahu soal difficulty (lihat docstring-nya sendiri), jadi grouping-per-difficulty
dilakukan lewat src.utils.raw_logs.group_by_difficulty() (diextract dari sini
setelah dim2_structure.py butuh pola yang sama persis -- lihat
context/IMPLEMENTATION_DECISIONS.md poin 12).
"""

from dataclasses import asdict
from typing import List, Optional

from src.core.config import PipelineConfig
from src.metrics.esm_ex_cm import QueryEvalResult, aggregate_ex
from src.metrics.token_consumption import compute_token_consumption, aggregate_token_consumption
from src.metrics.tep import compute_tep
from src.utils.raw_logs import DIFFICULTY_LEVELS, group_by_difficulty


def _token_consumption_stats(logs: List[dict], mu: float) -> Optional[dict]:
    """{"mean": ..., "median": ...} atau None kalau logs kosong (bukan crash/0)."""
    if not logs:
        return None
    t_values = [compute_token_consumption(e["token_input"], e["token_output"], mu) for e in logs]
    return aggregate_token_consumption(t_values)


def _ex_percent(logs: List[dict]) -> Optional[float]:
    """
    EX agregat (0-100) atau None kalau logs kosong. Rekonstruksi QueryEvalResult
    dari raw_logs (field-nya sudah sama persis) supaya reuse aggregate_ex()'s
    formula asli, bukan menghitung ulang sum(...)/len(...)*100 di sini.
    """
    if not logs:
        return None
    results = [
        QueryEvalResult(
            esm=e["esm_result"], ex=e["ex_result"],
            cm_per_clause=e["cm_per_clause"], difficulty=e["difficulty"],
        )
        for e in logs
    ]
    return aggregate_ex(results)


def _condition_stats(logs: List[dict], mu: float) -> dict:
    groups = group_by_difficulty(logs)
    return {
        "token_consumption": {
            "overall": _token_consumption_stats(logs, mu),
            "by_difficulty": {lvl: _token_consumption_stats(g, mu) for lvl, g in groups.items()},
        },
        "ex": {
            "overall": _ex_percent(logs),
            "by_difficulty": {lvl: _ex_percent(g) for lvl, g in groups.items()},
        },
    }


def run_dimension_1(baseline_logs: List[dict], graphrag_logs: List[dict], cfg: PipelineConfig) -> dict:
    """
    Dimensi 1 -- Efisiensi Token & Performa (RM1). Guide Bagian 3, "Dimensi 1".
    Urutan WAJIB: Token Consumption -> EX -> TEP (urutan key dict di bawah
    mengikuti ini persis, dan itu juga urutan print_dimension_1() menampilkannya).
    """
    if not baseline_logs or not graphrag_logs:
        raise ValueError(
            "baseline_logs/graphrag_logs kosong -- Dimensi 1 butuh minimal satu "
            "query di kedua kondisi untuk menghitung TEP (tidak bisa dari data kosong)."
        )

    mu = cfg.token_output_weight
    baseline = _condition_stats(baseline_logs, mu)
    graphrag = _condition_stats(graphrag_logs, mu)

    tep_result = compute_tep(
        ex_baseline=baseline["ex"]["overall"],
        ex_graphrag=graphrag["ex"]["overall"],
        t_baseline=baseline["token_consumption"]["overall"]["mean"],
        t_graphrag=graphrag["token_consumption"]["overall"]["mean"],
    )

    return {
        "token_consumption": {"baseline": baseline["token_consumption"], "graphrag": graphrag["token_consumption"]},
        "ex": {"baseline": baseline["ex"], "graphrag": graphrag["ex"]},
        "tep": asdict(tep_result),
    }


def print_dimension_1(result: dict) -> None:
    """Cetak hasil run_dimension_1() sesuai urutan WAJIB guide: token dulu, EX, baru TEP+interpretasi."""
    tc = result["token_consumption"]
    ex = result["ex"]
    tep = result["tep"]

    def _fmt(stats: Optional[dict], key: str) -> str:
        return f"{stats[key]:.1f}" if stats else "N/A"

    print("=" * 70)
    print("Token Consumption (T = T_in + mu*T_out)")
    print("=" * 70)
    print(f"{'':10} {'Baseline mean':>15} {'Baseline med':>15} {'GraphRAG mean':>15} {'GraphRAG med':>15}")
    b_o, g_o = tc["baseline"]["overall"], tc["graphrag"]["overall"]
    print(f"{'overall':10} {_fmt(b_o,'mean'):>15} {_fmt(b_o,'median'):>15} {_fmt(g_o,'mean'):>15} {_fmt(g_o,'median'):>15}")
    for lvl in DIFFICULTY_LEVELS:
        b, g = tc["baseline"]["by_difficulty"][lvl], tc["graphrag"]["by_difficulty"][lvl]
        print(f"{lvl:10} {_fmt(b,'mean'):>15} {_fmt(b,'median'):>15} {_fmt(g,'mean'):>15} {_fmt(g,'median'):>15}")

    print("\n" + "=" * 70)
    print("EX (%)")
    print("=" * 70)
    print(f"{'':10} {'Baseline':>12} {'GraphRAG':>12}")
    b_ex, g_ex = ex["baseline"]["overall"], ex["graphrag"]["overall"]
    print(f"{'overall':10} {b_ex:>12.2f} {g_ex:>12.2f}")
    for lvl in DIFFICULTY_LEVELS:
        b, g = ex["baseline"]["by_difficulty"][lvl], ex["graphrag"]["by_difficulty"][lvl]
        b_s = f"{b:.2f}" if b is not None else "N/A"
        g_s = f"{g:.2f}" if g is not None else "N/A"
        print(f"{lvl:10} {b_s:>12} {g_s:>12}")

    print("\n" + "=" * 70)
    print("TEP (Token Elasticity of Performance)")
    print("=" * 70)
    print(f"delta_EX = {tep['delta_ex']:.4f}   delta_T = {tep['delta_t']:.4f}   TEP = {tep['tep']:.4f}")
    print(f"Interpretasi: {tep['interpretation']}")
