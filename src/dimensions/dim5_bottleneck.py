"""
dim5_bottleneck.py — Dimensi 5: Bottleneck Retrieval vs Generation

Referensi: EVALUATION_ANALYSIS_GUIDE.md Bagian 3, "Dimensi 5".
Metrik: F1-Score SLA (table-level DAN column-level, dua diagnosis independen),
EX (satu nilai, dipakai di kedua diagnosis -- EX tidak punya level).

CATATAN TRIGGER: guide menyebut dimensi ini "hanya dijalankan kalau Dimensi 2
menunjukkan kondisi 'EX rendah, ESM rendah'" -- itu keputusan EKSTERNAL (siapa
yang manggil run_dimension_5(), kapan) bukan sesuatu yang dicek fungsi ini
sendiri. run_dimension_5() murni diagnostik untuk SATU kondisi yang dioper.

CATATAN DUA LEVEL: guide tidak menjelaskan cara resolusi kalau diagnosis
table-level dan column-level berbeda (mis. table-level "sehat", column-level
"retrieval bottleneck") -- implementasi ini TIDAK mengarang aturan resolusi,
cuma menampilkan keduanya + flag `levels_agree` kalau beda.

Threshold WAJIB (jangan diubah, sama untuk kedua level):
- F1-Score SLA: 80%
- EX: 65%
"""

from typing import List, Optional

from src.metrics.esm_ex_cm import QueryEvalResult, aggregate_ex
from src.metrics.sla import SLAResult, compute_sla, aggregate_sla

F1_THRESHOLD = 0.80
EX_THRESHOLD = 0.65

_DIAGNOSIS = {
    (True, True): "Pipeline sehat, retrieval & generation optimal bersamaan",
    (True, False): "Kegagalan di tahap generation -- schema relevan berhasil didapat, LLM gagal menyusun SQL",
    (False, False): "Kegagalan di tahap retrieval -- schema gagal diambil sejak awal",
    (False, True): "Jarang terjadi -- kemungkinan query sederhana atau LLM berhasil menalar dari schema parsial",
}


def _diagnose_level(sla: SLAResult, ex_score: float) -> dict:
    f1_ok = sla.f1 >= F1_THRESHOLD
    ex_ok = ex_score >= EX_THRESHOLD
    diagnosis = _DIAGNOSIS[(f1_ok, ex_ok)]

    precision_recall_note: Optional[str] = None
    if not f1_ok:
        if sla.recall < sla.precision:
            precision_recall_note = (
                f"Recall ({sla.recall:.2%}) lebih rendah dari precision ({sla.precision:.2%}) "
                f"-- lebih fatal, elemen schema yang dibutuhkan hilang dari hasil retrieval."
            )
        elif sla.precision < sla.recall:
            precision_recall_note = (
                f"Precision ({sla.precision:.2%}) lebih rendah dari recall ({sla.recall:.2%}) "
                f"-- masih bisa ditoleransi, ini soal noise/JOIN redundan, bukan elemen hilang."
            )
        else:
            precision_recall_note = (
                f"Precision dan recall sama ({sla.precision:.2%}) -- tidak ada sisi yang lebih dominan."
            )

    return {
        "f1_score_sla": sla.f1, "precision": sla.precision, "recall": sla.recall,
        "f1_meets_threshold": f1_ok, "ex_meets_threshold": ex_ok,
        "diagnosis": diagnosis, "precision_recall_note": precision_recall_note,
    }


def run_dimension_5(logs: List[dict], condition_label: str = "GraphRAG") -> dict:
    if not logs:
        raise ValueError(
            "logs kosong -- Dimensi 5 butuh minimal satu query untuk diagnosis bottleneck."
        )

    sla_table = aggregate_sla([
        compute_sla(set(e["gold_schema"]), set(e["predicted_schema"]), level="table")
        for e in logs
    ])
    sla_column = aggregate_sla([
        compute_sla(set(e["gold_schema"]), set(e["predicted_schema"]), level="column")
        for e in logs
    ])

    eval_results = [
        QueryEvalResult(esm=e["esm_result"], ex=e["ex_result"],
                         cm_per_clause=e["cm_per_clause"], difficulty=e["difficulty"])
        for e in logs
    ]
    ex_score = aggregate_ex(eval_results) / 100  # 0-100 -> 0-1, match threshold scale

    table_diag = _diagnose_level(sla_table, ex_score)
    column_diag = _diagnose_level(sla_column, ex_score)

    return {
        "condition_label": condition_label,
        "ex_score": ex_score,
        "table_level": table_diag,
        "column_level": column_diag,
        "levels_agree": table_diag["diagnosis"] == column_diag["diagnosis"],
    }


def print_dimension_5(result: dict) -> None:
    print("=" * 70)
    print(f"Dimensi 5 -- Bottleneck Analysis ({result['condition_label']})")
    print("=" * 70)
    print(f"EX: {result['ex_score']:.2%}  (threshold: {EX_THRESHOLD:.0%})")
    print()

    for level_name, key in [("Table-level", "table_level"), ("Column-level", "column_level")]:
        d = result[key]
        print(f"-- {level_name} SLA --")
        print(f"  F1: {d['f1_score_sla']:.2%}  (threshold: {F1_THRESHOLD:.0%})   "
              f"Precision: {d['precision']:.2%}   Recall: {d['recall']:.2%}")
        print(f"  Diagnosis: {d['diagnosis']}")
        if d["precision_recall_note"]:
            print(f"  Breakdown: {d['precision_recall_note']}")
        print()

    if not result["levels_agree"]:
        print("[!] Table-level dan column-level menghasilkan diagnosis BERBEDA -- "
              "guide tidak mendefinisikan cara resolusi, baca keduanya secara terpisah.")
