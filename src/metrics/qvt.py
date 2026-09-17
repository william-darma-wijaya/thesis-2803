"""
qvt.py — Query Variance Testing

Referensi: EVALUATION_ANALYSIS_GUIDE.md Bagian 1.6.

PRASYARAT DATA: dataset variasi NL question per gold SQL, taruh di
data/qvt_variations/. KOREKSI (2026-09-07, lihat context/IMPLEMENTATION_DECISIONS.md
poin 21): klaim lama di sini ("tidak datang otomatis dari dev set biasa") SALAH —
dev.json SPIDER punya ~470 SQL dengan 2 paraphrase NL alami (ditulis manusia saat
anotasi), mencakup 90.9% dev set. `src/experiments/build_qvt_variations.py`
membangun data/qvt_variations/{baseline,graphrag}_qvt.json dari situ + raw_logs
yang sudah ada (data/raw_logs/*.json) — tidak perlu generate parafrase baru.

Format data yang diharapkan (per gold SQL):
{
  "query_id": "...",          # canonical id group (query_id terkecil anggotanya)
  "gold_sql": "...",
  "variations": [
     {"query_id": "...", "nl_question": "...", "predicted_sql": "...", "is_correct": 0 | 1}
  ]
}

"query_id" DI DALAM tiap variasi tidak dipakai modul ini sama sekali — itu untuk
silang-ESM di src/dimensions/dim4_robustness.py, yang perlu melihat esm_result
tiap variasi di raw_logs (lihat IMPLEMENTATION_DECISIONS.md poin 23).

KEPUTUSAN (2026-08-15, lihat context/IMPLEMENTATION_DECISIONS.md poin 10):
"is_correct" per variasi diukur pakai **EX** (Execution Accuracy), bukan ESM.
Ini keputusan whoever MENGISI data/qvt_variations/*.json (mis. experiment
script yang menjalankan pipeline atas tiap variasi lalu memanggil
src.metrics.esm_ex_cm.evaluate_single_query() dan mengambil `.ex`) — qvt.py
sendiri hanya MENGONSUMSI field "is_correct" yang sudah jadi, tidak menghitung
EX/ESM sendiri. Dicatat di sini supaya kontrak data ini tidak diinterpretasi
ulang jadi ESM oleh sesi berikutnya.
"""

import statistics
from typing import List
from dataclasses import dataclass


@dataclass
class QVTQueryResult:
    query_id: str
    included: bool  # False kalau dibuang karena semua variasi gagal (filter wajib)
    score: float  # proporsi variasi benar, hanya valid kalau included=True


def compute_qvt_per_query(query_id: str, variations: List[dict]) -> QVTQueryResult:
    """
    Hitung skor level-1 untuk satu gold SQL.

    Referensi guide 1.6 poin (c) langkah 2-3. `variations` tidak boleh kosong
    (m_i = 0 berarti tidak ada variasi disiapkan untuk query ini sama sekali —
    itu bug di data prep, bukan kasus "semua gagal" yang filter wajib maksud).
    """
    if not variations:
        raise ValueError(
            f"query_id={query_id!r}: variations kosong — tidak ada variasi NL "
            "question untuk query ini (beda dengan 'semua variasi gagal', "
            "yang seharusnya tetap punya entri dengan is_correct=0)"
        )

    correct_count = sum(1 for v in variations if v["is_correct"] == 1)

    if correct_count == 0:
        # FILTER WAJIB — guide 1.6 poin (e): query yang semua variasinya gagal
        # dibuang dari perhitungan QVT sama sekali, bukan dihitung skor=0.
        return QVTQueryResult(query_id=query_id, included=False, score=0.0)

    score = correct_count / len(variations)
    return QVTQueryResult(query_id=query_id, included=True, score=score)


def aggregate_qvt(per_query_results: List[QVTQueryResult]) -> float:
    """
    QVT = mean(score) HANYA untuk query dengan included=True.
    Jangan ikutkan query yang included=False dalam pembagian (guide 1.6 poin (c)
    langkah 4 — M di formula adalah jumlah query yang LOLOS filter, bukan total
    semua query di dataset).
    """
    included_scores = [r.score for r in per_query_results if r.included]
    if not included_scores:
        raise ValueError(
            "Semua query dibuang oleh filter wajib (tidak ada satupun query "
            "dengan minimal 1 variasi benar) — QVT tidak terdefinisi untuk "
            "populasi ini. Ini kondisi anomali, bukan skor 0 — laporkan ke "
            "peneliti sebelum melanjutkan, jangan diam-diam return 0.0."
        )
    return statistics.mean(included_scores)
