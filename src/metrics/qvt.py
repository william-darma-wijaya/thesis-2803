"""
qvt.py — Query Variance Testing

Referensi: EVALUATION_ANALYSIS_GUIDE.md Bagian 1.6.

PRASYARAT DATA: dataset variasi NL question per gold SQL, taruh di
data/qvt_variations/. Ini TIDAK datang otomatis dari SPIDER dev set biasa,
harus disiapkan/digenerate terpisah.

Format data yang diharapkan (per gold SQL):
{
  "query_id": "...",
  "gold_sql": "...",
  "variations": [
     {"nl_question": "...", "predicted_sql": "...", "is_correct": 0 | 1}
  ]
}
"is_correct" diukur pakai EX ATAU ESM - tentukan salah satu dan pakai KONSISTEN
untuk semua perhitungan QVT (lihat guide 1.6 poin (e)).
"""

from typing import List, Dict
from dataclasses import dataclass


@dataclass
class QVTQueryResult:
    query_id: str
    included: bool  # False kalau dibuang karena semua variasi gagal (filter wajib)
    score: float  # proporsi variasi benar, hanya valid kalau included=True


def compute_qvt_per_query(query_id: str, variations: List[dict]) -> QVTQueryResult:
    """
    Hitung skor level-1 untuk satu gold SQL.

    Referensi guide 1.6 poin (c) langkah 2-3.

    TODO:
    - correct_count = jumlah variations dengan is_correct == 1
    - Kalau correct_count == 0 -> return QVTQueryResult(query_id, included=False, score=0.0)
      (FILTER WAJIB - lihat guide 1.6 poin (e), kesalahan paling umum kalau dilewatkan)
    - Kalau tidak -> score = correct_count / len(variations)
    - return QVTQueryResult(query_id, included=True, score=score)
    """
    raise NotImplementedError


def aggregate_qvt(per_query_results: List[QVTQueryResult]) -> float:
    """
    QVT = mean(score) HANYA untuk query dengan included=True.
    Jangan ikutkan query yang included=False dalam pembagian (guide 1.6 poin (c) langkah 4).

    TODO:
    - included_scores = [r.score for r in per_query_results if r.included]
    - return mean(included_scores) jika included_scores tidak kosong, else raise/warning
      (dataset kosong berarti semua query gagal di semua variasi - kondisi anomali,
      laporkan ke peneliti, jangan diam-diam return 0)
    """
    raise NotImplementedError
