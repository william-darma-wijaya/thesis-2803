"""
dim4_robustness.py — Dimensi 4: Robustness & Konsistensi Query

Referensi: EVALUATION_ANALYSIS_GUIDE.md Bagian 3, "Dimensi 4".
Menjawab RM2 -- apakah sistem tetap stabil saat pertanyaan diparafrase?
Metrik: QVT, ESM (berpasangan). Threshold delta QVT: +/-2% (WAJIB, jangan diubah).

PRASYARAT: data/qvt_variations/{baseline,graphrag}_qvt.json harus sudah terisi.
Diisi otomatis oleh `src/experiments/build_qvt_variations.py` dari paraphrase
ALAMI di dev.json + raw_logs (lihat IMPLEMENTATION_DECISIONS.md poin 21) --
bukan digenerate manual.

SKALA: aggregate_qvt() mengembalikan 0-1, jadi threshold "+/-2%" dari guide
dipakai sebagai 0.02 pada skala itu (2 poin persen), bukan 2.0.

KEPUTUSAN (2026-09-14, lihat IMPLEMENTATION_DECISIONS.md poin 23):

1. Signature diperluas dengan `baseline_logs` / `graphrag_logs` (opsional).
   Guide langkah 6 minta silang QVT x ESM, tapi data QVT sengaja TIDAK membawa
   ESM -- field "is_correct" di situ adalah EX (poin 10). Jadi ESM harus datang
   dari raw_logs terpisah. Kalau logs tidak dioper, `esm_cross` = None dan
   print_dimension_4() menyatakannya eksplisit -- langkah guide dilewati secara
   TERLIHAT, bukan diam-diam.

2. Populasi silang-ESM = query yang LOLOS filter wajib QVT (included=True).
   Query yang SEMUA variasinya gagal itu "konsisten salah", bukan "tidak
   konsisten" -- dan memang dibuang dari QVT by definition (guide 1.6 poin e).
   Jumlah yang dibuang tetap dilaporkan supaya tidak tersembunyi.

3. "Tidak konsisten" := qvt_score < 1.0 (tidak semua variasi benar).
   "Bermasalah di ESM" := tidak semua variasi query itu punya esm_result == 1.
   Dua-duanya pakai notion "all-or-nothing" yang sama supaya sumbu silangnya
   apple-to-apple. Kode tidak meng-hardcode m_i = 2, walau dengan paraphrase
   alami dev.json m_i memang selalu tepat 2 (poin 21).
"""

from typing import List, Optional

from src.metrics.qvt import QVTQueryResult, compute_qvt_per_query, aggregate_qvt

# WAJIB dari guide Bagian 3 Dimensi 4 -- jangan diubah. Alasan guide: variasi
# antar metode SOTA di literatur hanya 3-4 poin persen, dan model di riset ini
# tidak fine-tuned sehingga lebih rentan noise.
DELTA_QVT_THRESHOLD = 0.02

# Teks interpretasi persis dari tabel guide Bagian 3 Dimensi 4.
_INTERPRETATIONS = {
    "meningkat": (
        "delta_QVT >= +2%",
        "GraphRAG meningkatkan konsistensi query secara meaningful",
    ),
    "stabil": (
        "-2% <= delta_QVT < +2%",
        "Konsistensi relatif stabil, pemangkasan token tidak merusak robustness",
    ),
    "menurun": (
        "delta_QVT < -2%",
        "GraphRAG menurunkan konsistensi -- indikasi pemangkasan berlebih",
    ),
}


# ---------------------------------------------------------------------------
# Langkah 2-3 guide: filter wajib + agregasi QVT per kondisi
# ---------------------------------------------------------------------------

def _qvt_stats(qvt_data: List[dict], condition_label: str) -> dict:
    """
    Hitung QVT satu kondisi + bawa serta hasil per-query-nya (dipakai lagi oleh
    silang-ESM, supaya filter wajib tidak dihitung dua kali dengan cara berbeda).
    """
    if not qvt_data:
        raise ValueError(
            f"qvt_data kosong untuk kondisi {condition_label!r} -- Dimensi 4 butuh "
            "minimal satu gold SQL dengan variasi NL question. Jalankan "
            "`python src/experiments/build_qvt_variations.py` dulu."
        )

    per_query: List[QVTQueryResult] = [
        compute_qvt_per_query(e["query_id"], e["variations"]) for e in qvt_data
    ]
    included = [r for r in per_query if r.included]

    try:
        qvt = aggregate_qvt(per_query)
    except ValueError as exc:
        # aggregate_qvt() sengaja melempar (bukan return 0.0) kalau filter wajib
        # membuang SEMUA query. Re-raise dengan label kondisi supaya jelas sisi
        # mana yang anomali.
        raise ValueError(f"Kondisi {condition_label!r}: {exc}") from exc

    return {
        "qvt": qvt,
        "m_included": len(included),
        "n_excluded": len(per_query) - len(included),
        "n_total": len(per_query),
        "per_query": per_query,
        "inconsistent_query_ids": sorted(r.query_id for r in included if r.score < 1.0),
    }


# ---------------------------------------------------------------------------
# Langkah 5 guide: interpretasi delta terhadap threshold +/-2%
# ---------------------------------------------------------------------------

def _interpret_delta(delta_qvt: float) -> dict:
    if delta_qvt >= DELTA_QVT_THRESHOLD:
        key = "meningkat"
    elif delta_qvt < -DELTA_QVT_THRESHOLD:
        key = "menurun"
    else:
        key = "stabil"

    band, interpretation = _INTERPRETATIONS[key]
    return {"band": band, "category": key, "interpretation": interpretation}


# ---------------------------------------------------------------------------
# Langkah 6 guide: silang QVT x ESM
# ---------------------------------------------------------------------------

def _esm_cross(qvt_data: List[dict], stats: dict, logs: List[dict], condition_label: str) -> dict:
    """
    Kontingensi 2x2 (konsisten vs tidak) x (ESM bersih vs bermasalah), dihitung
    HANYA atas query yang lolos filter wajib QVT (lihat KEPUTUSAN 2 di docstring
    modul). ESM diambil per-variasi dari raw_logs lewat `query_id` di tiap
    variasi -- bukan cuma dari canonical query_id group-nya.
    """
    esm_by_id = {e["query_id"]: e["esm_result"] for e in logs}
    by_query_id = {e["query_id"]: e for e in qvt_data}

    counts = {
        ("konsisten", "esm_ok"): 0, ("konsisten", "esm_problem"): 0,
        ("tidak_konsisten", "esm_ok"): 0, ("tidak_konsisten", "esm_problem"): 0,
    }
    n_missing = 0

    for r in stats["per_query"]:
        if not r.included:
            continue
        variations = by_query_id[r.query_id]["variations"]

        if any("query_id" not in v for v in variations):
            raise ValueError(
                f"Kondisi {condition_label!r}, query_id={r.query_id!r}: variasi tidak "
                "punya field 'query_id', jadi ESM per-variasi tidak bisa dilihat di "
                "raw_logs. File qvt_variations ini dibangun oleh versi LAMA "
                "build_qvt_variations.py -- bangun ulang dengan versi sekarang "
                "(lihat IMPLEMENTATION_DECISIONS.md poin 23)."
            )

        esm_values = [esm_by_id.get(v["query_id"]) for v in variations]
        if any(v is None for v in esm_values):
            n_missing += 1
            continue

        consistency = "konsisten" if r.score >= 1.0 else "tidak_konsisten"
        esm_state = "esm_ok" if all(v == 1 for v in esm_values) else "esm_problem"
        counts[(consistency, esm_state)] += 1

    n_consistent = counts[("konsisten", "esm_ok")] + counts[("konsisten", "esm_problem")]
    n_inconsistent = (
        counts[("tidak_konsisten", "esm_ok")] + counts[("tidak_konsisten", "esm_problem")]
    )

    def _rate(problem: int, total: int) -> Optional[float]:
        return problem / total if total else None

    return {
        "n_consistent": n_consistent,
        "n_inconsistent": n_inconsistent,
        "consistent_esm_ok": counts[("konsisten", "esm_ok")],
        "consistent_esm_problem": counts[("konsisten", "esm_problem")],
        "inconsistent_esm_ok": counts[("tidak_konsisten", "esm_ok")],
        "inconsistent_esm_problem": counts[("tidak_konsisten", "esm_problem")],
        "esm_problem_rate_consistent": _rate(counts[("konsisten", "esm_problem")], n_consistent),
        "esm_problem_rate_inconsistent": _rate(
            counts[("tidak_konsisten", "esm_problem")], n_inconsistent
        ),
        "n_missing_in_logs": n_missing,
    }


def _cross_note(cross: dict) -> str:
    """Bacaan kualitatif dari dua rate di kontingensi -- tanpa threshold baru."""
    rate_i = cross["esm_problem_rate_inconsistent"]
    rate_c = cross["esm_problem_rate_consistent"]
    if rate_i is None or rate_c is None:
        return (
            "Salah satu kelompok (konsisten / tidak konsisten) kosong -- "
            "perbandingan rate ESM tidak bisa dibaca."
        )
    if rate_i > rate_c:
        return (
            f"Query TIDAK konsisten lebih sering bermasalah di ESM "
            f"({rate_i:.2%} vs {rate_c:.2%}) -- ketidakstabilan terhadap parafrase "
            "sejalan dengan kegagalan struktur SQL, bukan dua masalah terpisah."
        )
    if rate_i < rate_c:
        return (
            f"Query tidak konsisten JUSTRU lebih jarang bermasalah di ESM "
            f"({rate_i:.2%} vs {rate_c:.2%}) -- ketidakstabilan parafrase tampak "
            "terpisah dari akurasi struktur SQL."
        )
    return (
        f"Rate masalah ESM sama di kedua kelompok ({rate_i:.2%}) -- tidak ada "
        "kaitan yang terbaca antara konsistensi parafrase dan ESM."
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_dimension_4(
    baseline_qvt_data: List[dict],
    graphrag_qvt_data: List[dict],
    baseline_logs: Optional[List[dict]] = None,
    graphrag_logs: Optional[List[dict]] = None,
) -> dict:
    """
    Dimensi 4 -- Robustness & Konsistensi Query (RM2). Guide Bagian 3, "Dimensi 4".

    Args:
        baseline_qvt_data : isi data/qvt_variations/baseline_qvt.json
        graphrag_qvt_data : isi data/qvt_variations/graphrag_qvt.json
        baseline_logs     : isi data/raw_logs/baseline_log.json (opsional, HANYA
                            untuk silang-ESM langkah 6 -- lihat KEPUTUSAN 1)
        graphrag_logs     : isi data/raw_logs/graphrag_log.json (opsional, idem)

    Langkah 1 guide ("jalankan pipeline B dan G untuk tiap variasi NL") bukan
    tanggung jawab fungsi ini -- itu sudah terjadi di run_comparison() dan
    di-assemble oleh build_qvt_variations.py. Fungsi ini mulai dari langkah 2.
    """
    baseline = _qvt_stats(baseline_qvt_data, "baseline")
    graphrag = _qvt_stats(graphrag_qvt_data, "graphrag")

    delta_qvt = graphrag["qvt"] - baseline["qvt"]
    interpretation = _interpret_delta(delta_qvt)
    interpretation["delta_qvt"] = delta_qvt

    esm_cross: Optional[dict] = None
    if baseline_logs is not None and graphrag_logs is not None:
        esm_cross = {
            "baseline": _esm_cross(baseline_qvt_data, baseline, baseline_logs, "baseline"),
            "graphrag": _esm_cross(graphrag_qvt_data, graphrag, graphrag_logs, "graphrag"),
        }
        for cond in ("baseline", "graphrag"):
            esm_cross[cond]["note"] = _cross_note(esm_cross[cond])

    def _public(stats: dict) -> dict:
        return {k: v for k, v in stats.items() if k != "per_query"}

    return {
        "qvt": {"baseline": _public(baseline), "graphrag": _public(graphrag)},
        "delta_qvt": delta_qvt,
        "interpretation": interpretation,
        "esm_cross": esm_cross,
    }


def print_dimension_4(result: dict) -> None:
    """Cetak hasil run_dimension_4(): QVT berdampingan, interpretasi delta, silang-ESM."""
    qvt = result["qvt"]
    interp = result["interpretation"]

    print("=" * 78)
    print("Dimensi 4 -- Robustness & Konsistensi Query (QVT)")
    print("=" * 78)
    print(f"{'':10} {'QVT':>10} {'M (lolos)':>12} {'dibuang':>10} {'total':>8}")
    for label, key in [("Baseline", "baseline"), ("GraphRAG", "graphrag")]:
        s = qvt[key]
        print(f"{label:10} {s['qvt']:>10.4f} {s['m_included']:>12} "
              f"{s['n_excluded']:>10} {s['n_total']:>8}")
    print("\n('dibuang' = filter wajib guide 1.6(e): semua variasinya salah)")

    print("\n" + "=" * 78)
    print("Interpretasi (threshold +/-2%, WAJIB dari guide)")
    print("=" * 78)
    print(f"delta_QVT = {interp['delta_qvt']:+.4f} ({interp['delta_qvt'] * 100:+.2f} poin persen)")
    print(f"Band      : {interp['band']}")
    print(f"Bacaan    : {interp['interpretation']}")

    print("\n" + "=" * 78)
    print("Silang QVT x ESM (langkah 6 guide)")
    print("=" * 78)
    if result["esm_cross"] is None:
        print("[!] DILEWATI -- raw_logs tidak dioper ke run_dimension_4().")
        print("    Data QVT hanya membawa EX (field 'is_correct', poin 10), tidak ESM.")
        print("    Oper baseline_logs= dan graphrag_logs= untuk menjalankan langkah ini.")
        return

    for label, key in [("Baseline", "baseline"), ("GraphRAG", "graphrag")]:
        c = result["esm_cross"][key]
        print(f"\n-- {label} --")
        print(f"{'':18} {'ESM bersih':>12} {'ESM bermasalah':>16}")
        print(f"{'konsisten':18} {c['consistent_esm_ok']:>12} {c['consistent_esm_problem']:>16}")
        print(f"{'tidak konsisten':18} {c['inconsistent_esm_ok']:>12} "
              f"{c['inconsistent_esm_problem']:>16}")
        print(f"  {c['note']}")
        if c["n_missing_in_logs"]:
            print(f"  [!] {c['n_missing_in_logs']} query dilewati -- ada variasi yang "
                  f"query_id-nya tidak ada di raw_logs.")
