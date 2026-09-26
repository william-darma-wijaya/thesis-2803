"""
build_qvt_variations.py — Bangun data/qvt_variations/*.json dari paraphrase
ALAMI di SPIDER dev set, dijoin dengan raw_logs hasil run_comparison().

Referensi: context/EVALUATION_ANALYSIS_GUIDE.md Bagian 1.6 (QVT),
context/IMPLEMENTATION_DECISIONS.md poin 21.

LATAR BELAKANG
--------------
dev.json SPIDER (proses anotasinya melibatkan penulis pertanyaan berbeda +
langkah verifikasi paraphrase) ternyata sudah punya ~470 pasang (db_id, SQL)
dengan 2 pertanyaan NL berbeda untuk SQL yang PERSIS SAMA -- mencakup 90.9%
(940/1034) dev set. Itu persis N_i1, N_i2 yang dibutuhkan formula QVT (guide
1.6). Jadi TIDAK perlu generate parafrase baru (LLM atau manual) -- variasi
NL question yang genuinely ditulis manusia sudah ada di dev set itu sendiri.
Ini MENGGANTIKAN keputusan 2026-08-15 ("grouping dilakukan sendiri oleh
peneliti") -- lihat IMPLEMENTATION_DECISIONS.md poin 21 untuk alasan lengkap
dan batasannya (m_i selalu tepat 2, tidak pernah 3+; 94 baris singleton
di luar skema ini).

SYARAT PENTING -- BACA SEBELUM PAKAI
-------------------------------------
raw_logs (data/raw_logs/{baseline,graphrag}_log.json) HARUS berasal dari
`run_comparison()` (pipeline.py) dengan sample_ratio=1.0 (FULL dev set, TANPA
subsampling). `query_id` di raw_logs ("dev_XXXX") adalah POSISI baris di
dev_data saat pipeline jalan (lihat pipeline.py run_comparison(), sekitar
baris 648-652 & 782) -- itu hanya identik dengan posisi asli di dev.json kalau
tidak ada subsampling. Script ini mem-verifikasi len(raw_logs) == len(dev.json)
sebelum join; kalau tidak sama, kondisi itu DILEWATI dengan pesan error yang
jelas, bukan diam-diam menghasilkan pairing yang salah.

Cara pakai
----------
    # Sebelum raw_logs ada: cuma preview grouping (jalan tanpa GPU/pipeline)
    python src/experiments/build_qvt_variations.py --dry-run

    # Setelah `python src/experiments/pipeline.py --baseline` (sample=1.0)
    # selesai dan data/raw_logs/*.json ada:
    python src/experiments/build_qvt_variations.py
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.metrics.qvt import compute_qvt_per_query, aggregate_qvt  # noqa: E402


def build_groups(dev_data: List[dict]) -> List[List[dict]]:
    """
    Group baris dev.json berdasarkan (db_id, SQL text) identik. query_id per
    baris dihitung PERSIS seperti pipeline.py's run_comparison()
    (f"dev_{i+1:04d}", i = posisi 0-based di dev_data ASLI/tidak di-subsample)
    supaya nanti align 1:1 dengan raw_logs.

    Hanya group dengan >=2 anggota yang dikembalikan -- group beranggota 1
    tidak punya variasi untuk diukur variance-nya sama sekali (bukan bug,
    cuma di luar populasi QVT -- lihat poin 21).
    """
    groups: dict = defaultdict(list)
    for i, item in enumerate(dev_data):
        query_id = f"dev_{i + 1:04d}"
        key = (item["db_id"], item["query"].strip())
        groups[key].append({"query_id": query_id, "nl_question": item["question"]})

    multi = [members for members in groups.values() if len(members) >= 2]
    multi.sort(key=lambda members: min(m["query_id"] for m in members))
    return multi


def join_with_raw_logs(groups: List[List[dict]], raw_logs: List[dict]) -> List[dict]:
    """
    Isi tiap group dengan predicted_sql/ex_result dari raw_logs (satu kondisi).
    "is_correct" pakai ex_result sesuai keputusan poin 10 (EX, bukan ESM).

    Tiap variasi JUGA menyimpan "query_id" baris dev.json-nya sendiri (bukan cuma
    canonical_id group). Field ini tidak dipakai src/metrics/qvt.py, tapi WAJIB
    ada untuk silang-ESM di Dimensi 4 -- tanpa itu dim4 cuma bisa melihat ESM
    satu dari m_i variasi (lihat IMPLEMENTATION_DECISIONS.md poin 23).
    """
    by_id = {e["query_id"]: e for e in raw_logs}
    result = []
    skipped = 0
    for members in groups:
        variations = []
        gold_sql: Optional[str] = None
        ok = True
        for m in members:
            entry = by_id.get(m["query_id"])
            if entry is None:
                ok = False
                break
            gold_sql = entry["gold_sql"]
            variations.append({
                "query_id": m["query_id"],
                "nl_question": m["nl_question"],
                "predicted_sql": entry["predicted_sql"],
                "is_correct": entry["ex_result"],
            })
        if not ok:
            skipped += 1
            continue
        canonical_id = min(m["query_id"] for m in members)
        result.append({"query_id": canonical_id, "gold_sql": gold_sql, "variations": variations})

    if skipped:
        print(f"  [!] {skipped} grup dilewati -- query_id tidak ditemukan di raw_logs.")
    return result


def preview_qvt(qvt_data: List[dict]) -> None:
    """Sanity-check cepat: hitung QVT dari data yang baru dibangun, print ringkas."""
    per_query = [compute_qvt_per_query(e["query_id"], e["variations"]) for e in qvt_data]
    included = [r for r in per_query if r.included]
    excluded_n = len(per_query) - len(included)
    if not included:
        print("  QVT: semua query dibuang oleh filter wajib (semua variasi gagal) -- N/A")
        return
    qvt = aggregate_qvt(per_query)
    print(f"  QVT = {qvt:.4f}  (M={len(included)} query lolos filter, "
          f"{excluded_n} dibuang -- semua variasinya EX=0)")


def _print_grouping_stats(dev_data: List[dict], groups: List[List[dict]]) -> None:
    rows_covered = sum(len(g) for g in groups)
    print("=" * 70)
    print("QVT grouping -- paraphrase alami dari dev.json")
    print("=" * 70)
    print(f"  Total dev rows        : {len(dev_data)}")
    print(f"  Grup (SQL identik, >=2 pertanyaan): {len(groups)}")
    print(f"  Dev rows tercakup     : {rows_covered} ({rows_covered / len(dev_data) * 100:.1f}%)")
    print(f"  Rows di luar skema (singleton, 1 pertanyaan saja): {len(dev_data) - rows_covered}")


def build_qvt_files(dev_data: List[dict], raw_logs_dir: Path, out_dir: Path) -> dict:
    """
    Bangun {baseline,graphrag}_qvt.json di out_dir dari dev_data + raw_logs_dir.
    Dipanggil dari main() (CLI) dan dari pipeline.run_dimension_analysis() /
    notebook -- satu-satunya implementasi, supaya tidak ada dua jalur yang drift.

    Returns: {kondisi: jumlah grup tertulis}; kondisi yang dilewati (raw_logs
    tidak ada / hasil subsampling) tidak muncul di dict.
    """
    groups = build_groups(dev_data)
    _print_grouping_stats(dev_data, groups)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = {}
    for condition in ("baseline", "graphrag"):
        raw_logs_path = raw_logs_dir / f"{condition}_log.json"
        print(f"\n-- kondisi: {condition} --")
        if not raw_logs_path.exists():
            print(f"  [!] {raw_logs_path} belum ada -- jalankan "
                  f"`python src/experiments/pipeline.py --baseline` (sample=1.0) dulu. Dilewati.")
            continue

        with open(raw_logs_path, encoding="utf-8") as f:
            raw_logs = json.load(f)

        if len(raw_logs) != len(dev_data):
            print(f"  [!] len(raw_logs)={len(raw_logs)} != len(dev.json)={len(dev_data)} -- "
                  f"raw_logs ini kemungkinan dari run yang di-subsample (sample_ratio<1.0). "
                  f"query_id TIDAK bisa dipetakan balik ke posisi asli dev.json dengan aman. "
                  f"Jalankan ulang dengan --sample 1.0. Dilewati.")
            continue

        qvt_data = join_with_raw_logs(groups, raw_logs)
        out_path = out_dir / f"{condition}_qvt.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(qvt_data, f, ensure_ascii=False, indent=2)
        print(f"  Ditulis: {out_path} ({len(qvt_data)} query x 2 variasi)")
        preview_qvt(qvt_data)
        written[condition] = len(qvt_data)
    return written


def main():
    ap = argparse.ArgumentParser(
        description="Bangun data/qvt_variations/*.json dari paraphrase alami dev.json + raw_logs.")
    ap.add_argument("--dev-json", default="data/spider_data/dev.json",
                     help="Path ke dev.json SPIDER (default: data/spider_data/dev.json)")
    ap.add_argument("--raw-logs-dir", default="data/raw_logs")
    ap.add_argument("--out-dir", default="data/qvt_variations")
    ap.add_argument("--dry-run", action="store_true",
                     help="Cuma tampilkan statistik grouping, jangan baca raw_logs / tulis output "
                          "(dipakai sebelum pipeline --baseline dijalankan di Kaggle).")
    args = ap.parse_args()

    with open(Path(args.dev_json), encoding="utf-8") as f:
        dev_data = json.load(f)

    if args.dry_run:
        _print_grouping_stats(dev_data, build_groups(dev_data))
        print("\n[--dry-run] Berhenti di sini -- tidak baca raw_logs / tidak menulis output.")
        return

    build_qvt_files(dev_data, Path(args.raw_logs_dir), Path(args.out_dir))


if __name__ == "__main__":
    main()
