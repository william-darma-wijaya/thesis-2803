"""
combine_models.py — gabungkan hasil beberapa model jadi satu file perbandingan.

Cari setiap metrics_results.json di bawah --input-dir (default outputs/models/, boleh folder
hasil ekstrak zip dari tiap orang), baca juga dimensions_results.json di folder yang sama, lalu tulis:
    outputs/tables/hasil_gabungan.json
    outputs/tables/hasil_gabungan.xlsx

Peringatan ditulis di konsol dan sheet Info kalau ada model yang belum ada, run percobaan,
atau konfigurasi antar model yang berbeda (hasilnya jadi tidak sebanding).

Usage:
    python src/experiments/combine_models.py
    python src/experiments/combine_models.py --input-dir hasil_kaggle --out-dir outputs/tables
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
from openpyxl import Workbook

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments import result_tables as rt  # noqa: E402
from src.experiments.export_excel import D6_NOTE, Sheet, _read_csv  # noqa: E402
from src.experiments.metrics_summary import CONDITION_LABELS, CONDITIONS, RESULTS_FILENAME, VARIANT_KEYS  # noqa: E402
from src.utils.timing import RUN_FILENAME  # noqa: E402

# Harus sama di semua model supaya perbandingannya adil.
COMPARABLE_KEYS = ("commit", "sample_ratio", "seed", "few_shot_k", "few_shot_selection",
                   "table_threshold", "column_threshold", "alpha", "queries", "gpu", *VARIANT_KEYS)


def _slug(model: str) -> str:
    return str(model).rstrip("/").split("/")[-1]


def _model_size(slug: str) -> float:
    found = re.search(r"(\d+(?:\.\d+)?)B", slug)
    return float(found.group(1)) if found else float("inf")


def _expected_models() -> list:
    try:
        from src.core.config import PipelineConfig
        return PipelineConfig().llm_model_frame
    except ImportError:   # config.py butuh torch; tanpa torch cek model yang hilang dilewati
        return []


def load_runs(input_dir: Path) -> dict:
    """{slug model: {"metrics": ..., "dimensions": ...}}. Model yang sama dua kali = error, bukan dipilih diam-diam."""
    found: dict = {}
    for path in sorted(Path(input_dir).rglob(RESULTS_FILENAME)):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        slug = _slug(metrics["meta"].get("llm_model", "unknown"))
        if slug in found:
            raise ValueError(f"Hasil model {slug} ditemukan dua kali: {found[slug]['path']} dan {path}. "
                             "Sisakan satu sebelum menggabungkan.")
        dims_path = path.parent / "dimensions_results.json"
        dims = json.loads(dims_path.read_text(encoding="utf-8")) if dims_path.exists() else None
        logs_dir = path.parent.parent / "raw_logs"   # ikut di zip hasil_<MODEL>.zip
        logs = None
        if (logs_dir / "baseline_log.json").exists() and (logs_dir / "graphrag_log.json").exists():
            logs = tuple(json.loads((logs_dir / f"{c}_log.json").read_text(encoding="utf-8")) for c in CONDITIONS)
        run_path = path.parent / RUN_FILENAME
        latency_run = json.loads(run_path.read_text(encoding="utf-8")) if run_path.exists() else None
        found[slug] = {"path": path, "metrics": metrics, "dimensions": dims, "logs": logs, "latency_run": latency_run}
    return dict(sorted(found.items(), key=lambda kv: (_model_size(kv[0]), kv[0])))


def find_warnings(runs: dict, expected: Optional[list] = None) -> list:
    warnings = []
    missing = [_slug(m) for m in (expected or []) if _slug(m) not in runs]
    if missing:
        warnings.append(f"Model belum ada hasilnya: {', '.join(missing)}")
    for slug, run in runs.items():
        meta = run["metrics"]["meta"]
        if not meta.get("is_full_run"):
            warnings.append(f"{slug}: run percobaan (SAMPLE_RATIO = {meta.get('sample_ratio')})")
        if run["dimensions"] is None:
            warnings.append(f"{slug}: dimensions_results.json tidak ada")
        if run["logs"] is None:
            warnings.append(f"{slug}: raw_logs tidak ada, reduksi kolom dan token dilewati")
    for key in COMPARABLE_KEYS:
        values = {slug: run["metrics"]["meta"].get(key) for slug, run in runs.items()}
        if len({json.dumps(v, sort_keys=True) for v in values.values()}) > 1:
            warnings.append(f"Nilai {key} berbeda antar model: {values}")
    return warnings


def summary_table(runs: dict) -> pd.DataFrame:
    """Satu baris per model × varian: ringkasan keenam metrik."""
    rows = []
    for slug, run in runs.items():
        m, q = run["metrics"]["metrics"], run["metrics"]["meta"]["queries"]
        tep = m["token_tep"]["tep"]
        for c in CONDITIONS:
            sla, qvt = m["sla"][c]["column"] or {}, m["qvt"][c]
            rows.append({
                "Model": slug, "Variant": CONDITION_LABELS[c], "Queries": q[c],
                "SLA column F1": sla.get("f1"), "ESM": m["esm"][c]["overall"], "EX": m["ex"][c]["overall"],
                "T mean": m["token_tep"]["token"][c]["t_mean"],
                "TEP (2-stage vs 1-stage)": tep.get("tep") if c == "graphrag" else None,
                "QVT": qvt.get("qvt"),
            })
    return rt.with_formats(pd.DataFrame(rows), {"Queries": "int", "SLA column F1": "pct", "ESM": "pct", "EX": "pct",
                                                "T mean": "num1", "TEP (2-stage vs 1-stage)": "num4", "QVT": "pct"},
                           title="All models: six metrics")


def _concat_by_title(tables_per_model: list) -> list:
    """Gabungkan tabel berjudul sama dari tiap model, dengan kolom Model di depan."""
    grouped: dict = {}
    for slug, tables in tables_per_model:
        for table in tables:
            with_model = table.copy()
            with_model.insert(0, "Model", slug)
            grouped.setdefault(table.attrs.get("title"), []).append((with_model, table.attrs))
    combined = []
    for pairs in grouped.values():
        df = pd.concat([d for d, _ in pairs], ignore_index=True)
        df.attrs.update(pairs[0][1])
        combined.append(df)
    return combined


def write_outputs(runs: dict, warnings: list, out_dir: Path, shared_tables_dir: Path) -> tuple:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, xlsx_path = out_dir / "hasil_gabungan.json", out_dir / "hasil_gabungan.xlsx"
    json_path.write_text(json.dumps({
        "warnings": warnings,
        "models": {s: {"metrics": r["metrics"], "dimensions": r["dimensions"]} for s, r in runs.items()},
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    Sheet._n_tables = 0
    wb = Workbook()
    wb.remove(wb.active)
    # Tanpa sheet Info dan judul (poin 41): peringatan dan daftar run tetap ada, di sheet sendiri
    notes = list(warnings) + ([D6_NOTE] if any(r["dimensions"] for r in runs.values()) else [])
    checks = Sheet(wb, "Warnings")
    checks.table(pd.DataFrame({"Warning": notes or ["None"]}))
    checks.table(pd.DataFrame([{"Model": s, "Full run": "Yes" if r["metrics"]["meta"].get("is_full_run") else "No",
                                "Commit": r["metrics"]["meta"].get("commit"), "Source": str(r["path"])}
                               for s, r in runs.items()]))
    checks.finish()

    summary = Sheet(wb, "Summary")
    summary.table(summary_table(runs))
    summary.finish()

    all_metrics = Sheet(wb, "All Metrics")
    for table in _concat_by_title([(s, [rt.metrics_summary_table(r["metrics"])]) for s, r in runs.items()]):
        all_metrics.table(table)
    all_metrics.finish()

    per_metric = [(s, rt.metric_tables(r["metrics"])) for s, r in runs.items()]
    for idx, (sheet_name, _title, _) in enumerate(per_metric[0][1]):
        sheet = Sheet(wb, sheet_name)
        for table in _concat_by_title([(s, groups[idx][2]) for s, groups in per_metric]):
            sheet.table(table)
        sheet.finish()

    for n, name in rt.DIM_NAMES.items():
        tables = [(s, rt.DIM_TABLES[n](r["dimensions"])) for s, r in runs.items() if r["dimensions"]]
        tables = _concat_by_title(tables)
        if tables:
            sheet = Sheet(wb, f"D{n}")
            for table in tables:
                sheet.table(table)
            sheet.finish()

    reduction = [(s, rt.token_reduction(*r["logs"], r["metrics"]["meta"]["alpha"]) + [rt.token_reduction_by_difficulty(*r["logs"])])
                 for s, r in runs.items() if r["logs"]]
    if reduction:
        sheet = Sheet(wb, "Token Reduction")
        for table in _concat_by_title(reduction):
            sheet.table(table)
        sheet.finish()

    latency = [(s, rt.latency_tables(*(r["logs"] or ([], [])), r["latency_run"])) for s, r in runs.items()]
    latency = _concat_by_title(latency)
    if latency:
        sheet = Sheet(wb, "Latency")
        for table in latency:
            sheet.table(table)
        sheet.finish()

    ablation = rt.ablation_tables(_read_csv(Path(shared_tables_dir) / "ablation_results.csv"))
    if ablation:
        sheet = Sheet(wb, "Ablation")
        for table in ablation:
            sheet.table(table)
        sheet.finish()

    wb.save(xlsx_path)
    return json_path, xlsx_path


def main() -> int:
    ap = argparse.ArgumentParser(description="Gabungkan hasil beberapa model jadi satu file.")
    ap.add_argument("--input-dir", default="outputs/models")
    ap.add_argument("--out-dir", default="outputs/tables")
    ap.add_argument("--shared-tables-dir", default="outputs/tables", help="Lokasi ablation_results.csv.")
    args = ap.parse_args()

    runs = load_runs(Path(args.input_dir))
    if not runs:
        print(f"Tidak ada {RESULTS_FILENAME} di bawah {args.input_dir}.")
        return 1
    warnings = find_warnings(runs, _expected_models())
    for warning in warnings:
        print(f"[PERINGATAN] {warning}")
    json_path, xlsx_path = write_outputs(runs, warnings, Path(args.out_dir), Path(args.shared_tables_dir))
    print(f"Model digabung: {', '.join(runs)}")
    print(f"Tersimpan: {json_path}\nTersimpan: {xlsx_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
