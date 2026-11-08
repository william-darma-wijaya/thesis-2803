"""
export_excel.py — rangkum seluruh hasil run jadi SATU workbook Excel untuk dokumentasi.

Membaca (sheet yang sumbernya tidak ada hanya berisi catatan "Source not found", atau dibuang dengan --skip-missing):
    data/raw_logs/{baseline,graphrag}_log.json      -> Overall, By Difficulty, Retrieval, SQL Diagnostics, Per Query
    outputs/tables/metrics_results.json             -> M1..M6, satu sheet per metrik (metrics_summary.py)
    outputs/tables/latency_run.json                 -> Latency (src/utils/timing.py)
    outputs/tables/confound_diagnostics.json        -> SQL Diagnostics (pipeline.py --baseline / eval_pipeline.ipynb)
    outputs/tables/spider_official_eval.csv         -> Spider Official (eval_pipeline.ipynb bagian 14)
    outputs/tables/dimensions_results.json          -> Dimensions, D1..D6 (run_all_dimensions.py)
    outputs/tables/ablation_results.csv             -> Ablation (few-shot k, tahap 2)
    outputs/tables/model_comparison.csv             -> Models (tahap 3)
    outputs/tables/trial_preprocessing_recap.csv    -> Trial Preprocessing (trial_preprocessing.ipynb)
    outputs/tables/trial_thresholds_recap.csv       -> Trial Thresholds (trial_thresholds.ipynb)
    outputs/tables/sla_trial_recap.csv              -> Trial Preprocessing, Trial Thresholds (rekap lama graphrag_sla_trial.ipynb,
                                                       hanya kalau dua file di atas belum ada)

Tabelnya dibangun oleh src/experiments/result_tables.py, modul yang SAMA dengan yang ditampilkan
eval_pipeline.ipynb, jadi angka notebook dan workbook identik. Isinya hanya angka faktual:
teks interpretasi/kesimpulan dari dimensions_results.json tidak ikut diekspor.

Setiap sheet langsung berisi tabel mulai baris 1, tanpa sheet Info, judul, atau catatan (poin 41).
Setiap tabel adalah Excel Table (filter + baris berselang), persentase disimpan sebagai pecahan
dengan format %, selisih persentase dalam poin persen (pp).

Usage (dari root repo, lokal maupun Kaggle):
    python src/experiments/export_excel.py
    python src/experiments/export_excel.py --out outputs/tables/hasil_skripsi.xlsx
    python src/experiments/export_excel.py --skip-missing --out hasil_trial.xlsx   # ekspor parsial
    python src/experiments/export_excel.py --raw-logs-dir data/raw_logs/Qwen2.5-Coder-3B-Instruct \
        --tables-dir outputs/models/Qwen2.5-Coder-3B-Instruct/tables
"""

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments import result_tables as rt  # noqa: E402

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

D6_NOTE = ("EX per k comes from ablation.py, which runs on one model. "
           "The values do not describe the model of this workbook unless that model ran the ablation.")
FONT = "Arial"
GREY = "595959"
TABLE_STYLE = "TableStyleMedium2"
TEXT_WIDTH_CAP = 60      # kolom teks panjang (SQL) dipotong lebarnya, tidak di-wrap
NUM_WIDTH = 12


def _font(bold=False, italic=False, color="000000", size=10):
    return Font(name=FONT, bold=bold, italic=italic, color=color, size=size)


def _cell_value(v):
    if rt.is_missing(v):
        return None
    if hasattr(v, "item"):   # numpy scalar -> Python
        v = v.item()
    return v


class Sheet:
    """Satu worksheet: satu atau lebih tabel (masing-masing Excel Table), dimulai dari baris 1."""

    _n_tables = 0   # nama Excel Table harus unik di seluruh workbook

    def __init__(self, wb: Workbook, name: str):
        self.ws = wb.create_sheet(name)
        self.ws.sheet_view.showGridLines = False
        self.ws.page_setup.orientation = "landscape"
        self.ws.page_setup.fitToWidth = 1
        self.ws.page_setup.fitToHeight = 0
        self.ws.sheet_properties.pageSetUpPr.fitToPage = True
        self.widths: dict[int, float] = {}
        self.header_rows: list[int] = []
        self.has_data = False
        self.r = 1

    def note(self, text: str):
        self.ws.cell(self.r, 1, text).font = _font(italic=True, color=GREY)
        self.r += 1

    def _fit(self, col: int, width: float):
        self.widths[col] = max(self.widths.get(col, 8), min(width, TEXT_WIDTH_CAP))

    def table(self, df: pd.DataFrame):
        cols = rt.visible_columns(df)
        if df.empty or not cols:
            self.note("No rows.")
            self.r += 1
            return
        header = self.r
        for j, col in enumerate(cols, 1):
            c = self.ws.cell(header, j, str(col))
            c.font = _font(bold=True, color="FFFFFF")
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            self._fit(j, len(str(col)) * 1.1 + 4)
        self.ws.row_dimensions[header].height = 30
        for i in range(len(df)):
            row = header + 1 + i
            for j, col in enumerate(cols, 1):
                v = _cell_value(df[col].iloc[i])
                c = self.ws.cell(row, j, v)
                c.font = _font()
                kind = rt.cell_kind(df, i, col)
                xl = rt.KINDS.get(kind, rt.KINDS["text"])[1]
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    if xl:
                        c.number_format = xl
                    c.alignment = Alignment(horizontal="right")
                    self._fit(j, NUM_WIDTH)
                else:
                    if isinstance(v, bool):
                        c.value = "Yes" if v else "No"
                    c.alignment = Alignment(horizontal="left")
                    self._fit(j, len(str(v or "")) * 1.05 + 2)
        last = header + len(df)
        Sheet._n_tables += 1
        ref = f"A{header}:{get_column_letter(len(cols))}{last}"
        tbl = Table(displayName=f"T{Sheet._n_tables:03d}_{re.sub(r'[^A-Za-z0-9]', '', self.ws.title)[:20]}", ref=ref)
        tbl.tableStyleInfo = TableStyleInfo(name=TABLE_STYLE, showRowStripes=True)
        self.ws.add_table(tbl)
        self.header_rows.append(header)
        self.has_data = True
        self.r = last + 2

    def finish(self, freeze_col: str = "A"):
        for col, w in self.widths.items():
            self.ws.column_dimensions[get_column_letter(col)].width = w
        if len(self.header_rows) == 1:   # satu tabel: header tetap terlihat saat scroll
            self.ws.freeze_panes = f"{freeze_col}{self.header_rows[0] + 1}"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_csv(p: Path, **kw):
    try:
        df = pd.read_csv(p, **kw)
        return df if len(df) else None
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return None


def _alpha_default() -> float:
    try:
        from src.core.config import PipelineConfig
        return float(PipelineConfig().token_output_weight)
    except Exception:   # torch tidak ada -> pakai nilai tetap dari proposal
        return 3.0


# ---------------------------------------------------------------------------
# Workbook
# ---------------------------------------------------------------------------

def build_workbook(raw_logs_dir: Path, tables_dir: Path, out: Path, skip_missing: bool = False) -> Path:
    """skip_missing=True: buang sheet yang sumbernya tidak ada (ekspor parsial, mis. hanya trial SLA
    atau hanya ablation), alih-alih menampilkan sheet berisi catatan "source not found"."""
    Sheet._n_tables = 0
    alpha = _alpha_default()

    logs = {c: _read_json(raw_logs_dir / f"{c}_log.json") for c in ("baseline", "graphrag")}
    metrics = _read_json(tables_dir / "metrics_results.json")
    dims = _read_json(tables_dir / "dimensions_results.json")
    latency_run = _read_json(tables_dir / "latency_run.json")
    confound = _read_json(tables_dir / "confound_diagnostics.json")
    spider = _read_csv(tables_dir / "spider_official_eval.csv")
    ablation = _read_csv(tables_dir / "ablation_results.csv")
    models = _read_csv(tables_dir / "model_comparison.csv")
    trial_prep = _read_csv(tables_dir / "trial_preprocessing_recap.csv")
    trial_thr = _read_csv(tables_dir / "trial_thresholds_recap.csv")
    trial_old = _read_csv(tables_dir / "sla_trial_recap.csv") if trial_prep is None and trial_thr is None else None
    have_logs = bool(logs["baseline"]) and bool(logs["graphrag"])
    b, g = logs["baseline"] or [], logs["graphrag"] or []

    wb = Workbook()
    wb.remove(wb.active)
    sheets = []

    def add(name: str, tables: list, missing: str, freeze_col: str = "A"):
        sh = Sheet(wb, name)
        for t in tables:
            sh.table(t)
        if not sh.has_data:
            sh.note(f"Source not found: {missing}")
        sh.finish(freeze_col)
        sheets.append(sh)

    # Urutan sheet: ringkasan 6 metrik -> rincian metrik -> dimensi -> rincian pendukung.
    add("Summary", [rt.metrics_summary_table(metrics)] if metrics else [], f"{tables_dir}/metrics_results.json")
    metric_groups = rt.metric_tables(metrics) if metrics else []
    for name, _title, tables in metric_groups:
        add(name, tables, f"{tables_dir}/metrics_results.json")
    if not metric_groups:
        add("Metrics", [], f"{tables_dir}/metrics_results.json")

    # Alasan skip ditulis dalam bahasa Indonesia oleh run_all_dimensions.py, jadi tidak ikut ke Excel (ada di dimensions_report.txt)
    add("Dimensions", [rt.dim_status(dims).drop(columns="Reason")] if dims else [], f"{tables_dir}/dimensions_results.json")
    for n in rt.DIM_NAMES:
        add(f"D{n}", rt.DIM_TABLES[n](dims), f"dimension_{n} with status 'ok' in dimensions_results.json")

    raw_missing = f"{raw_logs_dir}/baseline_log.json and graphrag_log.json"
    add("Overall", [rt.overall_comparison(b, g, alpha)] if have_logs else [], raw_missing)
    add("By Difficulty", [rt.by_difficulty(b, g, alpha)] if have_logs else [], raw_missing)
    add("Token Reduction", (rt.token_reduction(b, g, alpha) + [rt.token_reduction_by_difficulty(b, g)]) if have_logs else [], raw_missing)
    add("Retrieval", [rt.retrieval_metrics(b, g), rt.retrieval_by_difficulty(b, g)] if have_logs else [], raw_missing)
    diag = ([rt.prediction_checks(b, g), rt.outcome_matrix(b, g), rt.paired_ex(b, g), rt.sql_feature_usage(b, g),
             rt.cm_from_logs(b, g)] if have_logs else []) + ([rt.confound_table(confound)] if confound else [])
    add("SQL Diagnostics", diag, raw_missing)
    if spider is not None:
        spider = spider.assign(Pipeline=spider["Pipeline"].map(rt.pipeline_label))     # CSV memakai nama lama
        parsed = {lab: sub.drop(columns="Pipeline") for lab, sub in spider.groupby("Pipeline", sort=False)}
        long = rt.with_formats(spider.copy(), {c: "num3" for c in spider.columns if c not in ("Pipeline", "Section", "Metric")})
        add("Spider Official", [rt.spider_summary(parsed), long], "")
    elif not skip_missing:
        add("Spider Official", [], f"{tables_dir}/spider_official_eval.csv (written by eval_pipeline.ipynb section 14)")
    add("Per Query", [rt.per_query(b, g, alpha)] if have_logs else [], raw_missing, freeze_col="D")
    add("Latency", rt.latency_tables(b, g, latency_run), raw_missing)
    latency_rows = rt.latency_per_query(b, g)
    add("Latency Per Query", [latency_rows] if latency_rows is not None else [], raw_missing, freeze_col="D")

    add("Ablation", rt.ablation_tables(ablation), f"{tables_dir}/ablation_results.csv")
    add("Models", rt.model_tables(models), f"{tables_dir}/model_comparison.csv")
    tt = rt.trial_tables(trial_prep, trial_thr) if trial_old is None else rt.sla_trial_tables(trial_old)
    # Kolom yang dibekukan = kolom yang mengenali baris saat digeser ke samping
    # (Trial Preprocessing: kelima pengaturan; Trial Thresholds: Pipeline, T, K)
    for name, key, source, freeze_col in (("Trial Preprocessing", "steps", "trial_preprocessing_recap.csv", "F"),
                                          ("Trial Thresholds", "grid", "trial_thresholds_recap.csv", "D")):
        add(name, [tt[key]] if tt[key] is not None else [], f"{tables_dir}/{source}", freeze_col=freeze_col)

    if skip_missing and any(sh.has_data for sh in sheets):     # workbook butuh minimal satu sheet
        for sh in sheets:
            if not sh.has_data:
                wb.remove(sh.ws)
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Rangkum hasil run jadi satu file Excel.")
    ap.add_argument("--raw-logs-dir", default="data/raw_logs")
    ap.add_argument("--tables-dir", default="outputs/tables")
    ap.add_argument("--out", default="outputs/tables/hasil_skripsi.xlsx")
    ap.add_argument("--skip-missing", action="store_true",
                    help="Buang sheet yang sumber datanya belum ada (ekspor parsial: hanya trial SLA, hanya ablation, dst.).")
    args = ap.parse_args()
    out = build_workbook(Path(args.raw_logs_dir), Path(args.tables_dir), Path(args.out), args.skip_missing)
    print(f"Excel tersimpan: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
