"""
export_excel.py — rangkum seluruh hasil run jadi SATU workbook Excel untuk dokumentasi.

Membaca (yang tidak ada dilewati, dan statusnya ditulis di sheet "Ringkasan"):
    data/raw_logs/{baseline,graphrag}_log.json      -> tabel perbandingan utama (Ringkasan)
    outputs/tables/dimensions_results.json          -> sheet D1..D6 (run_all_dimensions.py)
    outputs/tables/ablation_results.csv             -> sheet "Ablation Tahap 2" (few-shot k)
    outputs/tables/model_comparison.csv             -> sheet "Perbandingan Model" (tahap 3)
    outputs/tables/sla_trial_recap.csv              -> sheet "SLA Trial" (graphrag_sla_trial.ipynb)
    outputs/tables/sla_trial_T*_K*.csv              -> sheet "SLA Konfigurasi" (GraphRAG vs Baseline)

Kolom selisih / penghematan / rata-rata ditulis sebagai RUMUS Excel (bukan angka mati),
jadi ikut berubah kalau angka sumbernya diedit. Persentase disimpan sebagai pecahan dan
ditampilkan dengan format %.

Usage (dari root repo, lokal maupun Kaggle):
    python src/experiments/export_excel.py
    python src/experiments/export_excel.py --out outputs/tables/hasil_skripsi.xlsx
    python src/experiments/export_excel.py --raw-logs-dir data/raw_logs/Qwen2.5-Coder-3B-Instruct \
        --tables-dir outputs/models/Qwen2.5-Coder-3B-Instruct/tables
"""

import argparse
import glob
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

FONT = "Arial"
NAVY, LIGHT, GREEN, AMBER = "1F3864", "D9E2F3", "E2EFDA", "FFF2CC"
_THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)

PCT = "0.0%"
PP = '+0.0" pp";-0.0" pp";0.0" pp"'      # selisih dalam poin persentase
NUM1 = "#,##0.0"
DELTA1 = "+#,##0.0;-#,##0.0;0.0"
REL = "+0.0%;-0.0%;0.0%"

DIFFICULTIES = ["easy", "medium", "hard", "extra"]
DIFF_LABEL = {"easy": "Easy", "medium": "Medium", "hard": "Hard", "extra": "Extra hard"}
CLAUSES = ["select", "from", "where", "group_by", "order_by", "having", "keywords",
           "union", "union_all", "intersect", "except"]
CLAUSE_LABEL = {"select": "SELECT", "from": "FROM", "where": "WHERE", "group_by": "GROUP BY",
                "order_by": "ORDER BY", "having": "HAVING", "keywords": "Keywords",
                "union": "UNION", "union_all": "UNION ALL", "intersect": "INTERSECT", "except": "EXCEPT"}


def _font(bold=False, italic=False, color="000000", size=10):
    return Font(name=FONT, bold=bold, italic=italic, color=color, size=size)


def _num(x):
    """Angka bersih untuk sel; NaN/None/'' -> None."""
    if x is None or x == "":
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


class Page:
    """Satu sheet dengan kursor baris: judul, catatan, tabel, teks berwarp."""

    TEXT_COLS = 8   # lebar (jumlah kolom) untuk teks panjang yang di-merge

    def __init__(self, wb: Workbook, title: str, heading: str, note: str | None = None, first: bool = False):
        self.ws = wb.active if first else wb.create_sheet()
        self.ws.title = title
        self.ws.sheet_view.showGridLines = False
        self.r = 1
        self.widths: dict[int, float] = {}
        self.spans: list[tuple[int, int, int, str]] = []   # (row, first col, last col, text)
        c = self.ws.cell(self.r, 1, heading)
        c.font = _font(bold=True, size=14, color=NAVY)
        self.r += 1
        if note:
            self.text(note, italic=True, color="595959")
        self.r += 1

    def _fit(self, col: int, width: float):
        self.widths[col] = max(self.widths.get(col, 8), min(width, 60))
        self.ws.column_dimensions[get_column_letter(col)].width = self.widths[col]

    def text(self, text: str, italic=False, bold=False, color="000000", fill: str | None = None, ncols: int | None = None):
        ncols = ncols or self.TEXT_COLS
        self.ws.merge_cells(start_row=self.r, start_column=1, end_row=self.r, end_column=ncols)
        c = self.ws.cell(self.r, 1, text)
        c.font = _font(bold=bold, italic=italic, color=color)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        if fill:
            c.fill = PatternFill("solid", start_color=fill)
        self.spans.append((self.r, 1, ncols, str(text)))
        self.r += 1

    def caption(self, text: str):
        c = self.ws.cell(self.r, 1, text)
        c.font = _font(bold=True, size=11, color=NAVY)
        self.r += 1

    def gap(self, n: int = 1):
        self.r += n

    def table(self, headers: list[str], rows: list[list], fmts: list | None = None,
              wrap_cols: tuple = (), min_widths: dict | None = None) -> tuple[int, int]:
        """Tulis header + baris. Nilai str berawalan '=' adalah rumus; '{r}' diganti nomor barisnya.
        Return (baris data pertama, baris data terakhir)."""
        fmts = fmts or [None] * len(headers)
        for j, h in enumerate(headers, 1):
            c = self.ws.cell(self.r, j, h)
            c.font = _font(bold=True, color="FFFFFF")
            c.fill = PatternFill("solid", start_color=NAVY)
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            c.border = BORDER
            self._fit(j, len(h) * 0.9 + 4)
        self.ws.row_dimensions[self.r].height = 32
        self.r += 1
        first = self.r
        for row in rows:
            tallest = 1
            for j, v in enumerate(row, 1):
                v = _num(v)
                if isinstance(v, str) and v.startswith("="):
                    v = v.replace("{r}", str(self.r))
                c = self.ws.cell(self.r, j, v)
                c.font = _font()
                c.border = BORDER
                if fmts[j - 1]:
                    c.number_format = fmts[j - 1]
                if j in wrap_cols:
                    c.alignment = Alignment(wrap_text=True, vertical="top")
                    tallest = max(tallest, math.ceil(len(str(v or "")) / ((min_widths or {}).get(j, 40))))
                elif isinstance(v, str) and not v.startswith("="):
                    c.alignment = Alignment(horizontal="left", vertical="top")
                else:
                    c.alignment = Alignment(horizontal="right", vertical="top")
                if not isinstance(v, str) or v.startswith("="):
                    self._fit(j, 13)
                else:
                    self._fit(j, len(v) * 0.95 + 2 if j not in wrap_cols else (min_widths or {}).get(j, 40))
            if tallest > 1:
                self.ws.row_dimensions[self.r].height = 14 * tallest
            self.r += 1
        last = self.r - 1
        for j, w in (min_widths or {}).items():
            self._fit(j, w)
        return first, last

    def table_text(self, headers: list[str], rows: list[list], span: int = 7) -> tuple[int, int]:
        """Seperti table(), tapi kolom TERAKHIR adalah teks panjang yang di-merge sampai kolom `span`
        (supaya lebar kolom tabel lain di sheet yang sama tidak ikut melebar)."""
        n = len(headers)
        for j, h in enumerate(headers, 1):
            c = self.ws.cell(self.r, j, h)
            c.font = _font(bold=True, color="FFFFFF")
            c.fill = PatternFill("solid", start_color=NAVY)
            c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            c.border = BORDER
        for j in range(n + 1, span + 1):
            self.ws.cell(self.r, j).fill = PatternFill("solid", start_color=NAVY)
        self.ws.merge_cells(start_row=self.r, start_column=n, end_row=self.r, end_column=span)
        self.ws.row_dimensions[self.r].height = 22
        self.r += 1
        first = self.r
        for row in rows:
            for j, v in enumerate(row[:-1], 1):
                c = self.ws.cell(self.r, j, _num(v))
                c.font = _font(bold=(j == 1))
                c.border = BORDER
                c.alignment = Alignment(horizontal="left", vertical="top", wrap_text=True)
                self._fit(j, len(str(v)) * 0.95 + 2 if j == 1 else 12)
            self.ws.merge_cells(start_row=self.r, start_column=n, end_row=self.r, end_column=span)
            c = self.ws.cell(self.r, n, row[-1])
            c.font = _font()
            c.alignment = Alignment(wrap_text=True, vertical="top")
            for j in range(n, span + 1):
                self.ws.cell(self.r, j).border = BORDER
            self.spans.append((self.r, n, span, str(row[-1] or "")))
            self.r += 1
        return first, self.r - 1

    def kv(self, label: str, value, fmt: str | None = None, note: str | None = None, input_cell: bool = False) -> int:
        """Satu baris 'label | nilai | catatan'. input_cell -> huruf biru (asumsi yang bisa diubah)."""
        a = self.ws.cell(self.r, 1, label)
        a.font = _font(bold=True)
        a.border = BORDER
        b = self.ws.cell(self.r, 2, _num(value))
        b.font = _font(color="0000FF" if input_cell else "000000")
        b.border = BORDER
        b.alignment = Alignment(horizontal="right", vertical="top", wrap_text=isinstance(value, str))
        if fmt:
            b.number_format = fmt
        if note:
            c = self.ws.cell(self.r, 3, note)
            c.font = _font(italic=True, color="595959")
        self._fit(1, len(label) * 0.95 + 3)
        self._fit(2, 14)
        r = self.r
        self.r += 1
        return r

    def finish(self, freeze: str | None = None):
        """Idempotent: bekukan baris/kolom dan atur tinggi baris teks panjang dari lebar kolom yang SUDAH final."""
        if freeze:
            self.ws.freeze_panes = freeze
        for c in range(1, self.TEXT_COLS + 1):
            if c not in self.widths:
                self.ws.column_dimensions[get_column_letter(c)].width = 14
        for row, c1, c2, text in self.spans:
            width = sum(self.ws.column_dimensions[get_column_letter(c)].width or 8.43 for c in range(c1, c2 + 1))
            lines = sum(max(1, math.ceil(len(part) / (width * 1.05))) for part in text.split("\n"))
            self.ws.row_dimensions[row].height = max(15, 14 * lines + 2)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _read_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _read_csv(p: Path):
    try:
        df = pd.read_csv(p)
        return df if len(df) else None
    except (OSError, ValueError, pd.errors.EmptyDataError):
        return None


def _alpha_default() -> float:
    try:
        from src.core.config import PipelineConfig
        return float(PipelineConfig().token_output_weight)
    except Exception:   # torch tidak ada -> pakai nilai tetap dari proposal
        return 3.0


def _condition_stats(logs: list[dict]) -> dict:
    """Ringkasan satu kondisi (Baseline/GraphRAG) dari raw_logs. SLA = macro-average per query (sla.py)."""
    from src.metrics.sla import aggregate_sla, compute_sla

    def sla(entries, level):
        return aggregate_sla([compute_sla(set(e["gold_schema"]), set(e["predicted_schema"]), level=level) for e in entries])

    n = len(logs)
    ne = [e for e in logs if e["gold_schema"]]
    t, c, cne = sla(logs, "table"), sla(logs, "column"), sla(ne, "column") if ne else None
    mean = lambda key: sum(e[key] for e in logs) / n
    return {
        "n": n, "ex": mean("ex_result"), "esm": mean("esm_result"),
        "t_in": mean("token_input"), "t_out": mean("token_output"),
        "tbl_P": t.precision, "tbl_R": t.recall, "tbl_F1": t.f1,
        "col_P": c.precision, "col_R": c.recall, "col_F1": c.f1,
        "cols_sent": sum(len(e["predicted_schema"]) for e in logs) / n,
        "ne_n": len(ne), "col_P_ne": cne.precision if cne else None, "col_R_ne": cne.recall if cne else None,
    }


# ---------------------------------------------------------------------------
# Sheets
# ---------------------------------------------------------------------------

def sheet_main(page: Page, stats: dict | None, alpha: float) -> None:
    """Tabel perbandingan utama Baseline vs GraphRAG dari raw_logs."""
    page.caption("Perbandingan utama: Baseline vs GraphRAG")
    if not stats:
        page.text("data/raw_logs/baseline_log.json dan graphrag_log.json belum ada, jadi tabel ini kosong. "
                  "Jalankan eval_pipeline.ipynb (SAMPLE_RATIO = 1.0) atau pipeline.py --baseline dulu.",
                  fill=AMBER)
        return
    b, g = stats["baseline"], stats["graphrag"]
    page.text("Angka dihitung dari raw_logs (satu baris per query). Kolom Selisih dan Perubahan relatif adalah rumus: "
              "Selisih = GraphRAG - Baseline (poin persentase untuk persen, satuan asli untuk token/kolom).",
              italic=True, color="595959")
    ra = page.kv("α (bobot token output)", alpha, "0.0", "token_output_weight di config.py, fixed dari proposal. "
                 "Ubah di sini kalau perlu: baris Token Consumption ikut berubah.", input_cell=True)
    page.gap()
    first = page.r + 1   # baris data pertama
    rows_spec = [
        # label, key / formula, format, lebih baik jika, catatan
        ("Jumlah query", "n", "0", "-", "Harus sama dengan jumlah dev set kalau run penuh (1034)."),
        ("Execution Accuracy (EX)", "ex", PCT, "lebih tinggi", "Hasil eksekusi SQL sama dengan gold (evaluator resmi Spider, order-sensitive)."),
        ("Exact Set Match (ESM)", "esm", PCT, "lebih tinggi", "Struktur SQL sama dengan gold, bukan hanya hasilnya."),
        ("Rata-rata token input (T_in)", "t_in", NUM1, "lebih rendah", "Panjang prompt: schema + few-shot + pertanyaan."),
        ("Rata-rata token output (T_out)", "t_out", NUM1, "lebih rendah", "Panjang SQL yang dihasilkan."),
        ("Token Consumption (T)", None, NUM1, "lebih rendah", "T = T_in + α × T_out."),
        ("SLA tabel: precision", "tbl_P", PCT, "lebih tinggi", "Dari tabel yang masuk prompt, berapa yang benar-benar dibutuhkan."),
        ("SLA tabel: recall", "tbl_R", PCT, "lebih tinggi", "Dari tabel yang dibutuhkan, berapa yang masuk prompt."),
        ("SLA tabel: F1", "tbl_F1", PCT, "lebih tinggi", ""),
        ("SLA kolom: precision", "col_P", PCT, "lebih tinggi", "Query bergold kosong (mis. COUNT(*)) dihitung P = R = 0 oleh SLA resmi."),
        ("SLA kolom: recall", "col_R", PCT, "lebih tinggi", "Recall kolom paling menentukan EX: kolom yang hilang tidak bisa dipakai LLM."),
        ("SLA kolom: F1", "col_F1", PCT, "lebih tinggi", ""),
        ("SLA kolom: precision (gold tak kosong)", "col_P_ne", PCT, "lebih tinggi", "Diagnostik: hanya query dengan gold schema tidak kosong."),
        ("SLA kolom: recall (gold tak kosong)", "col_R_ne", PCT, "lebih tinggi", "Diagnostik: batas atas recall resmi 95.9% karena 42 query bergold kosong."),
        ("Kolom terkirim per query", "cols_sent", NUM1, "lebih rendah", "Rata-rata kolom di prompt. Baseline mengirim semua kolom tabel terpilih."),
    ]
    rows = []
    for i, (label, key, fmt, better, note) in enumerate(rows_spec):
        r = first + i
        if key is None:   # Token Consumption = T_in + alpha*T_out (rumus)
            vb, vg = f"=B{first + 3}+$B${ra}*B{first + 4}", f"=C{first + 3}+$B${ra}*C{first + 4}"
        else:
            vb, vg = b[key], g[key]
        is_pct = fmt == PCT
        diff = f"=(C{{r}}-B{{r}})*100" if is_pct else "=C{r}-B{r}"
        rel = '=IF(B{r}=0,"",(C{r}-B{r})/B{r})' if not is_pct and key != "n" else ""
        rows.append([label, vb, vg, diff if key != "n" else "", rel, better, note])
    f, l = page.table(
        ["Metrik", "Baseline", "GraphRAG", "Selisih (G - B)", "Perubahan relatif", "Lebih baik jika", "Catatan"],
        rows, [None, None, None, None, REL, None, None], wrap_cols=(7,), min_widths={1: 34, 7: 60})
    for i, (label, key, fmt, *_rest) in enumerate(rows_spec):
        r = f + i
        for col in (2, 3):
            page.ws.cell(r, col).number_format = fmt
        page.ws.cell(r, 4).number_format = PP if fmt == PCT else DELTA1
    for r in (f + 1, f + 5, f + 10):   # EX, T, recall kolom: baris kunci
        for col in range(1, 8):
            page.ws.cell(r, col).fill = PatternFill("solid", start_color=LIGHT)
    page.gap()


def _dim_record(dims: dict | None, n: int):
    return (dims or {}).get(f"dimension_{n}")


def _status_page(page: Page, rec) -> bool:
    """Tulis alasan kalau dimensi tidak 'ok'. True kalau sheet bisa dilanjutkan."""
    if rec is None:
        page.text("Dimensi ini belum dijalankan (dimensions_results.json tidak memuatnya).", fill=AMBER)
        return False
    if rec.get("status") != "ok":
        page.text(f"Status: {rec.get('status')}. {rec.get('reason', '')}", fill=AMBER)
        return False
    return True


def sheet_dim1(wb, dims):
    p = Page(wb, "D1 Efisiensi", "Dimensi 1: Efisiensi Token & Performa",
             "Apakah GraphRAG menghemat token tanpa menurunkan EX? TEP (Token Elasticity of Performance) = "
             "(ΔEX/EX_Baseline) / (ΔT/T_Baseline): perubahan EX relatif per perubahan token relatif.")
    rec = _dim_record(dims, 1)
    if not _status_page(p, rec):
        return p
    r = rec["result"]
    tc, ex = r["token_consumption"], r["ex"]
    p.caption("Token Consumption (T = T_in + α × T_out), rata-rata per query")
    rows = [["Keseluruhan", tc["baseline"]["overall"]["mean"], tc["graphrag"]["overall"]["mean"], "=C{r}-B{r}",
             "=IF(B{r}=0,\"\",(B{r}-C{r})/B{r})", tc["baseline"]["overall"]["median"], tc["graphrag"]["overall"]["median"]]]
    for d in DIFFICULTIES:
        if d in tc["baseline"]["by_difficulty"]:
            rows.append([DIFF_LABEL[d], tc["baseline"]["by_difficulty"][d]["mean"], tc["graphrag"]["by_difficulty"][d]["mean"],
                         "=C{r}-B{r}", "=IF(B{r}=0,\"\",(B{r}-C{r})/B{r})",
                         tc["baseline"]["by_difficulty"][d]["median"], tc["graphrag"]["by_difficulty"][d]["median"]])
    f, l = p.table(["Tingkat kesulitan", "Baseline (rata-rata T)", "GraphRAG (rata-rata T)", "Selisih (G - B)",
                    "Penghematan token", "Baseline (median T)", "GraphRAG (median T)"],
                   rows, [None, NUM1, NUM1, DELTA1, PCT, NUM1, NUM1])
    p.text("Penghematan token = (T_Baseline - T_GraphRAG) / T_Baseline; positif berarti GraphRAG lebih hemat.", italic=True, color="595959")
    p.gap()
    p.caption("Execution Accuracy per tingkat kesulitan")
    rows = [["Keseluruhan", ex["baseline"]["overall"] / 100, ex["graphrag"]["overall"] / 100, "=(C{r}-B{r})*100"]]
    for d in DIFFICULTIES:
        if d in ex["baseline"]["by_difficulty"]:
            rows.append([DIFF_LABEL[d], ex["baseline"]["by_difficulty"][d] / 100, ex["graphrag"]["by_difficulty"][d] / 100, "=(C{r}-B{r})*100"])
    p.table(["Tingkat kesulitan", "EX Baseline", "EX GraphRAG", "Selisih (G - B)"], rows, [None, PCT, PCT, PP])
    p.gap()
    p.caption("TEP")
    t = r["tep"]
    p.kv("ΔEX (poin persen)", t["delta_ex"], "+0.00;-0.00", "EX_GraphRAG - EX_Baseline")
    p.kv("ΔT (token)", t["delta_t"], DELTA1, "T_GraphRAG - T_Baseline; negatif = lebih hemat")
    p.kv("TEP", t["tep"], "0.000")
    p.text("Interpretasi: " + str(t["interpretation"]), bold=True, fill=GREEN)
    p.finish()
    return p


def sheet_dim2(wb, dims):
    p = Page(wb, "D2 Struktur SQL", "Dimensi 2: Akurasi Struktur SQL",
             "EX menilai hasil eksekusi, ESM menilai struktur SQL. Kombinasi keduanya menunjukkan apakah SQL benar karena "
             "konteks schema yang presisi, atau hanya kebetulan hasilnya sama.")
    rec = _dim_record(dims, 2)
    if not _status_page(p, rec):
        return p
    r = rec["result"]["esm_ex"]
    p.caption("ESM dan EX")
    rows = []
    for label, b, g in [("Keseluruhan", r["baseline"]["overall"], r["graphrag"]["overall"])] + [
            (DIFF_LABEL[d], r["baseline"]["by_difficulty"][d], r["graphrag"]["by_difficulty"][d])
            for d in DIFFICULTIES if d in r["baseline"]["by_difficulty"]]:
        rows.append([label, b["esm"] / 100, g["esm"] / 100, "=(C{r}-B{r})*100", b["ex"] / 100, g["ex"] / 100, "=(F{r}-E{r})*100"])
    p.table(["Tingkat kesulitan", "ESM Baseline", "ESM GraphRAG", "Selisih ESM", "EX Baseline", "EX GraphRAG", "Selisih EX"],
            rows, [None, PCT, PCT, PP, PCT, PCT, PP])
    p.gap()
    p.caption("Klasifikasi")
    c = rec["result"]["classification"]
    p.kv("Kombinasi", c["combination"])
    p.kv("ΔEX / ΔESM (poin)", f"{c['delta_ex']:+.2f} / {c['delta_esm']:+.2f}")
    p.kv("Lanjut ke Dimensi 5?", "Ya" if c["escalate_to_dimension_5"] else "Tidak")
    p.text("Interpretasi: " + str(c["interpretation"]), bold=True, fill=GREEN)
    p.finish()
    return p


def sheet_dim3(wb, dims):
    p = Page(wb, "D3 Komponen SQL", "Dimensi 3: Akurasi Komponen SQL (Component Match)",
             "Klausa SQL mana yang paling terdampak oleh konteks schema yang lebih ringkas. Skor = persentase query yang klausa-nya cocok dengan gold.")
    rec = _dim_record(dims, 3)
    if not _status_page(p, rec):
        return p
    r = rec["result"]
    cm = r["cm_per_clause"]
    rows = [[CLAUSE_LABEL.get(k, k), cm["baseline"][k] / 100, cm["graphrag"][k] / 100, "=(C{r}-B{r})*100"]
            for k in CLAUSES if k in cm["baseline"]]
    f, l = p.table(["Klausa", "CM Baseline", "CM GraphRAG", "Selisih (G - B)"], rows, [None, PCT, PCT, PP])
    p.ws.conditional_formatting.add(
        f"D{f}:D{l}", FormulaRule(formula=[f"D{f}<0"], font=Font(name=FONT, color="C00000", bold=True)))
    p.ws.conditional_formatting.add(
        f"A{f}:D{l}", FormulaRule(formula=[f"$D{f}=MIN($D${f}:$D${l})"], fill=PatternFill("solid", bgColor=AMBER)))
    p.text("Selisih negatif (merah) = GraphRAG lebih buruk dari Baseline di klausa itu. Baris berwarna kuning = penurunan terbesar.",
           italic=True, color="595959")
    p.gap()
    bd = r["biggest_drop"]
    p.kv("Klausa dengan penurunan terbesar", CLAUSE_LABEL.get(bd["clause"], bd["clause"]))
    p.kv("Selisih (poin)", bd["delta"], "+0.00;-0.00")
    p.text("Interpretasi: " + str(bd["interpretation"]), bold=True, fill=GREEN)
    p.finish()
    return p


def _flat(o, prefix=""):
    """Dict bertingkat -> {'a / b': nilai skalar}."""
    out = {}
    if isinstance(o, dict):
        for k, v in o.items():
            out.update(_flat(v, f"{prefix} / {k}" if prefix else str(k)))
    elif isinstance(o, list):
        out[prefix] = ", ".join(map(str, o))
    else:
        out[prefix] = o
    return out


def sheet_dim4(wb, dims):
    p = Page(wb, "D4 Robustness", "Dimensi 4: Robustness & Konsistensi Query (QVT)",
             "Seberapa stabil hasil kalau pertanyaan yang sama diparafrase. QVT dihitung dari paraphrase alami di dev.json; "
             "hanya tersedia untuk run penuh (SAMPLE_RATIO = 1.0).")
    rec = _dim_record(dims, 4)
    if not _status_page(p, rec):
        return p
    r = rec["result"]
    b, g = _flat(r["qvt"]["baseline"]), _flat(r["qvt"]["graphrag"])
    keys = list(dict.fromkeys(list(b) + list(g)))
    rows = []
    for k in keys:
        vb, vg = b.get(k), g.get(k)
        both = isinstance(vb, (int, float)) and isinstance(vg, (int, float)) and not isinstance(vb, bool)
        rows.append([k, vb, vg, "=C{r}-B{r}" if both else ""])
    p.caption("Statistik QVT per kondisi")
    f, l = p.table(["Ukuran", "Baseline", "GraphRAG", "Selisih (G - B)"], rows, [None, "0.0000", "0.0000", "+0.0000;-0.0000;0.0000"])
    p.gap()
    if r.get("delta_qvt") is not None:
        p.kv("ΔQVT (GraphRAG - Baseline)", r["delta_qvt"], "+0.0000;-0.0000;0.0000")
    p.text("Interpretasi: " + str(r.get("interpretation", "")), bold=True, fill=GREEN)
    if r.get("esm_cross"):
        p.gap()
        p.caption("Silang QVT × ESM")
        flat = _flat(r["esm_cross"])
        p.table(["Ukuran", "Nilai"], [[k, v if not isinstance(v, bool) else ("Ya" if v else "Tidak")] for k, v in flat.items()],
                [None, "0.0000"])
    p.finish()
    return p


def sheet_dim5(wb, dims):
    p = Page(wb, "D5 Bottleneck", "Dimensi 5: Bottleneck Retrieval vs Generation",
             "Jika EX rendah, apakah penyebabnya retrieval (schema gagal diambil) atau generation (LLM gagal menyusun SQL)? "
             "Ambang: F1 SLA >= 0.80 dan EX >= 0.65. Hanya dijalankan kalau Dimensi 2 menunjukkan kegagalan menyeluruh "
             "(atau dipaksa dengan --always-dim5).")
    rec = _dim_record(dims, 5)
    if not _status_page(p, rec):
        return p
    rows = []
    for cond in ("baseline", "graphrag"):
        d = rec["result"][cond]
        for level, label in (("table_level", "Tabel"), ("column_level", "Kolom")):
            x = d[level]
            rows.append([d["condition_label"], label, d["ex_score"], x["f1_score_sla"], x["precision"], x["recall"],
                         "Ya" if x["f1_meets_threshold"] else "Tidak", "Ya" if x["ex_meets_threshold"] else "Tidak",
                         x["diagnosis"], x.get("precision_recall_note") or ""])
    f, l = p.table(["Kondisi", "Level SLA", "EX", "F1 SLA", "Precision", "Recall", "F1 ≥ 0.80?", "EX ≥ 0.65?", "Diagnosis", "Catatan"],
                   rows, [None, None, PCT, PCT, PCT, PCT, None, None, None, None], wrap_cols=(9, 10), min_widths={9: 48, 10: 52})
    p.ws.conditional_formatting.add(
        f"G{f}:H{l}", FormulaRule(formula=[f'G{f}="Tidak"'], fill=PatternFill("solid", bgColor="F8CBAD")))
    p.ws.conditional_formatting.add(
        f"G{f}:H{l}", FormulaRule(formula=[f'G{f}="Ya"'], fill=PatternFill("solid", bgColor=GREEN)))
    p.finish()
    return p


def sheet_dim6(wb, dims):
    p = Page(wb, "D6 Ablation Few-Shot", "Dimensi 6: Ablation Few-Shot (EX vs k)",
             "Mencari elbow point: berapa contoh few-shot (k) yang cukup sebelum EX tidak naik lagi atau malah turun.")
    rec = _dim_record(dims, 6)
    if not _status_page(p, rec):
        return p
    r = rec["result"]
    ks = sorted(r["table"], key=lambda x: int(x))
    f, l = p.table(["k (jumlah contoh)", "EX", "Selisih dari k sebelumnya"],
                   [[int(k), r["table"][k] / 100, ""] for k in ks], [None, PCT, PP])
    for row in range(f + 1, l + 1):   # selisih terhadap baris k sebelumnya (rumus)
        p.ws.cell(row, 3).value = f"=(B{row}-B{row - 1})*100"
        p.ws.cell(row, 3).number_format = PP
    p.ws.conditional_formatting.add(f"A{f}:C{l}", FormulaRule(formula=[f"$B{f}=MAX($B${f}:$B${l})"], fill=PatternFill("solid", bgColor=GREEN)))
    p.text("Baris hijau = EX tertinggi. Kolom selisih adalah rumus.", italic=True, color="595959")
    p.gap()
    p.kv("k final", r["k_final"])
    p.kv("Pola", r["pattern"])
    p.text("Alasan: " + str(r["reasoning"]), bold=True, fill=GREEN)
    if r.get("steps"):
        p.gap()
        p.caption("Langkah antar-k")
        p.table(["Dari k", "Ke k", "Selisih EX (poin)", "Signifikan?"],
                [[s["from_k"], s["to_k"], s["delta"], "Ya" if s["significant"] else "Tidak"] for s in r["steps"]],
                [None, None, "+0.00;-0.00", None])
    p.finish()
    return p


def _df_table(page: Page, df: pd.DataFrame, spec: list, wrap_cols=(), min_widths=None):
    """spec: [(kolom, header, format, skala)] -> tabel; kolom yang tidak ada dilewati."""
    spec = [s for s in spec if s[0] in df.columns]
    rows = []
    for _, rw in df.iterrows():
        row = []
        for key, _h, _fmt, scale in spec:
            v = _num(rw[key])
            row.append(v * scale if scale and isinstance(v, (int, float)) else v)
        rows.append(row)
    return page.table([s[1] for s in spec], rows, [s[2] for s in spec], wrap_cols, min_widths), spec


def sheet_ablation(wb, df):
    p = Page(wb, "Ablation Tahap 2", "Ablation Tahap 2: Few-Shot k (dipilih dengan EX)",
             "EX/ESM dihitung in-process per query. k bersama untuk Baseline dan GraphRAG = k dengan rata-rata EX kedua pipeline tertinggi "
             "(seri -> k lebih kecil). Baris hijau di tabel ringkas = k pemenang.")
    if df is None:
        p.text("outputs/tables/ablation_results.csv belum ada. Jalankan ablation.py --mode baseline dan --mode graphrag.", fill=AMBER)
        return p
    if "few_shot_selection" not in df.columns:
        df = df.assign(few_shot_selection="question")
    df["few_shot_selection"] = df["few_shot_selection"].fillna("question")
    for sel, sub in df.groupby("few_shot_selection", sort=False):
        p.caption(f"Ringkasan EX per k, seleksi few-shot: {sel}")
        piv = sub.pivot_table(index="k", columns="mode", values="avg_ex", aggfunc="last").sort_index()
        rows = []
        for k, rw in piv.iterrows():
            pct = lambda v: None if _num(v) is None else v / 100
            rows.append([int(k), pct(rw.get("baseline")), pct(rw.get("graphrag")),
                         "=IF(COUNT(B{r}:C{r})=0,\"\",AVERAGE(B{r}:C{r}))",
                         '=IF(COUNT(B{r}:C{r})=2,"","hanya 1 pipeline")'])
        f, l = p.table(["k", "EX Baseline", "EX GraphRAG", "Rata-rata EX (dipakai memilih k)", "Catatan"], rows, [None, PCT, PCT, PCT, None])
        p.ws.conditional_formatting.add(
            f"A{f}:E{l}", FormulaRule(formula=[f"AND(ISNUMBER($D{f}),$D{f}=MAX($D${f}:$D${l}))"], fill=PatternFill("solid", bgColor=GREEN)))
        p.gap()
    p.caption("Detail per pipeline dan k")
    spec = [("mode", "Pipeline", None, None), ("few_shot_selection", "Seleksi few-shot", None, None), ("k", "k", "0", None),
            ("avg_ex", "EX", PCT, 0.01), ("avg_esm", "ESM", PCT, 0.01), ("avg_prompt_tokens", "T_in", NUM1, None),
            ("avg_output_tokens", "T_out", NUM1, None), ("avg_token_consumption", "Token Consumption", NUM1, None),
            ("n_samples", "n query", "0", None), ("tables_pct", "Batas tabel (pct)", "0.00", None),
            ("columns_pct", "Batas kolom (pct)", "0.00", None), ("llm_model", "Model LLM", None, None)]
    _df_table(p, df.sort_values(["few_shot_selection", "mode", "k"]), spec)
    p.finish()
    return p


def sheet_models(wb, df):
    p = Page(wb, "Perbandingan Model", "Ablation Tahap 3: Perbandingan Model LLM",
             "Satu baris per model dan pipeline. EX/ESM in-process; T = T_in + α × T_out.")
    if df is None:
        p.text("outputs/tables/model_comparison.csv belum ada. Jalankan pipeline.py --baseline --models ...", fill=AMBER)
        return p
    p.caption("Ringkasan per model")
    models = list(dict.fromkeys(df["llm_model"]))
    rows = []
    for m in models:
        sub = df[df["llm_model"] == m].set_index("mode")
        g = lambda mode, col: _num(sub[col].get(mode)) if mode in sub.index else None
        div = lambda x: x / 100 if x is not None else None
        rows.append([m.split("/")[-1], div(g("baseline", "ex")), div(g("graphrag", "ex")), "=(C{r}-B{r})*100",
                     g("baseline", "avg_token_consumption"), g("graphrag", "avg_token_consumption"),
                     '=IF(OR(E{r}="",F{r}="",E{r}=0),"",(E{r}-F{r})/E{r})'])
    f, l = p.table(["Model", "EX Baseline", "EX GraphRAG", "Selisih EX", "T Baseline", "T GraphRAG", "Penghematan token"],
                   rows, [None, PCT, PCT, PP, NUM1, NUM1, PCT])
    p.ws.conditional_formatting.add(
        f"B{f}:B{l}", FormulaRule(formula=[f"B{f}=MAX($B${f}:$C${l})"], fill=PatternFill("solid", bgColor=GREEN)))
    p.ws.conditional_formatting.add(
        f"C{f}:C{l}", FormulaRule(formula=[f"C{f}=MAX($B${f}:$C${l})"], fill=PatternFill("solid", bgColor=GREEN)))
    p.text("Sel hijau = EX tertinggi di antara semua model dan pipeline.", italic=True, color="595959")
    p.gap()
    p.caption("Detail")
    spec = [("llm_model", "Model", None, None), ("mode", "Pipeline", None, None), ("n", "n query", "0", None),
            ("ex", "EX", PCT, 0.01), ("esm", "ESM", PCT, 0.01), ("avg_t_in", "T_in", NUM1, None), ("avg_t_out", "T_out", NUM1, None),
            ("avg_token_consumption", "Token Consumption", NUM1, None), ("few_shot_k", "k few-shot", "0", None),
            ("few_shot_selection", "Seleksi few-shot", None, None), ("tables_pct", "Batas tabel (pct)", "0.00", None),
            ("columns_pct", "Batas kolom (pct)", "0.00", None)]
    _df_table(p, df, spec)
    p.finish()
    return p


TRIAL_SPEC = [("trial", "Trial", None, None), ("table_threshold", "Threshold tabel (T)", "0.00", None),
              ("column_threshold", "Threshold kolom (K)", "0.00", None), ("stopword", "Stopword", None, None),
              ("value_link", "Value linking", None, None), ("traversal", "Traversal", None, None),
              ("no_pk", "Tabel tanpa PK", None, None), ("nama_embed", "Nama yang di-embed", None, None),
              ("tbl_P", "Tabel P", PCT, None), ("tbl_R", "Tabel R", PCT, None), ("col_P", "Kolom P", PCT, None),
              ("col_R", "Kolom R", PCT, None), ("col_P_ne", "Kolom P (gold tak kosong)", PCT, None),
              ("col_R_ne", "Kolom R (gold tak kosong)", PCT, None), ("cols_sent", "Kolom terkirim / query", NUM1, None),
              ("col_reduction", "Penghematan kolom vs seluruh DB", PCT, None), ("n", "n query", "0", None),
              ("max_ngram", "max_ngram", "0", None)]


def sheet_trial(wb, df):
    p = Page(wb, "SLA Trial", "Trial Schema Linking GraphRAG (SLA)",
             "Setiap baris = satu konfigurasi preprocessing (G1-G7) pada satu pasangan threshold. P = precision, R = recall (SLA resmi). "
             "'gold tak kosong' = tanpa 42 query bergold kosong yang selalu bernilai 0 di SLA resmi (diagnostik).")
    if df is None:
        p.text("outputs/tables/sla_trial_recap.csv belum ada. Jalankan graphrag_sla_trial.ipynb sampai rekap bagian 9.", fill=AMBER)
        return p
    rp = p.kv("Target precision", 0.50, PCT, "Baris hijau = kolom P dan R (gold tak kosong) memenuhi KEDUA target.", input_cell=True)
    rr = p.kv("Target recall", 0.90, PCT, "Ubah angka biru untuk menyorot ulang.", input_cell=True)
    p.gap()
    (f, l), spec = _df_table(p, df, TRIAL_SPEC, wrap_cols=(1,), min_widths={1: 40})
    keys = [s[0] for s in spec]
    cp, cr = get_column_letter(keys.index("col_P_ne") + 1), get_column_letter(keys.index("col_R_ne") + 1)
    last_col = get_column_letter(len(spec))
    p.ws.conditional_formatting.add(
        f"A{f}:{last_col}{l}", FormulaRule(formula=[f"AND(${cp}{f}>=$B${rp},${cr}{f}>=$B${rr})"], fill=PatternFill("solid", bgColor=GREEN)))
    p.finish(freeze=f"B{f}")
    return p


def sheet_configs(wb, frames: list[pd.DataFrame]):
    p = Page(wb, "SLA Konfigurasi", "SLA per Konfigurasi: GraphRAG vs Baseline",
             "Satu baris per pipeline per pasangan threshold (dari sla_trial_T*_K*.csv). Dipakai membandingkan recall tabel Baseline "
             "dan GraphRAG di threshold tabel yang sama untuk memilih T bersama.")
    df = pd.concat(frames)
    df.index.name = "pipeline"
    df = df.reset_index()
    df = df.sort_values(["table_threshold", "column_threshold", "pipeline"])
    spec = [("pipeline", "Pipeline", None, None), ("table_threshold", "Threshold tabel (T)", "0.00", None),
            ("column_threshold", "Threshold kolom (K)", "0.00", None), ("max_ngram", "max_ngram", "0", None),
            ("tbl_P", "Tabel P", PCT, None), ("tbl_R", "Tabel R", PCT, None), ("tbl_F1", "Tabel F1", PCT, None),
            ("col_P", "Kolom P", PCT, None), ("col_R", "Kolom R", PCT, None), ("col_F1", "Kolom F1", PCT, None),
            ("col_P_ne", "Kolom P (gold tak kosong)", PCT, None), ("col_R_ne", "Kolom R (gold tak kosong)", PCT, None),
            ("cols_sent", "Kolom terkirim / query", NUM1, None), ("col_reduction", "Penghematan kolom", PCT, None),
            ("n", "n query", "0", None)]
    (f, l), _ = _df_table(p, df, spec)
    p.finish(freeze=f"B{f}")
    return p


# ---------------------------------------------------------------------------
# Ringkasan
# ---------------------------------------------------------------------------

def _findings(dims: dict | None) -> list[list]:
    """Satu baris per dimensi: judul, status, kesimpulan (teks interpretasi dari dimensions_results.json)."""
    titles = {1: "D1 Efisiensi Token & Performa", 2: "D2 Akurasi Struktur SQL", 3: "D3 Akurasi Komponen SQL",
              4: "D4 Robustness (QVT)", 5: "D5 Bottleneck Retrieval vs Generation", 6: "D6 Ablation Few-Shot"}
    out = []
    for n in range(1, 7):
        rec = _dim_record(dims, n)
        if rec is None:
            out.append([titles[n], "Belum dijalankan", ""])
            continue
        if rec.get("status") != "ok":
            out.append([titles[n], rec.get("status"), rec.get("reason", "")])
            continue
        r = rec["result"]
        if n == 1:
            t = r["tep"]
            text = (f"EX Baseline {r['ex']['baseline']['overall']:.1f}% vs GraphRAG {r['ex']['graphrag']['overall']:.1f}%; "
                    f"T {r['token_consumption']['baseline']['overall']['mean']:.0f} vs {r['token_consumption']['graphrag']['overall']['mean']:.0f}; "
                    f"TEP = {t['tep']:.3f}. {t['interpretation']}")
        elif n == 2:
            c = r["classification"]
            text = f"{c['combination']}. {c['interpretation']}"
        elif n == 3:
            text = r["biggest_drop"]["interpretation"]
        elif n == 4:
            text = str(r.get("interpretation", ""))
        elif n == 5:
            g = r["graphrag"]
            text = f"GraphRAG (kolom): {g['column_level']['diagnosis']}. Baseline (kolom): {r['baseline']['column_level']['diagnosis']}."
        else:
            text = f"k final = {r['k_final']} (pola {r['pattern']}). {r['reasoning']}"
        out.append([titles[n], "ok", text])
    return out


GLOSSARY = [
    ("EX", "Execution Accuracy: persentase query yang hasil eksekusi SQL prediksinya sama dengan gold."),
    ("ESM", "Exact Set Match: persentase query yang struktur SQL-nya sama dengan gold (komponen demi komponen)."),
    ("SLA", "Schema Linking Accuracy: precision/recall/F1 schema yang masuk prompt vs schema yang dipakai gold SQL (macro-average per query). "
            "Recall tinggi paling penting, karena kolom yang hilang tidak bisa dipakai LLM."),
    ("T", "Token Consumption = T_in + α × T_out (α = 3.0 dari proposal). Makin rendah makin hemat."),
    ("TEP", "Token Elasticity of Performance: perubahan EX relatif per perubahan token relatif, GraphRAG terhadap Baseline."),
    ("pp", "Poin persentase: selisih dua persentase (65% - 60% = +5 pp)."),
    ("Warna sel", "Huruf biru = angka yang boleh diubah (asumsi/target). Hijau = pemenang/terpenuhi, kuning = perlu perhatian, "
                  "merah = penurunan."),
]


def build_workbook(raw_logs_dir: Path, tables_dir: Path, out: Path) -> Path:
    wb = Workbook()
    summary = Page(wb, "Ringkasan", "Ringkasan Hasil: GraphRAG vs Baseline Text-to-SQL (Spider dev)",
                   f"Dibuat otomatis oleh src/experiments/export_excel.py pada {datetime.now():%Y-%m-%d %H:%M}. "
                   "Semua angka berasal dari file hasil run; tidak ada yang diketik manual.", first=True)

    # --- sumber data
    logs = {c: _read_json(raw_logs_dir / f"{c}_log.json") for c in ("baseline", "graphrag")}
    dims = _read_json(tables_dir / "dimensions_results.json")
    ablation = _read_csv(tables_dir / "ablation_results.csv")
    models = _read_csv(tables_dir / "model_comparison.csv")
    trial = _read_csv(tables_dir / "sla_trial_recap.csv")
    cfg_frames = []
    for f in sorted(glob.glob(str(tables_dir / "sla_trial_T*_K*.csv"))):
        d = pd.read_csv(f, index_col=0)
        if len(d):
            cfg_frames.append(d)

    stats = None
    if logs["baseline"] and logs["graphrag"]:
        stats = {c: _condition_stats(logs[c]) for c in logs}

    # --- sheet berurutan (Ringkasan sudah dibuat paling depan)
    sheet_main(summary, stats, _alpha_default())
    summary.caption("Temuan per dimensi analisis")
    summary.table_text(["Dimensi", "Status", "Kesimpulan"], _findings(dims))
    summary.gap()
    summary.caption("Sumber data yang terbaca")
    sources = [
        ("raw_logs Baseline", logs["baseline"] is not None, str(raw_logs_dir / "baseline_log.json")),
        ("raw_logs GraphRAG", logs["graphrag"] is not None, str(raw_logs_dir / "graphrag_log.json")),
        ("Dimensi 1-6", dims is not None, str(tables_dir / "dimensions_results.json")),
        ("Ablation tahap 2 (few-shot k)", ablation is not None, str(tables_dir / "ablation_results.csv")),
        ("Perbandingan model (tahap 3)", models is not None, str(tables_dir / "model_comparison.csv")),
        ("Trial SLA (rekap)", trial is not None, str(tables_dir / "sla_trial_recap.csv")),
        ("Trial SLA (per konfigurasi)", bool(cfg_frames), str(tables_dir / "sla_trial_T*_K*.csv")),
    ]
    f, l = summary.table_text(["Sumber", "Ditemukan?", "Lokasi"], [[a, "Ya" if ok else "Tidak ada", c] for a, ok, c in sources])
    summary.ws.conditional_formatting.add(f"B{f}:B{l}", FormulaRule(formula=[f'B{f}="Tidak ada"'], fill=PatternFill("solid", bgColor=AMBER)))
    summary.gap()
    summary.caption("Istilah")
    summary.table_text(["Istilah", "Arti"], [list(g) for g in GLOSSARY])

    pages = [summary]
    for build, arg in ((sheet_dim1, dims), (sheet_dim2, dims), (sheet_dim3, dims), (sheet_dim4, dims), (sheet_dim5, dims),
                       (sheet_dim6, dims), (sheet_ablation, ablation), (sheet_models, models), (sheet_trial, trial)):
        pages.append(build(wb, arg))
    if cfg_frames:
        pages.append(sheet_configs(wb, cfg_frames))
    for pg in pages:
        pg.finish()

    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Rangkum hasil run jadi satu file Excel.")
    ap.add_argument("--raw-logs-dir", default="data/raw_logs")
    ap.add_argument("--tables-dir", default="outputs/tables")
    ap.add_argument("--out", default="outputs/tables/hasil_skripsi.xlsx")
    args = ap.parse_args()
    out = build_workbook(Path(args.raw_logs_dir), Path(args.tables_dir), Path(args.out))
    print(f"Excel tersimpan: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
