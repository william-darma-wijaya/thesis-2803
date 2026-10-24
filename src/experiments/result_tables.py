"""
result_tables.py — tabel hasil evaluasi sebagai DataFrame, dipakai notebook DAN export_excel.py.

Satu sumber untuk dua tampilan: `notebooks/eval_pipeline.ipynb` menampilkannya dengan `show()`
(Styler + display), `src/experiments/export_excel.py` menulis DataFrame yang SAMA ke sheet Excel.
Jadi angka di notebook dan di workbook tidak bisa berbeda.

Isinya HANYA angka faktual: tidak ada interpretasi, kesimpulan, atau rekomendasi. Field teks
interpretasi di dimensions_results.json (`interpretation`, `diagnosis`, `note`, `reasoning`,
`pattern`, `k_final`, `combination`, ...) sengaja tidak dibaca di sini.

Tidak ada metrik yang dihitung ulang dengan definisi baru:
- EX/ESM/CM/token dibaca dari raw_logs (nilai evaluator in-process, esm_ex_cm.py).
- SLA = compute_sla() + aggregate_sla() dari src/metrics/sla.py (macro-average per query).
- Tabel dimensi dibaca apa adanya dari hasil run_all_dimensions.py.

Format kolom disimpan di `df.attrs` (lihat KINDS) supaya kedua renderer memakai format yang sama:
    attrs["formats"]   {kolom: kind}
    attrs["row_kind"]  {"cols": [...], "delta": [...]} -> kind per baris dibaca dari kolom "_kind"
                       (untuk tabel yang barisnya beda satuan, mis. perbandingan utama)
"""

import math
import re
from typing import Optional

import pandas as pd

DIFFICULTIES = ["easy", "medium", "hard", "extra"]
DIFF_LABEL = {"easy": "Easy", "medium": "Medium", "hard": "Hard", "extra": "Extra hard", "all": "All"}
CLAUSES = ["select", "from", "where", "group_by", "order_by", "having", "keywords",
           "union", "union_all", "intersect", "except"]
CLAUSE_LABEL = {"select": "SELECT", "from": "FROM", "where": "WHERE", "group_by": "GROUP BY",
                "order_by": "ORDER BY", "having": "HAVING", "keywords": "Keywords",
                "union": "UNION", "union_all": "UNION ALL", "intersect": "INTERSECT", "except": "EXCEPT"}
CONDITIONS = (("baseline", "Baseline"), ("graphrag", "GraphRAG"))

# kind -> (format Python untuk notebook, number_format Excel)
KINDS = {
    "int":   ("{:,.0f}", "#,##0"),
    "num1":  ("{:,.1f}", "#,##0.0"),
    "num2":  ("{:,.2f}", "#,##0.00"),
    "num3":  ("{:.3f}", "0.000"),
    "num4":  ("{:.4f}", "0.0000"),
    "pct":   ("{:.1%}", "0.0%"),
    "pp":    ("{:+.1f} pp", '+0.0" pp";-0.0" pp";0.0" pp"'),
    "rel":   ("{:+.1%}", "+0.0%;-0.0%;0.0%"),
    "dint":  ("{:+,.0f}", "+#,##0;-#,##0;0"),
    "dnum1": ("{:+,.1f}", "+#,##0.0;-#,##0.0;0.0"),
    "dnum3": ("{:+.3f}", "+0.000;-0.000;0.000"),
    "dnum4": ("{:+.4f}", "+0.0000;-0.0000;0.0000"),
    "text":  (None, None),
}
_DELTA_OF = {"pct": "pp", "int": "dint", "num1": "dnum1", "num3": "dnum3", "num4": "dnum4"}


def with_formats(df: pd.DataFrame, formats: Optional[dict] = None, row_kind: Optional[dict] = None,
                  title: Optional[str] = None) -> pd.DataFrame:
    df.attrs["formats"] = formats or {}
    if row_kind:
        df.attrs["row_kind"] = row_kind
    if title:
        df.attrs["title"] = title
    return df


def cell_kind(df: pd.DataFrame, row: int, col: str) -> str:
    """Kind satu sel: dari kolom _kind untuk kolom per-baris, selain itu dari attrs['formats']."""
    rk = df.attrs.get("row_kind")
    if rk and col in rk["cols"] and "_kind" in df.columns:
        kind = df["_kind"].iloc[row]
        return _DELTA_OF.get(kind, kind) if col in rk.get("delta", []) else kind
    if col in df.attrs.get("formats", {}):
        return df.attrs["formats"][col]
    dtype = df[col].dtype
    if pd.api.types.is_bool_dtype(dtype):
        return "text"
    if pd.api.types.is_integer_dtype(dtype):
        return "int"
    if pd.api.types.is_float_dtype(dtype):
        return "num2"
    return "text"


def visible_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if not str(c).startswith("_")]


def is_missing(v) -> bool:
    return v is None or (isinstance(v, float) and math.isnan(v))


def formatted(df: pd.DataFrame) -> pd.DataFrame:
    """Salinan berisi teks terformat (kolom tampil saja); dipakai kalau Styler tidak tersedia."""
    cols = visible_columns(df)
    out = {}
    for col in cols:
        vals = []
        for i in range(len(df)):
            v = df[col].iloc[i]
            pyfmt = KINDS.get(cell_kind(df, i, col), KINDS["text"])[0]
            if is_missing(v):
                vals.append("–")
            elif pyfmt and isinstance(v, (int, float)) and not isinstance(v, bool):
                vals.append(pyfmt.format(v))
            else:
                vals.append(v)
        out[col] = vals
    return pd.DataFrame(out)


def show(df: pd.DataFrame, caption: Optional[str] = None):
    """Tampilkan DataFrame berformat di notebook (tanpa index, NaN -> '–')."""
    from IPython.display import Markdown, display

    caption = caption or df.attrs.get("title")
    cols = visible_columns(df)
    view = df[cols]
    try:
        styler = view.style
    except (AttributeError, ImportError):   # Styler butuh jinja2
        if caption:
            display(Markdown(f"**{caption}**"))
        display(formatted(df))
        return
    styler = styler.hide(axis="index")
    for j, col in enumerate(cols):
        for i in range(len(view)):
            kind = cell_kind(df, i, col)
            pyfmt = KINDS.get(kind, KINDS["text"])[0]
            v = view.iat[i, j]
            if is_missing(v):
                fn = lambda _v: "–"
            elif pyfmt and isinstance(v, (int, float)) and not isinstance(v, bool):
                fn = (lambda f: (lambda x: f.format(x)))(pyfmt)
            else:
                continue
            styler = styler.format(fn, subset=pd.IndexSlice[view.index[i], col])
    styler = styler.set_table_styles([
        {"selector": "th", "props": [("text-align", "left"), ("background-color", "#1F3864"),
                                     ("color", "white"), ("font-weight", "600"), ("padding", "4px 8px")]},
        {"selector": "td", "props": [("padding", "3px 8px")]},
        {"selector": "caption", "props": [("caption-side", "top"), ("text-align", "left"),
                                          ("font-weight", "600"), ("font-size", "1.05em"), ("padding", "6px 0")]},
    ])
    num_cols = [c for c in cols if pd.api.types.is_numeric_dtype(view[c]) and not pd.api.types.is_bool_dtype(view[c])]
    if num_cols:
        styler = styler.set_properties(subset=num_cols, **{"text-align": "right"})
    if caption:
        styler = styler.set_caption(caption)
    display(styler)


# ---------------------------------------------------------------------------
# raw_logs -> tabel
# ---------------------------------------------------------------------------

def _t(entry: dict, alpha: float) -> float:
    return entry["token_input"] + alpha * entry["token_output"]


def _tables_of(schema: list) -> set:
    return {s.split(".", 1)[0] for s in schema}


def _sla(entries: list, level: str):
    from src.metrics.sla import aggregate_sla, compute_sla

    if not entries:
        return None
    return aggregate_sla([compute_sla(set(e["gold_schema"]), set(e["predicted_schema"]), level=level)
                          for e in entries])


def condition_stats(logs: list, alpha: float) -> dict:
    """Ringkasan satu kondisi dari raw_logs. EX/ESM/SLA sebagai pecahan 0-1."""
    n = len(logs)
    if not n:
        return {"n": 0}
    s = pd.Series
    t = s([_t(e, alpha) for e in logs])
    tbl, col = _sla(logs, "table"), _sla(logs, "column")
    ne = [e for e in logs if e["gold_schema"]]
    col_ne = _sla(ne, "column")
    return {
        "n": n,
        "ex": s([e["ex_result"] for e in logs]).mean(),
        "esm": s([e["esm_result"] for e in logs]).mean(),
        "t_in": s([e["token_input"] for e in logs]).mean(),
        "t_out": s([e["token_output"] for e in logs]).mean(),
        "t": t.mean(), "t_median": t.median(),
        "tbl_P": tbl.precision, "tbl_R": tbl.recall, "tbl_F1": tbl.f1,
        "col_P": col.precision, "col_R": col.recall, "col_F1": col.f1,
        "ne_n": len(ne),
        "col_P_ne": col_ne.precision if col_ne else None, "col_R_ne": col_ne.recall if col_ne else None,
        "col_F1_ne": col_ne.f1 if col_ne else None,
        "cols_sent": s([len(e["predicted_schema"]) for e in logs]).mean(),
        "tables_sent": s([len(_tables_of(e["predicted_schema"])) for e in logs]).mean(),
        "gold_cols": s([len(e["gold_schema"]) for e in logs]).mean(),
    }


_MAIN_ROWS = [
    # label, key, kind, relative change ditampilkan?
    ("Queries evaluated", "n", "int", False),
    ("Execution Accuracy (EX)", "ex", "pct", False),
    ("Exact Set Match (ESM)", "esm", "pct", False),
    ("Mean input tokens (T_in)", "t_in", "num1", True),
    ("Mean output tokens (T_out)", "t_out", "num1", True),
    ("Mean token consumption (T)", "t", "num1", True),
    ("Median token consumption (T)", "t_median", "num1", True),
    ("SLA table precision", "tbl_P", "pct", False),
    ("SLA table recall", "tbl_R", "pct", False),
    ("SLA table F1", "tbl_F1", "pct", False),
    ("SLA column precision", "col_P", "pct", False),
    ("SLA column recall", "col_R", "pct", False),
    ("SLA column F1", "col_F1", "pct", False),
    ("Queries with non-empty gold schema", "ne_n", "int", False),
    ("SLA column precision (non-empty gold)", "col_P_ne", "pct", False),
    ("SLA column recall (non-empty gold)", "col_R_ne", "pct", False),
    ("SLA column F1 (non-empty gold)", "col_F1_ne", "pct", False),
    ("Mean tables in prompt", "tables_sent", "num1", True),
    ("Mean columns in prompt", "cols_sent", "num1", True),
    ("Mean gold columns", "gold_cols", "num1", False),
]


def overall_comparison(baseline_logs: list, graphrag_logs: list, alpha: float,
                       stats: Optional[dict] = None) -> pd.DataFrame:
    """Baseline vs GraphRAG, satu baris per metrik. Δ = GraphRAG − Baseline (pp untuk persentase)."""
    stats = stats or {"baseline": condition_stats(baseline_logs, alpha), "graphrag": condition_stats(graphrag_logs, alpha)}
    b, g = stats["baseline"], stats["graphrag"]
    rows = []
    for label, key, kind, show_rel in _MAIN_ROWS:
        vb, vg = b.get(key), g.get(key)
        delta = rel = None
        if not (is_missing(vb) or is_missing(vg)):
            delta = (vg - vb) * (100 if kind == "pct" else 1)
            rel = (vg - vb) / vb if show_rel and vb else None
        rows.append({"Metric": label, "Baseline": vb, "GraphRAG": vg, "Δ (GraphRAG − Baseline)": delta,
                     "Relative change": rel, "_kind": kind})
    df = pd.DataFrame(rows)
    return with_formats(df, {"Relative change": "rel"},
                         {"cols": ["Baseline", "GraphRAG", "Δ (GraphRAG − Baseline)"], "delta": ["Δ (GraphRAG − Baseline)"]},
                         "Overall comparison: Baseline vs GraphRAG")


def by_difficulty(baseline_logs: list, graphrag_logs: list, alpha: float) -> pd.DataFrame:
    """EX, ESM, dan token consumption per tingkat kesulitan (+ baris All)."""
    rows = []
    for d in DIFFICULTIES + ["all"]:
        sel = (lambda logs: logs) if d == "all" else (lambda logs, d=d: [e for e in logs if e["difficulty"] == d])
        b, g = sel(baseline_logs), sel(graphrag_logs)
        if not b and not g:
            continue
        mean = lambda xs: sum(xs) / len(xs) if xs else None
        r = {"Difficulty": DIFF_LABEL[d], "Queries": len(g) or len(b)}
        for metric, key in (("EX", "ex_result"), ("ESM", "esm_result")):
            vb, vg = mean([e[key] for e in b]), mean([e[key] for e in g])
            r[f"{metric} Baseline"], r[f"{metric} GraphRAG"] = vb, vg
            r[f"Δ {metric}"] = None if vb is None or vg is None else (vg - vb) * 100
        tb, tg = mean([_t(e, alpha) for e in b]), mean([_t(e, alpha) for e in g])
        r["T Baseline"], r["T GraphRAG"] = tb, tg
        r["Δ T"] = None if tb is None or tg is None else tg - tb
        rows.append(r)
    df = pd.DataFrame(rows)
    return with_formats(df, {"Queries": "int", "EX Baseline": "pct", "EX GraphRAG": "pct", "Δ EX": "pp",
                              "ESM Baseline": "pct", "ESM GraphRAG": "pct", "Δ ESM": "pp",
                              "T Baseline": "num1", "T GraphRAG": "num1", "Δ T": "dnum1"},
                         title="Metrics by difficulty")


def retrieval_metrics(baseline_logs: list, graphrag_logs: list) -> pd.DataFrame:
    """SLA resmi (sla.py) per pipeline × level × populasi query."""
    rows = []
    for cond, label in CONDITIONS:
        logs = {"baseline": baseline_logs, "graphrag": graphrag_logs}[cond]
        ne = [e for e in logs if e["gold_schema"]]
        for level in ("table", "column"):
            for pop, entries in (("All queries", logs), ("Non-empty gold schema", ne)):
                r = _sla(entries, level)
                if r is None:
                    continue
                if level == "table":
                    sent = sum(len(_tables_of(e["predicted_schema"])) for e in entries) / len(entries)
                    gold = sum(len(_tables_of(e["gold_schema"])) for e in entries) / len(entries)
                else:
                    sent = sum(len(e["predicted_schema"]) for e in entries) / len(entries)
                    gold = sum(len(e["gold_schema"]) for e in entries) / len(entries)
                rows.append({"Pipeline": label, "Level": level.capitalize(), "Population": pop,
                             "Queries": len(entries), "Precision": r.precision, "Recall": r.recall, "F1": r.f1,
                             "Mean items in prompt": sent, "Mean gold items": gold})
    df = pd.DataFrame(rows)
    return with_formats(df, {"Queries": "int", "Precision": "pct", "Recall": "pct", "F1": "pct",
                              "Mean items in prompt": "num1", "Mean gold items": "num1"},
                         title="Schema linking accuracy (SLA)")


def retrieval_by_difficulty(baseline_logs: list, graphrag_logs: list) -> pd.DataFrame:
    """SLA level kolom per tingkat kesulitan."""
    rows = []
    for d in DIFFICULTIES:
        r = {"Difficulty": DIFF_LABEL[d]}
        for cond, label in CONDITIONS:
            logs = [e for e in {"baseline": baseline_logs, "graphrag": graphrag_logs}[cond] if e["difficulty"] == d]
            s = _sla(logs, "column")
            r["Queries"] = len(logs)
            for m in ("Precision", "Recall", "F1"):
                r[f"Column {m} {label}"] = getattr(s, m.lower()) if s else None
        if r["Queries"]:
            rows.append(r)
    df = pd.DataFrame(rows)
    return with_formats(df, {c: ("int" if c == "Queries" else "pct") for c in df.columns if c != "Difficulty"},
                         title="Column-level SLA by difficulty")


def outcome_matrix(baseline_logs: list, graphrag_logs: list) -> pd.DataFrame:
    """Jumlah query per kombinasi EX × ESM, per pipeline."""
    rows = []
    for cond, label in CONDITIONS:
        logs = {"baseline": baseline_logs, "graphrag": graphrag_logs}[cond]
        n = len(logs)
        for ex in (1, 0):
            for esm in (1, 0):
                c = sum(1 for e in logs if e["ex_result"] == ex and e["esm_result"] == esm)
                rows.append({"Pipeline": label, "EX": ex, "ESM": esm, "Queries": c, "Share": c / n if n else None})
    df = pd.DataFrame(rows)
    return with_formats(df, {"EX": "int", "ESM": "int", "Queries": "int", "Share": "pct"},
                         title="EX × ESM outcome counts")


def paired_ex(baseline_logs: list, graphrag_logs: list) -> pd.DataFrame:
    """EX berpasangan per query_id (hanya query yang ada di kedua raw_logs)."""
    b = {e["query_id"]: e["ex_result"] for e in baseline_logs}
    g = {e["query_id"]: e["ex_result"] for e in graphrag_logs}
    ids = [q for q in g if q in b]
    cells = [("Both correct", 1, 1), ("GraphRAG only", 0, 1), ("Baseline only", 1, 0), ("Both wrong", 0, 0)]
    rows = []
    for label, vb, vg in cells:
        c = sum(1 for q in ids if b[q] == vb and g[q] == vg)
        rows.append({"Outcome (EX)": label, "Baseline EX": vb, "GraphRAG EX": vg, "Queries": c,
                     "Share": c / len(ids) if ids else None})
    df = pd.DataFrame(rows)
    return with_formats(df, {"Baseline EX": "int", "GraphRAG EX": "int", "Queries": "int", "Share": "pct"},
                         title="Paired EX per query")


_SQL_FEATURES = [
    ("JOIN", r"\bJOIN\b"),
    ("Outer JOIN (LEFT/RIGHT/FULL)", r"\b(LEFT|RIGHT|FULL)\s+(OUTER\s+)?JOIN\b"),
    ("WHERE", r"\bWHERE\b"),
    ("GROUP BY", r"\bGROUP\s+BY\b"),
    ("HAVING", r"\bHAVING\b"),
    ("ORDER BY", r"\bORDER\s+BY\b"),
    ("LIMIT", r"\bLIMIT\b"),
    ("DISTINCT", r"\bDISTINCT\b"),
    ("Nested SELECT", None),
    ("UNION", r"\bUNION\b"),
    ("INTERSECT", r"\bINTERSECT\b"),
    ("EXCEPT", r"\bEXCEPT\b"),
]


def _has_feature(sql: str, pattern: Optional[str]) -> bool:
    sql = sql or ""
    if pattern is None:   # SELECT di dalam kurung
        return bool(re.search(r"\(\s*SELECT\b", sql, re.IGNORECASE))
    return bool(re.search(pattern, sql, re.IGNORECASE))


def sql_feature_usage(baseline_logs: list, graphrag_logs: list) -> pd.DataFrame:
    """Persentase query yang memuat fitur SQL tertentu: gold vs prediksi tiap pipeline (pencocokan teks)."""
    gold = graphrag_logs or baseline_logs
    rows = []
    for label, pat in _SQL_FEATURES:
        share = lambda logs, key: (sum(_has_feature(e[key], pat) for e in logs) / len(logs)) if logs else None
        rows.append({"SQL feature": label, "Gold": share(gold, "gold_sql"),
                     "Baseline prediction": share(baseline_logs, "predicted_sql"),
                     "GraphRAG prediction": share(graphrag_logs, "predicted_sql")})
    df = pd.DataFrame(rows)
    return with_formats(df, {"Gold": "pct", "Baseline prediction": "pct", "GraphRAG prediction": "pct"},
                         title="SQL feature usage (share of queries)")


def prediction_checks(baseline_logs: list, graphrag_logs: list) -> pd.DataFrame:
    """Hitungan prediksi yang perlu dicek manual: fallback, kosong, identik dengan gold."""
    norm = lambda s: re.sub(r"\s+", " ", (s or "").strip().rstrip(";")).lower()
    rows = []
    for cond, label in CONDITIONS:
        logs = {"baseline": baseline_logs, "graphrag": graphrag_logs}[cond]
        rows.append({
            "Pipeline": label, "Queries": len(logs),
            "Fallback 'SELECT 1'": sum(norm(e["predicted_sql"]) == "select 1" for e in logs),
            "Empty prediction": sum(not (e["predicted_sql"] or "").strip() for e in logs),
            "Identical to gold (text)": sum(norm(e["predicted_sql"]) == norm(e["gold_sql"]) for e in logs),
            "EX=1": sum(e["ex_result"] for e in logs),
            "ESM=1": sum(e["esm_result"] for e in logs),
        })
    df = pd.DataFrame(rows)
    return with_formats(df, {c: "int" for c in df.columns if c != "Pipeline"}, title="Prediction checks")


def cm_from_logs(baseline_logs: list, graphrag_logs: list) -> pd.DataFrame:
    """Component Match per klausa dari raw_logs (definisi aggregate_cm: rata-rata nilai non-None)."""
    def agg(logs, c):
        vals = [e["cm_per_clause"].get(c) for e in logs if e["cm_per_clause"].get(c) is not None]
        return sum(vals) / len(vals) if vals else None
    rows = []
    for c in CLAUSES:
        vb, vg = agg(baseline_logs, c), agg(graphrag_logs, c)
        rows.append({"Clause": CLAUSE_LABEL[c], "CM Baseline": vb, "CM GraphRAG": vg,
                     "Δ CM": None if vb is None or vg is None else (vg - vb) * 100})
    df = pd.DataFrame(rows)
    return with_formats(df, {"CM Baseline": "pct", "CM GraphRAG": "pct", "Δ CM": "pp"},
                         title="Component match (CM) per clause")


def per_query(baseline_logs: list, graphrag_logs: list, alpha: float) -> pd.DataFrame:
    """Satu baris per query: SQL, EX/ESM, token, dan SLA kolom kedua pipeline."""
    from src.metrics.sla import compute_sla

    b = {e["query_id"]: e for e in baseline_logs}
    rows = []
    for g in graphrag_logs:
        bb = b.get(g["query_id"])
        r = {"Query ID": g["query_id"], "DB": g["db_id"], "Difficulty": DIFF_LABEL.get(g["difficulty"], g["difficulty"]),
             "Gold SQL": g["gold_sql"],
             "Baseline SQL": bb["predicted_sql"] if bb else None, "GraphRAG SQL": g["predicted_sql"]}
        for label, e in (("Baseline", bb), ("GraphRAG", g)):
            r[f"EX {label}"] = e["ex_result"] if e else None
            r[f"ESM {label}"] = e["esm_result"] if e else None
        for label, e in (("Baseline", bb), ("GraphRAG", g)):
            r[f"T_in {label}"] = e["token_input"] if e else None
            r[f"T_out {label}"] = e["token_output"] if e else None
            r[f"T {label}"] = _t(e, alpha) if e else None
        r["Gold columns"] = len(g["gold_schema"])
        for label, e in (("Baseline", bb), ("GraphRAG", g)):
            if e is None:
                r[f"Columns sent {label}"] = r[f"Column recall {label}"] = r[f"Column precision {label}"] = None
                continue
            s = compute_sla(set(e["gold_schema"]), set(e["predicted_schema"]), level="column")
            r[f"Columns sent {label}"] = len(e["predicted_schema"])
            r[f"Column recall {label}"] = s.recall
            r[f"Column precision {label}"] = s.precision
        rows.append(r)
    df = pd.DataFrame(rows)
    fmts = {c: "int" for c in df.columns if c.startswith(("EX ", "ESM ", "T_in ", "T_out ", "Columns sent", "Gold columns"))}
    fmts.update({c: "num1" for c in df.columns if c.startswith("T ")})
    fmts.update({c: "pct" for c in df.columns if c.startswith(("Column recall", "Column precision"))})
    return with_formats(df, fmts, title="Per-query results")


# ---------------------------------------------------------------------------
# Confound diagnostics (dict dari pipeline._new_confound_diag)
# ---------------------------------------------------------------------------

def confound_table(diag: dict) -> pd.DataFrame:
    """Counter confound diagnostics (read-only, poin 20) per pipeline."""
    rows = [{"Diagnostic": "Samples", "Gold": diag["n_samples"], "Baseline": None, "GraphRAG": None},
            {"Diagnostic": "Queries with outer JOIN", "Gold": diag["outer_join_gold"],
             "Baseline": diag["outer_join_pred_baseline"], "GraphRAG": diag["outer_join_pred_graphrag"]},
            {"Diagnostic": "Outer-JOIN predictions with ESM=1 and EX=0", "Gold": None,
             "Baseline": diag["outer_join_norm_masked_baseline"], "GraphRAG": diag["outer_join_norm_masked_graphrag"]},
            {"Diagnostic": "Official EX=0, order-insensitive row match", "Gold": None,
             "Baseline": diag["ex0_orderinsensitive_match_baseline"], "GraphRAG": diag["ex0_orderinsensitive_match_graphrag"]},
            {"Diagnostic": "  of which gold has ORDER BY", "Gold": None,
             "Baseline": diag["ex0_orderinsensitive_match_baseline_gold_orderby"],
             "GraphRAG": diag["ex0_orderinsensitive_match_graphrag_gold_orderby"]},
            {"Diagnostic": "  of which gold has no ORDER BY", "Gold": None,
             "Baseline": diag["ex0_orderinsensitive_match_baseline"] - diag["ex0_orderinsensitive_match_baseline_gold_orderby"],
             "GraphRAG": diag["ex0_orderinsensitive_match_graphrag"] - diag["ex0_orderinsensitive_match_graphrag_gold_orderby"]},
            {"Diagnostic": "Gold SQL not executable (excluded)", "Gold": diag["gold_unexecutable"],
             "Baseline": None, "GraphRAG": None}]
    df = pd.DataFrame(rows)
    return with_formats(df, {"Gold": "int", "Baseline": "int", "GraphRAG": "int"}, title="Confound diagnostics (counts)")


# ---------------------------------------------------------------------------
# Output evaluator resmi Spider (stdout evaluation.py) -> tabel
# ---------------------------------------------------------------------------

_SPIDER_SECTIONS = [("EXECUTION ACCURACY", "Execution accuracy"), ("EXACT MATCHING ACCURACY", "Exact match"),
                    ("PARTIAL MATCHING ACCURACY", "Partial match accuracy"), ("PARTIAL MATCHING RECALL", "Partial match recall"),
                    ("PARTIAL MATCHING F1", "Partial match F1")]


def parse_spider_output(stdout: str) -> pd.DataFrame:
    """Parse print_scores() evaluation.py resmi: kolom Section, Metric, easy..extra, all."""
    levels, section, rows = None, "Count", []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        head = [l for l in line.split() if l]
        if head[:5] == ["easy", "medium", "hard", "extra", "all"]:
            levels = head
            continue
        marker = next((lab for key, lab in _SPIDER_SECTIONS if key in line.upper()), None)
        if marker:
            section = marker
            continue
        if levels is None or len(line) <= 20:
            continue
        name, vals = line[:20].strip(), line[20:].split()
        if len(vals) != len(levels):
            continue
        try:
            nums = [float(v) for v in vals]
        except ValueError:
            continue
        rows.append({"Section": "Count" if name == "count" else section, "Metric": name,
                     **{DIFF_LABEL[l]: v for l, v in zip(levels, nums)}})
    df = pd.DataFrame(rows)
    level_cols = [DIFF_LABEL[l] for l in (levels or [])]
    return with_formats(df, {c: "num3" for c in level_cols}, title="Official Spider evaluation")


def spider_summary(parsed: dict) -> pd.DataFrame:
    """{label: parse_spider_output()} -> count / EX / exact match per difficulty untuk semua pipeline."""
    rows = []
    for label, df in parsed.items():
        if df is None or df.empty:
            continue
        for _, r in df[df["Section"].isin(["Count", "Execution accuracy", "Exact match"])].iterrows():
            rows.append({"Pipeline": label, "Metric": {"count": "Queries", "execution": "Execution accuracy",
                                                        "exact match": "Exact match"}.get(r["Metric"], r["Metric"]),
                         **{DIFF_LABEL[l]: r.get(DIFF_LABEL[l]) for l in DIFFICULTIES + ["all"]},
                         "_kind": "int" if r["Metric"] == "count" else "pct"})
    df = pd.DataFrame(rows).drop_duplicates(subset=["Pipeline", "Metric"]) if rows else pd.DataFrame()
    cols = [DIFF_LABEL[l] for l in DIFFICULTIES + ["all"]]
    return with_formats(df, row_kind={"cols": cols}, title="Official Spider evaluation (subprocess)")


# ---------------------------------------------------------------------------
# dimensions_results.json -> tabel (angka saja)
# ---------------------------------------------------------------------------

DIM_NAMES = {1: "Token efficiency & performance", 2: "SQL structure accuracy", 3: "SQL component accuracy",
             4: "Robustness (QVT)", 5: "Retrieval vs generation (SLA × EX)", 6: "Few-shot ablation"}


def _dim(dims: Optional[dict], n: int) -> Optional[dict]:
    rec = (dims or {}).get(f"dimension_{n}")
    return rec["result"] if rec and rec.get("status") == "ok" else None


def dim_status(dims: Optional[dict]) -> pd.DataFrame:
    rows = []
    for n, name in DIM_NAMES.items():
        rec = (dims or {}).get(f"dimension_{n}")
        if rec is None:
            rows.append({"Dimension": f"D{n}", "Name": name, "Status": "not run", "Reason": ""})
        else:
            rows.append({"Dimension": f"D{n}", "Name": name, "Status": rec.get("status"),
                         "Reason": rec.get("reason") or rec.get("error") or ""})
    return with_formats(pd.DataFrame(rows), title="Dimension status")


def dim1_tables(dims: Optional[dict]) -> list:
    r = _dim(dims, 1)
    if r is None:
        return []
    tc, ex = r["token_consumption"], r["ex"]
    rows = []
    for d in ["overall"] + DIFFICULTIES:
        get = (lambda c: tc[c]["overall"]) if d == "overall" else (lambda c, d=d: tc[c]["by_difficulty"].get(d))
        b, g = get("baseline"), get("graphrag")
        if not b or not g:
            continue
        exb = ex["baseline"]["overall"] if d == "overall" else ex["baseline"]["by_difficulty"].get(d)
        exg = ex["graphrag"]["overall"] if d == "overall" else ex["graphrag"]["by_difficulty"].get(d)
        rows.append({"Difficulty": "All" if d == "overall" else DIFF_LABEL[d],
                     "Mean T Baseline": b["mean"], "Mean T GraphRAG": g["mean"], "Δ mean T": g["mean"] - b["mean"],
                     "Relative change T": (g["mean"] - b["mean"]) / b["mean"] if b["mean"] else None,
                     "Median T Baseline": b["median"], "Median T GraphRAG": g["median"],
                     "EX Baseline": None if exb is None else exb / 100, "EX GraphRAG": None if exg is None else exg / 100,
                     "Δ EX": None if exb is None or exg is None else exg - exb})
    tokens = with_formats(pd.DataFrame(rows), {
        "Mean T Baseline": "num1", "Mean T GraphRAG": "num1", "Δ mean T": "dnum1", "Relative change T": "rel",
        "Median T Baseline": "num1", "Median T GraphRAG": "num1", "EX Baseline": "pct", "EX GraphRAG": "pct", "Δ EX": "pp"},
        title="D1 · Token consumption and EX by difficulty")
    t = r["tep"]
    tep = with_formats(pd.DataFrame([{"ΔEX (pp)": t["delta_ex"], "ΔT (tokens)": t["delta_t"], "TEP": t["tep"]}]),
                        {"ΔEX (pp)": "dnum3", "ΔT (tokens)": "dnum1", "TEP": "dnum4"},
                        title="D1 · Token Elasticity of Performance (TEP)")
    return [tokens, tep]


def dim2_tables(dims: Optional[dict]) -> list:
    r = _dim(dims, 2)
    if r is None:
        return []
    e = r["esm_ex"]
    rows = []
    for d in ["overall"] + DIFFICULTIES:
        get = (lambda c: e[c]["overall"]) if d == "overall" else (lambda c, d=d: e[c]["by_difficulty"].get(d))
        b, g = get("baseline"), get("graphrag")
        if not b or not g:
            continue
        rows.append({"Difficulty": "All" if d == "overall" else DIFF_LABEL[d],
                     "ESM Baseline": b["esm"] / 100, "ESM GraphRAG": g["esm"] / 100, "Δ ESM": g["esm"] - b["esm"],
                     "EX Baseline": b["ex"] / 100, "EX GraphRAG": g["ex"] / 100, "Δ EX": g["ex"] - b["ex"]})
    return [with_formats(pd.DataFrame(rows), {"ESM Baseline": "pct", "ESM GraphRAG": "pct", "Δ ESM": "pp",
                                               "EX Baseline": "pct", "EX GraphRAG": "pct", "Δ EX": "pp"},
                          title="D2 · ESM and EX by difficulty")]


def dim3_tables(dims: Optional[dict]) -> list:
    r = _dim(dims, 3)
    if r is None:
        return []
    cm = r["cm_per_clause"]
    rows = [{"Clause": CLAUSE_LABEL.get(c, c),
             "CM Baseline": None if cm["baseline"].get(c) is None else cm["baseline"][c] / 100,
             "CM GraphRAG": None if cm["graphrag"].get(c) is None else cm["graphrag"][c] / 100,
             "Δ CM": cm.get("delta", {}).get(c)}
            for c in CLAUSES if c in cm["baseline"]]
    return [with_formats(pd.DataFrame(rows), {"CM Baseline": "pct", "CM GraphRAG": "pct", "Δ CM": "pp"},
                          title="D3 · Component match per clause")]


def dim4_tables(dims: Optional[dict]) -> list:
    r = _dim(dims, 4)
    if r is None:
        return []
    rows = []
    for cond, label in CONDITIONS:
        q = r["qvt"][cond]
        rows.append({"Pipeline": label, "QVT": q["qvt"], "Query groups": q["n_total"],
                     "Included (≥1 variant correct)": q["m_included"], "Excluded (all variants wrong)": q["n_excluded"],
                     "Groups with QVT < 1": len(q.get("inconsistent_query_ids") or [])})
    qvt = with_formats(pd.DataFrame(rows), {"QVT": "num4", "Query groups": "int", "Included (≥1 variant correct)": "int",
                                             "Excluded (all variants wrong)": "int", "Groups with QVT < 1": "int"},
                        title="D4 · Query Variance Testing (QVT)")
    delta = with_formats(pd.DataFrame([{"ΔQVT (GraphRAG − Baseline)": r.get("delta_qvt")}]),
                          {"ΔQVT (GraphRAG − Baseline)": "dnum4"}, title="D4 · ΔQVT")
    out = [qvt, delta]
    if r.get("esm_cross"):
        rows = []
        for cond, label in CONDITIONS:
            x = r["esm_cross"].get(cond)
            if not x:
                continue
            rows.append({"Pipeline": label,
                         "QVT = 1, all ESM = 1": x["consistent_esm_ok"], "QVT = 1, some ESM = 0": x["consistent_esm_problem"],
                         "QVT < 1, all ESM = 1": x["inconsistent_esm_ok"], "QVT < 1, some ESM = 0": x["inconsistent_esm_problem"],
                         "Share ESM issue (QVT = 1)": x["esm_problem_rate_consistent"],
                         "Share ESM issue (QVT < 1)": x["esm_problem_rate_inconsistent"],
                         "Variants missing in raw_logs": x["n_missing_in_logs"]})
        df = pd.DataFrame(rows)
        out.append(with_formats(df, {c: ("pct" if c.startswith("Share") else "int") for c in df.columns if c != "Pipeline"},
                                 title="D4 · QVT × ESM cross-tabulation (query groups)"))
    return out


def dim5_tables(dims: Optional[dict]) -> list:
    r = _dim(dims, 5)
    if r is None:
        return []
    rows = []
    for cond, label in CONDITIONS:
        d = r.get(cond)
        if not d:
            continue
        for level, lab in (("table_level", "Table"), ("column_level", "Column")):
            x = d[level]
            rows.append({"Pipeline": label, "SLA level": lab, "EX": d["ex_score"], "SLA F1": x["f1_score_sla"],
                         "SLA precision": x["precision"], "SLA recall": x["recall"],
                         "F1 ≥ 0.80": "Yes" if x["f1_meets_threshold"] else "No",
                         "EX ≥ 0.65": "Yes" if x["ex_meets_threshold"] else "No"})
    return [with_formats(pd.DataFrame(rows), {"EX": "pct", "SLA F1": "pct", "SLA precision": "pct", "SLA recall": "pct"},
                          title="D5 · SLA and EX against the proposal thresholds")]


def dim6_tables(dims: Optional[dict]) -> list:
    r = _dim(dims, 6)
    if r is None:
        return []
    ks = sorted(r["table"], key=lambda k: int(k))
    rows, prev = [], None
    for k in ks:
        v = r["table"][k]
        rows.append({"k (few-shot examples)": int(k), "EX": v / 100, "Δ EX vs previous k": None if prev is None else v - prev})
        prev = v
    return [with_formats(pd.DataFrame(rows), {"k (few-shot examples)": "int", "EX": "pct", "Δ EX vs previous k": "pp"},
                          title="D6 · EX per few-shot k")]


DIM_TABLES = {1: dim1_tables, 2: dim2_tables, 3: dim3_tables, 4: dim4_tables, 5: dim5_tables, 6: dim6_tables}


# ---------------------------------------------------------------------------
# CSV hasil ablation / perbandingan model / trial SLA -> tabel
# ---------------------------------------------------------------------------

def _select(df: pd.DataFrame, spec: list, title: str) -> pd.DataFrame:
    """spec: [(kolom sumber, header, kind, skala)] -> DataFrame baru; kolom yang tidak ada dilewati."""
    out, fmts = {}, {}
    for src, header, kind, scale in spec:
        if src not in df.columns:
            continue
        col = df[src]
        if scale:
            col = pd.to_numeric(col, errors="coerce") * scale
        out[header] = col.values
        if kind:
            fmts[header] = kind
    return with_formats(pd.DataFrame(out), fmts, title=title)


def ablation_tables(df: Optional[pd.DataFrame]) -> list:
    if df is None or df.empty:
        return []
    df = df.copy()
    if "few_shot_selection" not in df.columns:
        df["few_shot_selection"] = "question"
    df["few_shot_selection"] = df["few_shot_selection"].fillna("question")
    out = []
    for sel, sub in df.groupby("few_shot_selection", sort=False):
        piv = sub.pivot_table(index="k", columns="mode", values="avg_ex", aggfunc="last").sort_index()
        rows = []
        for k, rw in piv.iterrows():
            vb, vg = rw.get("baseline"), rw.get("graphrag")
            vals = [v for v in (vb, vg) if not is_missing(v)]
            rows.append({"k": int(k), "EX Baseline": None if is_missing(vb) else vb / 100,
                         "EX GraphRAG": None if is_missing(vg) else vg / 100,
                         "Mean EX (both pipelines)": sum(vals) / len(vals) / 100 if len(vals) == 2 else None})
        out.append(with_formats(pd.DataFrame(rows), {"k": "int", "EX Baseline": "pct", "EX GraphRAG": "pct",
                                                      "Mean EX (both pipelines)": "pct"},
                                 title=f"Ablation stage 2 · EX per k (few-shot selection: {sel})"))
    spec = [("mode", "Pipeline", None, None), ("few_shot_selection", "Few-shot selection", None, None), ("k", "k", "int", None),
            ("avg_ex", "EX", "pct", 0.01), ("avg_esm", "ESM", "pct", 0.01), ("avg_prompt_tokens", "T_in", "num1", None),
            ("avg_output_tokens", "T_out", "num1", None), ("avg_token_consumption", "Token consumption", "num1", None),
            ("avg_recall", "Proxy recall", "pct", 0.01), ("avg_precision", "Proxy precision", "pct", 0.01),
            ("n_samples", "Queries", "int", None), ("tables_pct", "Tables pct", "num2", None),
            ("columns_pct", "Columns pct", "num2", None), ("llm_model", "LLM", None, None)]
    out.append(_select(df.sort_values(["few_shot_selection", "mode", "k"]), spec, "Ablation stage 2 · detail per pipeline and k"))
    return out


def model_tables(df: Optional[pd.DataFrame]) -> list:
    if df is None or df.empty:
        return []
    rows = []
    for m in dict.fromkeys(df["llm_model"]):
        sub = df[df["llm_model"] == m].set_index("mode")
        g = lambda mode, col: sub[col].get(mode) if mode in sub.index and col in sub.columns else None
        exb, exg = g("baseline", "ex"), g("graphrag", "ex")
        tb, tg = g("baseline", "avg_token_consumption"), g("graphrag", "avg_token_consumption")
        rows.append({"Model": str(m).split("/")[-1],
                     "EX Baseline": None if is_missing(exb) else exb / 100, "EX GraphRAG": None if is_missing(exg) else exg / 100,
                     "Δ EX": None if is_missing(exb) or is_missing(exg) else exg - exb,
                     "T Baseline": tb, "T GraphRAG": tg,
                     "Relative change T": (tg - tb) / tb if not (is_missing(tb) or is_missing(tg)) and tb else None})
    summary = with_formats(pd.DataFrame(rows), {"EX Baseline": "pct", "EX GraphRAG": "pct", "Δ EX": "pp",
                                                 "T Baseline": "num1", "T GraphRAG": "num1", "Relative change T": "rel"},
                            title="Model comparison · summary")
    spec = [("llm_model", "Model", None, None), ("mode", "Pipeline", None, None), ("n", "Queries", "int", None),
            ("ex", "EX", "pct", 0.01), ("esm", "ESM", "pct", 0.01), ("avg_t_in", "T_in", "num1", None),
            ("avg_t_out", "T_out", "num1", None), ("avg_token_consumption", "Token consumption", "num1", None),
            ("few_shot_k", "Few-shot k", "int", None), ("few_shot_selection", "Few-shot selection", None, None),
            ("tables_pct", "Tables pct", "num2", None), ("columns_pct", "Columns pct", "num2", None)]
    return [summary, _select(df, spec, "Model comparison · detail")]


_TRIAL_SPEC = [("trial", "Trial", None, None), ("table_threshold", "Table threshold (T)", "num2", None),
               ("column_threshold", "Column threshold (K)", "num2", None), ("stopword", "Stopword mode", None, None),
               ("value_link", "Value linking", None, None), ("traversal", "Traversal", None, None),
               ("no_pk", "Connect no-PK tables", None, None), ("nama_embed", "Embedded names", None, None),
               ("max_ngram", "max_ngram", "int", None),
               ("tbl_P", "Table P", "pct", None), ("tbl_R", "Table R", "pct", None), ("tbl_F1", "Table F1", "pct", None),
               ("col_P", "Column P", "pct", None), ("col_R", "Column R", "pct", None), ("col_F1", "Column F1", "pct", None),
               ("col_P_ne", "Column P (non-empty gold)", "pct", None), ("col_R_ne", "Column R (non-empty gold)", "pct", None),
               ("cols_sent", "Columns in prompt", "num1", None), ("col_reduction", "Column reduction vs full DB", "pct", None),
               ("n", "Queries", "int", None)]


def sla_trial_tables(recap: Optional[pd.DataFrame], configs: Optional[list]) -> list:
    out = []
    if recap is not None and not recap.empty:
        out.append(_select(recap, _TRIAL_SPEC, "SLA trial · recap (G1-G7)"))
    if configs:
        df = pd.concat(configs)
        df.index.name = "pipeline"
        df = df.reset_index().sort_values(["table_threshold", "column_threshold", "pipeline"])
        out.append(_select(df, [("pipeline", "Pipeline", None, None)] + _TRIAL_SPEC[1:3] + _TRIAL_SPEC[8:],
                           "SLA trial · per threshold pair"))
    return out
