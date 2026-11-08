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
# Label tampilan pipeline (keputusan peneliti, poin 41): baseline = 1-stage (GraphRAG 1-Stage), graphrag = 2-stage
# (GraphRAG 2-Stage). Kode, raw_logs, dan CSV tetap memakai baseline/graphrag.
CONDITIONS = (("baseline", "1-stage"), ("graphrag", "2-stage"))

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


# Nama pipeline yang mungkin ada di file lama (CSV evaluator resmi, rekap trial) -> kunci kode
_PIPELINE_ALIASES = {"baseline": "baseline", "graphrag": "graphrag", "1-stage": "baseline", "2-stage": "graphrag",
                     "graphrag 1-stage": "baseline", "graphrag 2-stage": "graphrag"}


def pipeline_label(mode) -> str:
    key = _PIPELINE_ALIASES.get(str(mode).lower())
    return dict(CONDITIONS)[key] if key else mode


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
    """GraphRAG 1-Stage vs GraphRAG 2-Stage, satu baris per metrik. Δ = GraphRAG 2-Stage − GraphRAG 1-Stage (pp untuk persentase)."""
    stats = stats or {"baseline": condition_stats(baseline_logs, alpha), "graphrag": condition_stats(graphrag_logs, alpha)}
    b, g = stats["baseline"], stats["graphrag"]
    rows = []
    for label, key, kind, show_rel in _MAIN_ROWS:
        vb, vg = b.get(key), g.get(key)
        delta = rel = None
        if not (is_missing(vb) or is_missing(vg)):
            delta = (vg - vb) * (100 if kind == "pct" else 1)
            rel = (vg - vb) / vb if show_rel and vb else None
        rows.append({"Metric": label, "1-stage": vb, "2-stage": vg, "Δ (2-stage − 1-stage)": delta,
                     "Relative change": rel, "_kind": kind})
    df = pd.DataFrame(rows)
    return with_formats(df, {"Relative change": "rel"},
                         {"cols": ["1-stage", "2-stage", "Δ (2-stage − 1-stage)"], "delta": ["Δ (2-stage − 1-stage)"]},
                         "Overall comparison: 1-stage vs 2-stage")


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
            r[f"{metric} 1-stage"], r[f"{metric} 2-stage"] = vb, vg
            r[f"Δ {metric}"] = None if vb is None or vg is None else (vg - vb) * 100
        tb, tg = mean([_t(e, alpha) for e in b]), mean([_t(e, alpha) for e in g])
        r["T 1-stage"], r["T 2-stage"] = tb, tg
        r["Δ T"] = None if tb is None or tg is None else tg - tb
        rows.append(r)
    df = pd.DataFrame(rows)
    return with_formats(df, {"Queries": "int", "EX 1-stage": "pct", "EX 2-stage": "pct", "Δ EX": "pp",
                              "ESM 1-stage": "pct", "ESM 2-stage": "pct", "Δ ESM": "pp",
                              "T 1-stage": "num1", "T 2-stage": "num1", "Δ T": "dnum1"},
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


def token_reduction(baseline_logs: list, graphrag_logs: list, alpha: float) -> list:
    """
    Column reduction vs token reduction vs EX, GraphRAG 2-Stage terhadap GraphRAG 1-Stage di query yang sama
    (src/metrics/token_reduction.py, IMPLEMENTATION_DECISIONS.md poin 33). Reduction = 1 − ΣG/ΣB,
    positif = GraphRAG 2-Stage lebih hemat. Baris token schema/non-schema butuh raw_logs dengan field
    `schema_tokens` (run sesudah 2026-11-03); log lama hanya mengisi baris kolom dan T_in.
    """
    from src.metrics import token_reduction as tr

    pairs = tr.pair_logs(baseline_logs, graphrag_logs)
    if not pairs:
        return []
    summ = tr.reduction_summary(pairs, alpha)
    rows = [{"Quantity": v["label"], "1-stage (mean)": v["baseline_mean"], "2-stage (mean)": v["graphrag_mean"],
             "Reduction (1 − Σ2-stage/Σ1-stage)": v["reduction"], "Median per-query reduction": v["median_reduction"],
             "Queries": v["n"]} for v in summ.values()]
    main = with_formats(pd.DataFrame(rows), {"1-stage (mean)": "num1", "2-stage (mean)": "num1",
                                             "Reduction (1 − Σ2-stage/Σ1-stage)": "rel", "Median per-query reduction": "rel",
                                             "Queries": "int"},
                        title="Column & token reduction: 2-stage vs 1-stage (paired, same queries)")

    bs, gs = [b for b, _ in pairs], [g for _, g in pairs]
    ex_b, ex_g = sum(e["ex_result"] for e in bs) / len(bs), sum(e["ex_result"] for e in gs) / len(gs)
    facts = [("Queries paired", len(pairs), "int"), ("EX 1-stage", ex_b, "pct"), ("EX 2-stage", ex_g, "pct"),
             ("Δ EX (2-stage − 1-stage)", (ex_g - ex_b) * 100, "pp")]
    if "schema_tokens" in summ:
        sb = sum(e["schema_tokens"] for e in bs) / sum(e["token_input"] for e in bs)
        facts.append(("Schema share of 1-stage T_in (Σschema / ΣT_in)", sb, "pct"))
    dec = tr.column_decomposition(pairs)
    if dec:
        facts += [("Column reduction: table-selection effect", dec["table_selection"], "rel"),
                  ("Column reduction: column-pruning effect", dec["column_pruning"], "rel")]
    for label, logs in (("1-stage", bs), ("2-stage", gs)):
        rate = tr.full_schema_rate(logs)
        if rate is not None:
            facts.append((f"Prompts containing the whole database ({label})", rate, "pct"))
    detail = pd.DataFrame([{"Metric": m, "Value": v, "_kind": k} for m, v, k in facts])
    detail = with_formats(detail, row_kind={"cols": ["Value"]}, title="Reduction context: EX, schema share, decomposition")
    return [main, detail]


def token_reduction_by_difficulty(baseline_logs: list, graphrag_logs: list) -> pd.DataFrame:
    """Column reduction, schema-token reduction, T_in reduction, dan Δ EX per tingkat kesulitan."""
    from src.metrics.token_reduction import pair_logs, total_reduction

    pairs = pair_logs(baseline_logs, graphrag_logs)
    rows = []
    for d in DIFFICULTIES + ["all"]:
        sel = [(b, g) for b, g in pairs if d == "all" or g["difficulty"] == d]
        if not sel:
            continue
        red = lambda get: total_reduction([get(g) for _, g in sel], [get(b) for b, _ in sel])
        has_schema = all("schema_tokens" in b and "schema_tokens" in g for b, g in sel)
        ex_b = sum(b["ex_result"] for b, _ in sel) / len(sel)
        ex_g = sum(g["ex_result"] for _, g in sel) / len(sel)
        rows.append({"Difficulty": DIFF_LABEL[d], "Queries": len(sel),
                     "Column reduction": red(lambda e: len(e["predicted_schema"])),
                     "Schema-token reduction": red(lambda e: e["schema_tokens"]) if has_schema else None,
                     "T_in reduction": red(lambda e: e["token_input"]),
                     "EX 1-stage": ex_b, "EX 2-stage": ex_g, "Δ EX": (ex_g - ex_b) * 100})
    return with_formats(pd.DataFrame(rows), {"Queries": "int", "Column reduction": "rel",
                                             "Schema-token reduction": "rel", "T_in reduction": "rel",
                                             "EX 1-stage": "pct", "EX 2-stage": "pct", "Δ EX": "pp"},
                        title="Column & token reduction by difficulty")


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
    cells = [("Both correct", 1, 1), ("2-stage only", 0, 1), ("1-stage only", 1, 0), ("Both wrong", 0, 0)]
    rows = []
    for label, vb, vg in cells:
        c = sum(1 for q in ids if b[q] == vb and g[q] == vg)
        rows.append({"Outcome (EX)": label, "1-stage EX": vb, "2-stage EX": vg, "Queries": c,
                     "Share": c / len(ids) if ids else None})
    df = pd.DataFrame(rows)
    return with_formats(df, {"1-stage EX": "int", "2-stage EX": "int", "Queries": "int", "Share": "pct"},
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
                     "1-stage prediction": share(baseline_logs, "predicted_sql"),
                     "2-stage prediction": share(graphrag_logs, "predicted_sql")})
    df = pd.DataFrame(rows)
    return with_formats(df, {"Gold": "pct", "1-stage prediction": "pct", "2-stage prediction": "pct"},
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


# ---------------------------------------------------------------------------
# metrics_results.json -> tabel per metrik (1-Stage = "baseline", 2-Stage = "graphrag" di kode)
# ---------------------------------------------------------------------------

_STAGES = (("baseline", "1-stage"), ("graphrag", "2-stage"))


def _diff_rows(values: dict, kind_key: str = "by_difficulty") -> list:
    """Baris per difficulty + All untuk metrik berbentuk {overall, by_difficulty} per kondisi."""
    rows = []
    for d in DIFFICULTIES + ["all"]:
        vb, vg = (values[c]["overall"] if d == "all" else values[c][kind_key].get(d) for c, _ in _STAGES)
        if vb is None and vg is None:
            continue
        rows.append({"Difficulty": DIFF_LABEL[d], "1-stage": vb, "2-stage": vg,
                     "Δ (2-stage − 1-stage)": None if vb is None or vg is None else (vg - vb) * 100})
    return rows


def _accuracy_table(values: dict, title: str) -> pd.DataFrame:
    return with_formats(pd.DataFrame(_diff_rows(values)),
                        {"1-stage": "pct", "2-stage": "pct", "Δ (2-stage − 1-stage)": "pp"}, title=title)


def _sla_tables(sla: dict) -> list:
    rows, diff_rows = [], []
    for key, level, population in (("table", "Table", "All queries"), ("column", "Column", "All queries"),
                                   ("column_non_empty_gold", "Column", "Non-empty gold schema")):
        for c, label in _STAGES:
            r = sla[c][key]
            if r:
                rows.append({"Variant": label, "Level": level, "Population": population,
                             "Precision": r["precision"], "Recall": r["recall"], "F1": r["f1"]})
    for d in DIFFICULTIES:
        row = {"Difficulty": DIFF_LABEL[d]}
        for c, label in _STAGES:
            r = sla[c]["column_by_difficulty"].get(d) or {}
            for m in ("precision", "recall", "f1"):
                row[f"{m.capitalize()} {label}"] = r.get(m)
        if any(not is_missing(v) for k, v in row.items() if k != "Difficulty"):
            diff_rows.append(row)
    pct = {"Precision": "pct", "Recall": "pct", "F1": "pct"}
    diff_formats = {k: "pct" for row in diff_rows for k in row if k != "Difficulty"}
    return [with_formats(pd.DataFrame(rows), pct, title="SLA: precision, recall, F1"),
            with_formats(pd.DataFrame(diff_rows), diff_formats, title="SLA column level by difficulty")]


def _cm_table(cm: dict) -> pd.DataFrame:
    rows = []
    for clause in CLAUSES:
        vb, vg = cm["baseline"].get(clause), cm["graphrag"].get(clause)
        rows.append({"Clause": CLAUSE_LABEL[clause], "1-stage": vb, "2-stage": vg,
                     "Δ (2-stage − 1-stage)": None if vb is None or vg is None else (vg - vb) * 100})
    return with_formats(pd.DataFrame(rows), {"1-stage": "pct", "2-stage": "pct", "Δ (2-stage − 1-stage)": "pp"},
                        title="CM per clause")


def _token_tep_tables(block: dict) -> list:
    rows = []
    for key, label in (("t_in", "Mean input tokens (T_in)"), ("t_out", "Mean output tokens (T_out)"),
                       ("t_mean", "Mean token consumption (T)"), ("t_median", "Median token consumption (T)")):
        vb, vg = block["token"]["baseline"].get(key), block["token"]["graphrag"].get(key)
        rows.append({"Measure": label, "1-stage": vb, "2-stage": vg,
                     "Δ (2-stage − 1-stage)": None if is_missing(vb) or is_missing(vg) else vg - vb,
                     "Relative change": (vg - vb) / vb if not is_missing(vb) and not is_missing(vg) and vb else None})
    diff_rows = []
    for d in DIFFICULTIES:
        vb = block["token"]["baseline"]["t_mean_by_difficulty"].get(d)
        vg = block["token"]["graphrag"]["t_mean_by_difficulty"].get(d)
        if vb is not None or vg is not None:
            diff_rows.append({"Difficulty": DIFF_LABEL[d], "T 1-stage": vb, "T 2-stage": vg,
                              "Δ T": None if vb is None or vg is None else vg - vb})
    tep = block["tep"]
    tep_row = {"Status": tep["status"], "ΔEX (pp)": tep["delta_ex"] * 100 if "delta_ex" in tep else None,
               "ΔT": tep.get("delta_t"), "TEP": tep.get("tep"), "Reason": tep.get("reason", "")}
    return [with_formats(pd.DataFrame(rows), {"1-stage": "num1", "2-stage": "num1", "Δ (2-stage − 1-stage)": "dnum1",
                                              "Relative change": "rel"}, title="Token consumption"),
            with_formats(pd.DataFrame(diff_rows), {"T 1-stage": "num1", "T 2-stage": "num1", "Δ T": "dnum1"},
                         title="Mean token consumption by difficulty"),
            with_formats(pd.DataFrame([tep_row]), {"ΔEX (pp)": "pp", "ΔT": "dnum1", "TEP": "num4"},
                         title="TEP (2-stage vs 1-stage)")]


def _qvt_table(qvt: dict) -> pd.DataFrame:
    rows = [{"Variant": label, "Status": qvt[c]["status"], "QVT": qvt[c].get("qvt"),
             "Queries included": qvt[c].get("queries_included"), "Queries total": qvt[c].get("queries_total"),
             "Reason": qvt[c].get("reason", "")} for c, label in _STAGES]
    return with_formats(pd.DataFrame(rows), {"QVT": "pct", "Queries included": "int", "Queries total": "int"}, title="QVT")


def metrics_summary_table(results: dict) -> pd.DataFrame:
    """Satu tabel berisi keenam metrik. CM tetap satu baris per klausa, tidak dirata-ratakan (guide 1.4)."""
    m = results["metrics"]
    rows = []

    def add(label, vb, vg, kind, note=""):
        delta = None
        if not (is_missing(vb) or is_missing(vg)):
            delta = (vg - vb) * (100 if kind == "pct" else 1)
        rows.append({"Metric": label, "1-stage": vb, "2-stage": vg, "Difference (2-stage − 1-stage)": delta,
                     "Note": note, "_kind": kind})

    for key, label in (("table", "SLA table F1"), ("column", "SLA column F1")):
        b, g = m["sla"]["baseline"][key] or {}, m["sla"]["graphrag"][key] or {}
        add(label, b.get("f1"), g.get("f1"), "pct")
    b, g = m["sla"]["baseline"]["column"] or {}, m["sla"]["graphrag"]["column"] or {}
    add("SLA column precision", b.get("precision"), g.get("precision"), "pct")
    add("SLA column recall", b.get("recall"), g.get("recall"), "pct")
    add("ESM", m["esm"]["baseline"]["overall"], m["esm"]["graphrag"]["overall"], "pct")
    add("EX", m["ex"]["baseline"]["overall"], m["ex"]["graphrag"]["overall"], "pct")
    for clause in CLAUSES:
        vb, vg = m["cm"]["baseline"].get(clause), m["cm"]["graphrag"].get(clause)
        if vb is not None or vg is not None:
            add(f"CM {CLAUSE_LABEL[clause]}", vb, vg, "pct")
    token = m["token_tep"]["token"]
    add("Mean input tokens (T_in)", token["baseline"].get("t_in"), token["graphrag"].get("t_in"), "num1")
    add("Mean output tokens (T_out)", token["baseline"].get("t_out"), token["graphrag"].get("t_out"), "num1")
    add("Mean token consumption (T)", token["baseline"].get("t_mean"), token["graphrag"].get("t_mean"), "num1")
    tep = m["token_tep"]["tep"]
    add("TEP (2-stage vs 1-stage)", None, tep.get("tep"), "num4", tep.get("reason", ""))
    qvt_b, qvt_g = m["qvt"]["baseline"], m["qvt"]["graphrag"]
    add("QVT", qvt_b.get("qvt"), qvt_g.get("qvt"), "pct", qvt_g.get("reason", ""))
    return with_formats(pd.DataFrame(rows), {},
                        {"cols": ["1-stage", "2-stage", "Difference (2-stage − 1-stage)"],
                         "delta": ["Difference (2-stage − 1-stage)"]},
                        "Six metrics: 1-stage vs 2-stage")


def metric_tables(results: dict) -> list:
    """[(nama sheet, judul, [tabel])] untuk keenam metrik, berurutan, dipakai notebook DAN Excel."""
    m = results["metrics"]
    return [
        ("M1 SLA", "Metric 1: Schema Linking Accuracy (SLA)", _sla_tables(m["sla"])),
        ("M2 ESM", "Metric 2: Exact Set Match (ESM)", [_accuracy_table(m["esm"], "ESM by difficulty")]),
        ("M3 EX", "Metric 3: Execution Accuracy (EX)", [_accuracy_table(m["ex"], "EX by difficulty")]),
        ("M4 CM", "Metric 4: Component Match (CM)", [_cm_table(m["cm"])]),
        ("M5 Token TEP", "Metric 5: Token Consumption and TEP", _token_tep_tables(m["token_tep"])),
        ("M6 QVT", "Metric 6: Query Variance Testing (QVT)", [_qvt_table(m["qvt"])]),
    ]


# ---------------------------------------------------------------------------
# Latensi per tahap (lampiran, bukan salah satu dari 6 metrik; poin 37)
# ---------------------------------------------------------------------------

LATENCY_NOTE = ("Appendix, not one of the six metrics. Times are in seconds and depend on the GPU of the run. "
                "The first question of a run is left out of the statistics (model warm-up); failed generations are left out too. "
                "End-to-end excludes proxy evaluation, T_in counting, and SQL evaluation.")
LATENCY_STEPS = {
    "stage1_tables": "Stage 1: select tables", "stage2_columns": "Stage 2: select columns",
    "value_linking": "Value linking", "traversal": "Graph traversal",
    "table_linking": "Select tables", "path_tracing": "Table path tracing",
    "retrieval_search": "Schema search (sum of the steps above)", "build_ddl": "Build DDL",
    "retrieval_total": "Total retrieval (search + DDL)", "few_shot": "Few-shot retrieval", "build_prompt": "Build prompt",
    "prompt_tokenize": "Tokenize prompt for the LLM", "llm_generate": "LLM generation", "sql_clean": "Decode and clean SQL",
    "end_to_end": "End-to-end", "count_t_in": "Count T_in (not in end-to-end)",
    "proxy_eval": "Proxy evaluation (not in end-to-end)", "evaluation": "ESM, EX, CM evaluation (not in end-to-end)",
}
_LATENCY_FORMATS = {"Queries": "int", "Mean (s)": "num3", "Median (s)": "num3", "P95 (s)": "num3"}


def _latency_entries(logs: list) -> list:
    return [e for e in (logs[1:] if len(logs) > 1 else logs) if e.get("latency")]


def _latency_row(values: list) -> dict:
    s = pd.Series(values, dtype=float)
    return {"Queries": len(s), "Mean (s)": s.mean(), "Median (s)": s.median(), "P95 (s)": s.quantile(0.95)}


def _latency_by_step(logs_by_stage: dict) -> pd.DataFrame:
    rows = []
    for c, label in _STAGES:
        entries = _latency_entries(logs_by_stage[c])
        for key, name in LATENCY_STEPS.items():
            values = [e["latency"][key] for e in entries if key in e["latency"]]
            if values:
                rows.append({"Variant": label, "Step": name, **_latency_row(values)})
    return with_formats(pd.DataFrame(rows), _LATENCY_FORMATS, title="Per-question latency by step")


def _latency_by_difficulty(logs_by_stage: dict) -> pd.DataFrame:
    rows = []
    for c, label in _STAGES:
        entries = _latency_entries(logs_by_stage[c])
        for d in DIFFICULTIES:
            subset = [e for e in entries if e["difficulty"] == d]
            for key, name in LATENCY_STEPS.items():
                values = [e["latency"][key] for e in subset if key in e["latency"]]
                if values:
                    rows.append({"Variant": label, "Difficulty": DIFF_LABEL[d], "Step": name, **_latency_row(values)})
    return with_formats(pd.DataFrame(rows), _LATENCY_FORMATS, title="Per-question latency by difficulty")


def _latency_derived(logs_by_stage: dict) -> pd.DataFrame:
    rows = []
    for c, label in _STAGES:
        entries = _latency_entries(logs_by_stage[c])
        if not entries:
            continue

        def total(key):
            return sum(e["latency"].get(key, 0.0) for e in entries)

        generate, end_to_end = total("llm_generate"), total("end_to_end")
        rows.append({
            "Variant": label,
            "Output tokens per second": sum(e["token_output"] for e in entries) / generate if generate else None,
            "Input tokens per generation second": sum(e["token_input"] for e in entries) / generate if generate else None,
            "Retrieval share of end-to-end": total("retrieval_total") / end_to_end if end_to_end else None,
            "Generation share of end-to-end": generate / end_to_end if end_to_end else None,
        })
    return with_formats(pd.DataFrame(rows), {"Output tokens per second": "num1", "Input tokens per generation second": "num1",
                                             "Retrieval share of end-to-end": "pct", "Generation share of end-to-end": "pct"},
                        title="Derived latency measures")


def _latency_run_table(run: dict) -> pd.DataFrame:
    rows = [{"Item": name, "Value": f"{seconds:,.3f} s"} for name, seconds in run.get("steps", {}).items()]
    if run.get("elapsed_s") is not None:
        rows.append({"Item": "Elapsed, first timed step to saving this file", "Value": f"{run['elapsed_s']:,.1f} s"})
    rows.append({"Item": "GPU", "Value": str(run.get("gpu"))})
    if run.get("peak_vram_gb") is not None:
        rows.append({"Item": "Peak VRAM allocated", "Value": f"{run['peak_vram_gb']:.2f} GB"})
    return with_formats(pd.DataFrame(rows), title="One-time steps, GPU, and peak VRAM")


def latency_tables(baseline_logs: list, graphrag_logs: list, run: Optional[dict] = None) -> list:
    """Tabel ringkasan latensi. raw_logs lama tanpa field latency -> tabel per pertanyaan dilewati."""
    tables = [_latency_run_table(run)] if run else []
    logs = {"baseline": baseline_logs, "graphrag": graphrag_logs}
    if _latency_entries(baseline_logs) or _latency_entries(graphrag_logs):
        tables += [_latency_by_step(logs), _latency_derived(logs), _latency_by_difficulty(logs)]
    return tables


def latency_per_query(baseline_logs: list, graphrag_logs: list) -> Optional[pd.DataFrame]:
    """Satu baris per pertanyaan dan varian, semua tahap dalam detik. Pertanyaan pertama tetap ditampilkan."""
    rows = []
    for (_, label), logs in zip(_STAGES, (baseline_logs, graphrag_logs)):
        for e in logs:
            if e.get("latency"):
                rows.append({"Query ID": e["query_id"], "DB": e["db_id"], "Variant": label, "Difficulty": DIFF_LABEL[e["difficulty"]],
                             **{name: e["latency"].get(key) for key, name in LATENCY_STEPS.items()}})
    if not rows:
        return None
    return with_formats(pd.DataFrame(rows), {name: "num3" for name in LATENCY_STEPS.values()}, title="Latency per question")


def cm_from_logs(baseline_logs: list, graphrag_logs: list) -> pd.DataFrame:
    """Component Match per klausa dari raw_logs (definisi aggregate_cm: rata-rata nilai non-None)."""
    def agg(logs, c):
        vals = [e["cm_per_clause"].get(c) for e in logs if e["cm_per_clause"].get(c) is not None]
        return sum(vals) / len(vals) if vals else None
    rows = []
    for c in CLAUSES:
        vb, vg = agg(baseline_logs, c), agg(graphrag_logs, c)
        rows.append({"Clause": CLAUSE_LABEL[c], "CM 1-stage": vb, "CM 2-stage": vg,
                     "Δ CM": None if vb is None or vg is None else (vg - vb) * 100})
    df = pd.DataFrame(rows)
    return with_formats(df, {"CM 1-stage": "pct", "CM 2-stage": "pct", "Δ CM": "pp"},
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
             "1-stage SQL": bb["predicted_sql"] if bb else None, "2-stage SQL": g["predicted_sql"]}
        for label, e in (("1-stage", bb), ("2-stage", g)):
            r[f"EX {label}"] = e["ex_result"] if e else None
            r[f"ESM {label}"] = e["esm_result"] if e else None
        for label, e in (("1-stage", bb), ("2-stage", g)):
            r[f"T_in {label}"] = e["token_input"] if e else None
            r[f"T_out {label}"] = e["token_output"] if e else None
            r[f"T {label}"] = _t(e, alpha) if e else None
        r["Gold columns"] = len(g["gold_schema"])
        for label, e in (("1-stage", bb), ("2-stage", g)):
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
    rows = [{"Diagnostic": "Samples", "Gold": diag["n_samples"], "1-stage": None, "2-stage": None},
            {"Diagnostic": "Queries with outer JOIN", "Gold": diag["outer_join_gold"],
             "1-stage": diag["outer_join_pred_baseline"], "2-stage": diag["outer_join_pred_graphrag"]},
            {"Diagnostic": "Outer-JOIN predictions with ESM=1 and EX=0", "Gold": None,
             "1-stage": diag["outer_join_norm_masked_baseline"], "2-stage": diag["outer_join_norm_masked_graphrag"]},
            {"Diagnostic": "Official EX=0, order-insensitive row match", "Gold": None,
             "1-stage": diag["ex0_orderinsensitive_match_baseline"], "2-stage": diag["ex0_orderinsensitive_match_graphrag"]},
            {"Diagnostic": "  of which gold has ORDER BY", "Gold": None,
             "1-stage": diag["ex0_orderinsensitive_match_baseline_gold_orderby"],
             "2-stage": diag["ex0_orderinsensitive_match_graphrag_gold_orderby"]},
            {"Diagnostic": "  of which gold has no ORDER BY", "Gold": None,
             "1-stage": diag["ex0_orderinsensitive_match_baseline"] - diag["ex0_orderinsensitive_match_baseline_gold_orderby"],
             "2-stage": diag["ex0_orderinsensitive_match_graphrag"] - diag["ex0_orderinsensitive_match_graphrag_gold_orderby"]},
            {"Diagnostic": "Gold SQL not executable (excluded)", "Gold": diag["gold_unexecutable"],
             "1-stage": None, "2-stage": None}]
    df = pd.DataFrame(rows)
    return with_formats(df, {"Gold": "int", "1-stage": "int", "2-stage": "int"}, title="Confound diagnostics (counts)")


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
                     "Mean T 1-stage": b["mean"], "Mean T 2-stage": g["mean"], "Δ mean T": g["mean"] - b["mean"],
                     "Relative change T": (g["mean"] - b["mean"]) / b["mean"] if b["mean"] else None,
                     "Median T 1-stage": b["median"], "Median T 2-stage": g["median"],
                     "EX 1-stage": None if exb is None else exb / 100, "EX 2-stage": None if exg is None else exg / 100,
                     "Δ EX": None if exb is None or exg is None else exg - exb})
    tokens = with_formats(pd.DataFrame(rows), {
        "Mean T 1-stage": "num1", "Mean T 2-stage": "num1", "Δ mean T": "dnum1", "Relative change T": "rel",
        "Median T 1-stage": "num1", "Median T 2-stage": "num1", "EX 1-stage": "pct", "EX 2-stage": "pct", "Δ EX": "pp"},
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
        # Tingkat kesulitan tanpa query (mis. pada sampel kecil) bernilai None.
        if not b or not g or b["esm"] is None or g["esm"] is None:
            continue
        rows.append({"Difficulty": "All" if d == "overall" else DIFF_LABEL[d],
                     "ESM 1-stage": b["esm"] / 100, "ESM 2-stage": g["esm"] / 100, "Δ ESM": g["esm"] - b["esm"],
                     "EX 1-stage": b["ex"] / 100, "EX 2-stage": g["ex"] / 100, "Δ EX": g["ex"] - b["ex"]})
    return [with_formats(pd.DataFrame(rows), {"ESM 1-stage": "pct", "ESM 2-stage": "pct", "Δ ESM": "pp",
                                               "EX 1-stage": "pct", "EX 2-stage": "pct", "Δ EX": "pp"},
                          title="D2 · ESM and EX by difficulty")]


def dim3_tables(dims: Optional[dict]) -> list:
    r = _dim(dims, 3)
    if r is None:
        return []
    cm = r["cm_per_clause"]
    rows = [{"Clause": CLAUSE_LABEL.get(c, c),
             "CM 1-stage": None if cm["baseline"].get(c) is None else cm["baseline"][c] / 100,
             "CM 2-stage": None if cm["graphrag"].get(c) is None else cm["graphrag"][c] / 100,
             "Δ CM": cm.get("delta", {}).get(c)}
            for c in CLAUSES if c in cm["baseline"]]
    return [with_formats(pd.DataFrame(rows), {"CM 1-stage": "pct", "CM 2-stage": "pct", "Δ CM": "pp"},
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
    delta = with_formats(pd.DataFrame([{"ΔQVT (2-stage − 1-stage)": r.get("delta_qvt")}]),
                          {"ΔQVT (2-stage − 1-stage)": "dnum4"}, title="D4 · ΔQVT")
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
            rows.append({"k": int(k), "Few-shot selection": sel, "EX 1-stage": None if is_missing(vb) else vb / 100,
                         "EX 2-stage": None if is_missing(vg) else vg / 100,
                         "Mean EX (both pipelines)": sum(vals) / len(vals) / 100 if len(vals) == 2 else None})
        out.append(with_formats(pd.DataFrame(rows), {"k": "int", "EX 1-stage": "pct", "EX 2-stage": "pct",
                                                      "Mean EX (both pipelines)": "pct"},
                                 title=f"Ablation stage 2 · EX per k (few-shot selection: {sel})"))
    spec = [("mode", "Pipeline", None, None), ("few_shot_selection", "Few-shot selection", None, None), ("k", "k", "int", None),
            ("avg_ex", "EX", "pct", 0.01), ("avg_esm", "ESM", "pct", 0.01), ("avg_prompt_tokens", "T_in", "num1", None),
            ("avg_output_tokens", "T_out", "num1", None), ("avg_token_consumption", "Token consumption", "num1", None),
            ("avg_recall", "Proxy recall", "pct", 0.01), ("avg_precision", "Proxy precision", "pct", 0.01),
            ("n_samples", "Queries", "int", None), ("tables_pct", "Tables pct", "num2", None),
            ("columns_pct", "Columns pct", "num2", None), ("llm_model", "LLM", None, None)]
    detail = df.sort_values(["few_shot_selection", "mode", "k"]).assign(mode=lambda d: d["mode"].map(pipeline_label))
    out.append(_select(detail, spec, "Ablation stage 2 · detail per pipeline and k"))
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
                     "EX 1-stage": None if is_missing(exb) else exb / 100, "EX 2-stage": None if is_missing(exg) else exg / 100,
                     "Δ EX": None if is_missing(exb) or is_missing(exg) else exg - exb,
                     "T 1-stage": tb, "T 2-stage": tg,
                     "Relative change T": (tg - tb) / tb if not (is_missing(tb) or is_missing(tg)) and tb else None})
    summary = with_formats(pd.DataFrame(rows), {"EX 1-stage": "pct", "EX 2-stage": "pct", "Δ EX": "pp",
                                                 "T 1-stage": "num1", "T 2-stage": "num1", "Relative change T": "rel"},
                            title="Model comparison · summary")
    spec = [("llm_model", "Model", None, None), ("mode", "Pipeline", None, None), ("n", "Queries", "int", None),
            ("ex", "EX", "pct", 0.01), ("esm", "ESM", "pct", 0.01), ("avg_t_in", "T_in", "num1", None),
            ("avg_t_out", "T_out", "num1", None), ("avg_token_consumption", "Token consumption", "num1", None),
            ("few_shot_k", "Few-shot k", "int", None), ("few_shot_selection", "Few-shot selection", None, None),
            ("tables_pct", "Tables pct", "num2", None), ("columns_pct", "Columns pct", "num2", None)]
    return [summary, _select(df.assign(mode=df["mode"].map(pipeline_label)), spec, "Model comparison · detail")]


def _with_median(field: str, header: str, kind: str) -> list:
    """Rata-rata, lalu median per query (median_summary() di token_reduction.py)."""
    return [(field, f"Avg {header}", kind, None), (f"{field}_med", f"Median {header}", kind, None)]


_TRIAL_THRESHOLDS = [("table_threshold", "Table threshold (T)", "num2", None),
                     ("column_threshold", "Column threshold (K)", "num2", None)]
_TRIAL_SETTINGS = [("stopword", "Stopwords", None, None), ("value_link", "Value linking", None, None),
                   ("traversal", "Traversal", None, None), ("no_pk", "No-PK tables", None, None),
                   ("nama_embed", "Embedded names", None, None)]
_TRIAL_MEASURES = [("max_ngram", "max_ngram", "int", None),
                   # SLA kolom (non-empty gold) tepat di samping EX, supaya keduanya mudah dibandingkan
                   ("ex", "EX (zero-shot)", "pct", None),
                   ("col_P_ne", "Column P (non-empty gold)", "pct", None), ("col_R_ne", "Column R (non-empty gold)", "pct", None),
                   ("tbl_P", "Table P", "pct", None), ("tbl_R", "Table R", "pct", None), ("tbl_F1", "Table F1", "pct", None),
                   ("col_P", "Column P", "pct", None), ("col_R", "Column R", "pct", None), ("col_F1", "Column F1", "pct", None),
                   *_with_median("tables_sent", "tables in prompt", "num1"), *_with_median("tables_db", "tables in DB", "num1"),
                   *_with_median("tables_pct", "tables retrieved, % of DB", "pct"),
                   *_with_median("cols_sent", "columns in prompt", "num1"),
                   ("cols_tables_full", "Avg columns of prompt tables, if sent whole", "num1", None),
                   *_with_median("cols_db", "columns in DB", "num1"), *_with_median("cols_pct", "columns retrieved, % of DB", "pct"),
                   ("col_reduction", "Column reduction vs full DDL", "pct", None),
                   *_with_median("schema_tokens", "schema tokens", "num1"), ("schema_tokens_db", "Avg schema tokens, full DDL", "num1", None),
                   ("schema_red_db", "Schema-token reduction vs full DDL", "pct", None),
                   *_with_median("prompt_tokens_k0", "prompt tokens (k=0)", "num1"),
                   ("prompt_tokens_k0_db", "Avg prompt tokens, full DDL (k=0)", "num1", None),
                   ("prompt_red_db", "Prompt-token reduction vs full DDL", "pct", None),
                   ("n", "Queries", "int", None)]
# Nilai pengaturan yang ditulis notebook trial -> teks Excel
_TRIAL_SETTING_VALUES = {"value_link": {"mati": "off", "teks": "text", "teks+angka": "text + numbers"},
                         "no_pk": {"lepas": "not connected", "sambung": "connected"},
                         "nama_embed": {"asli": "original"}}

# Tiap langkah preprocessing dibandingkan dengan langkah yang menjadi dasarnya: G5 dibangun dari G3, bukan G4 (angka tidak dipakai).
_STEP_PARENT = {"G2": "G1", "G3": "G2", "G4": "G3", "G5": "G3", "G6": "G5", "G7": "G6", "G8": "G7"}   # G8 = alternatif dari G7
_STEP_DELTAS = {"ex": "d_ex", "col_P_ne": "d_col_P_ne", "col_R_ne": "d_col_R_ne", "prompt_tokens_k0": "d_prompt_tokens"}
_TRIAL_STEP_SPEC = [("trial", "Trial", None, None), *_TRIAL_THRESHOLDS, *_TRIAL_SETTINGS,
    ("compared_to", "Compared with", None, None),
    ("ex", "EX (zero-shot)", "pct", None), ("d_ex", "Δ EX", "pp", 100),
    ("col_P_ne", "Column P (non-empty gold)", "pct", None), ("d_col_P_ne", "Δ Column P", "pp", 100),
    ("col_R_ne", "Column R (non-empty gold)", "pct", None), ("d_col_R_ne", "Δ Column R", "pp", 100),
    ("tables_pct", "Tables retrieved, % of DB", "pct", None), ("cols_pct", "Columns retrieved, % of DB", "pct", None),
    ("cols_sent", "Avg columns in prompt", "num1", None), ("col_reduction", "Column reduction vs full DDL", "pct", None),
    ("schema_tokens", "Avg schema tokens", "num1", None), ("schema_red_db", "Schema-token reduction vs full DDL", "pct", None),
    ("prompt_tokens_k0", "Avg prompt tokens (k=0)", "num1", None), ("d_prompt_tokens", "Δ Prompt tokens", "dnum1", None),
    ("prompt_red_db", "Prompt-token reduction vs full DDL", "pct", None),
    ("n", "Queries", "int", None)]


def _english_settings(df: pd.DataFrame) -> pd.DataFrame:
    return df.assign(**{c: df[c].replace(m) for c, m in _TRIAL_SETTING_VALUES.items() if c in df.columns})


def preprocessing_steps(recap: pd.DataFrame) -> Optional[pd.DataFrame]:
    """G1-G7 pada satu pasangan (T, K): apa yang berubah di tiap langkah preprocessing, untuk bab metodologi."""
    step = recap["trial"].str.extract(r"^(G\d) ")[0]
    first = recap[step == "G1"]
    if first.empty:
        return None
    same_pair = (recap["table_threshold"] == first["table_threshold"].iloc[0]) & (
        recap["column_threshold"] == first["column_threshold"].iloc[0])
    df = recap[step.notna() & same_pair].copy()
    df.index = step[df.index]
    df = df.sort_index()
    df["compared_to"] = df.index.map(_STEP_PARENT)
    for src, delta in _STEP_DELTAS.items():
        if src in df.columns:
            df[delta] = df[src] - df["compared_to"].map(df[src])
    return _select(df, _TRIAL_STEP_SPEC, f"SLA trial · preprocessing steps at T{first['table_threshold'].iloc[0]:.2f}"
                                         f"/K{first['column_threshold'].iloc[0]:.2f}")


def _trial_pipeline(trial: str) -> str:
    if trial.startswith("GraphRAG 1-Stage+ ") or trial.startswith("Baseline "):
        return pipeline_label("baseline") + "+"
    return pipeline_label("baseline" if trial.startswith("GraphRAG 1-Stage ") else "graphrag")


def sla_trial_tables(recap: Optional[pd.DataFrame]) -> dict:
    """Satu tabel per sheet Excel: riwayat preprocessing dan grid threshold; None kalau belum ada. Baris dikenali dari id trial di CSV."""
    if recap is None or recap.empty:
        return {"steps": None, "grid": None}
    recap = _english_settings(recap)
    # G7 (preprocessing akhir) dan G8 (alternatif) jalan di seluruh grid; di sheet preprocessing hanya pada pasangan
    # (T, K) yang sama dengan langkah G1-G6.
    ladder = recap["trial"].str.match(r"G[1-6] ")
    final = recap["trial"].str.match(r"G[78] ")
    same_pair = pd.MultiIndex.from_frame(recap[["table_threshold", "column_threshold"]]).isin(
        pd.MultiIndex.from_frame(recap.loc[ladder, ["table_threshold", "column_threshold"]]))
    steps = recap[ladder | (final & same_pair)]
    # Baris 1-Stage: "GraphRAG 1-Stage | T.." (nama tabel saja, poin 39) dan "GraphRAG 1-Stage+ | T.." (nama + kolom).
    # Rekap lama menulis "Baseline | T..", yang dulu selalu meng-embed nama + kolom, jadi dibaca sebagai 1-Stage+.
    one_stage = recap["trial"].str.match(r"GraphRAG 1-Stage\+? \||Baseline \|")
    # 1-stage (tanpa K) di atas 2-stage pada T yang sama; stable: G7 sebelum G8 di (T, K) yang sama
    grid = recap[final | one_stage].sort_values(
        ["table_threshold", "column_threshold"], na_position="first", kind="stable")
    grid = grid.assign(pipeline=grid["trial"].map(_trial_pipeline))
    # Tanpa kolom Trial: baris preprocessing sudah dibedakan oleh kolom pengaturannya
    step_spec = _TRIAL_SETTINGS + _TRIAL_THRESHOLDS + _TRIAL_MEASURES
    grid_spec = [("pipeline", "Pipeline", None, None), *_TRIAL_THRESHOLDS, *_TRIAL_SETTINGS, *_TRIAL_MEASURES]
    return {"steps": _select(steps, step_spec, "Threshold trial · preprocessing steps") if len(steps) else None,
            "grid": _select(grid, grid_spec, "Threshold trial · threshold grid") if len(grid) else None}


# ---------------------------------------------------------------------------
# Trial dua notebook (poin 42): trial_preprocessing.ipynb dan trial_thresholds.ipynb
# ---------------------------------------------------------------------------

def trial_tables(preprocessing: Optional[pd.DataFrame], thresholds: Optional[pd.DataFrame]) -> dict:
    """Tabel sheet Trial Preprocessing dan Trial Thresholds dari rekap kedua notebook trial."""
    out = dict.fromkeys(("steps", "grid"))
    if preprocessing is not None and not preprocessing.empty:
        out["steps"] = _select(_english_settings(preprocessing), _TRIAL_SETTINGS + _TRIAL_THRESHOLDS + _TRIAL_MEASURES,
                               "Preprocessing trial · all combinations")
    if thresholds is not None and not thresholds.empty:
        # 1-stage (tanpa K) di atas 2-stage pada T yang sama; stable: urutan preprocessing notebook tetap
        grid = _english_settings(thresholds).sort_values(["table_threshold", "column_threshold"], na_position="first",
                                                         kind="stable")
        grid = grid.assign(pipeline=grid["pipeline"].map(pipeline_label))
        out["grid"] = _select(grid, [("pipeline", "Pipeline", None, None), *_TRIAL_THRESHOLDS, *_TRIAL_SETTINGS,
                                     *_TRIAL_MEASURES], "Threshold trial · threshold grid")
    return out
