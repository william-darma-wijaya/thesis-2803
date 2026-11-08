"""
token_reduction.py — Column reduction & token reduction GraphRAG vs Baseline

Kode custom skripsi (bukan SPIDER). Permintaan dosen pembimbing (2026-11-02):
bandingkan pengurangan kolom dengan pengurangan token input prompt, dan
dampaknya ke EX. Definisi apple-to-apple: context/IMPLEMENTATION_DECISIONS.md
poin 33.

Ringkas:
- Berpasangan per query (query_id sama, pertanyaan sama, k dan contoh few-shot
  sama), jadi Baseline di query yang SAMA adalah pembandingnya.
- Reduksi = 1 - sum(GraphRAG) / sum(Baseline) atas query yang sama (rasio total,
  setara rasio rata-rata karena n sama). Median reduksi per query dilaporkan
  sebagai pemeriksaan robust, karena rata-rata rasio per query bisa didominasi
  query dengan prompt Baseline kecil.
- Nilai positif = GraphRAG lebih hemat.
"""

import statistics
from typing import Callable, Dict, List, Optional, Tuple


def pair_logs(baseline_logs: List[dict], graphrag_logs: List[dict]) -> List[Tuple[dict, dict]]:
    """(baseline, graphrag) per query_id yang ada di kedua log, urut query_id."""
    b = {e["query_id"]: e for e in baseline_logs}
    return [(b[e["query_id"]], e) for e in sorted(graphrag_logs, key=lambda e: e["query_id"])
            if e["query_id"] in b]


def total_reduction(g_values: List[float], b_values: List[float]) -> Optional[float]:
    """1 - sum(G) / sum(B); None kalau sum(B) = 0."""
    sb = sum(b_values)
    return None if sb == 0 else 1 - sum(g_values) / sb


def median_reduction(g_values: List[float], b_values: List[float]) -> Optional[float]:
    """Median dari 1 - g/b per query (query dengan b = 0 dilewati)."""
    r = [1 - g / b for g, b in zip(g_values, b_values) if b > 0]
    return statistics.median(r) if r else None


def _n_columns(e: dict) -> int:
    return len(e["predicted_schema"])


def _n_tables(e: dict) -> int:
    return len({s.split(".", 1)[0] for s in e["predicted_schema"]})


def quantities(alpha: float) -> List[Tuple[str, str, Callable[[dict], Optional[float]]]]:
    """(key, label, getter) yang dibandingkan. Getter None = field tidak ada di log lama."""
    return [
        ("tables", "Tables in prompt", _n_tables),
        ("columns", "Columns in prompt", _n_columns),
        ("schema_tokens", "Schema tokens (DDL only)", lambda e: e.get("schema_tokens")),
        ("non_schema_tokens", "Non-schema prompt tokens (T_in − schema)",
         lambda e: None if e.get("schema_tokens") is None else e["token_input"] - e["schema_tokens"]),
        ("t_in", "Prompt input tokens (T_in)", lambda e: e["token_input"]),
        ("t_out", "Output tokens (T_out)", lambda e: e["token_output"]),
        ("t", f"Token consumption T = T_in + {alpha:g}·T_out", lambda e: e["token_input"] + alpha * e["token_output"]),
    ]


def reduction_summary(pairs: List[Tuple[dict, dict]], alpha: float) -> Dict[str, dict]:
    """key -> {label, baseline_mean, graphrag_mean, reduction, median_reduction, n}."""
    out = {}
    for key, label, get in quantities(alpha):
        vals = [(get(b), get(g)) for b, g in pairs]
        vals = [(vb, vg) for vb, vg in vals if vb is not None and vg is not None]
        if not vals:
            continue
        bs, gs = [v[0] for v in vals], [v[1] for v in vals]
        out[key] = {"label": label, "baseline_mean": statistics.mean(bs), "graphrag_mean": statistics.mean(gs),
                    "reduction": total_reduction(gs, bs), "median_reduction": median_reduction(gs, bs),
                    "n": len(vals)}
    return out


def column_decomposition(pairs: List[Tuple[dict, dict]]) -> Optional[dict]:
    """
    Column reduction = table-selection effect x column-pruning effect:
        (1 - total) = (1 - table_selection) * (1 - column_pruning)
        table_selection = 1 - sum(GraphRAG tables at full width) / sum(Baseline columns)
        column_pruning  = 1 - sum(GraphRAG columns) / sum(GraphRAG tables at full width)
    table_selection < 0 means GraphRAG's prompt touches MORE/larger tables than
    Baseline's (value linking and traversal can add tables).
    """
    vals = [(_n_columns(b), g.get("n_columns_tables_full"), _n_columns(g)) for b, g in pairs]
    vals = [v for v in vals if v[1] is not None]
    if not vals:
        return None
    b_cols, g_full, g_cols = (sum(v[i] for v in vals) for i in range(3))
    return {
        "table_selection": total_reduction([g_full], [b_cols]),
        "column_pruning": total_reduction([g_cols], [g_full]),
        "total": total_reduction([g_cols], [b_cols]),
        "n": len(vals),
    }


def full_schema_rate(logs: List[dict]) -> Optional[float]:
    """Fraksi query yang prompt-nya berisi SELURUH kolom DB (fallback / DB kecil)."""
    vals = [_n_columns(e) >= e["n_columns_db"] for e in logs if e.get("n_columns_db")]
    return sum(vals) / len(vals) if vals else None


MEDIAN_FIELDS = ("tables_sent", "tables_db", "tables_pct", "cols_sent", "cols_db", "cols_pct", "schema_tokens", "prompt_tokens_k0")
MEDIAN_COLUMNS = tuple(f"{f}_med" for f in MEDIAN_FIELDS)


def median_summary(rows: List[dict]) -> Dict[str, float]:
    """
    Median per query, pelengkap rata-rata karena ukuran DB dan fallback seluruh DB membuat
    distribusinya miring (poin 35). Rata-rata tetap dipakai untuk total, biaya, dan TEP.
    """
    values = {f: [r[f] for r in rows] for f in MEDIAN_FIELDS if not f.endswith("_pct")}
    values["tables_pct"] = [r["tables_sent"] / r["tables_db"] for r in rows]
    values["cols_pct"] = [r["cols_sent"] / r["cols_db"] for r in rows]
    return {f"{field}_med": float(statistics.median(vals)) for field, vals in values.items()}


def full_db_summary(rows: List[dict]) -> Dict[str, float]:
    """
    Satu konfigurasi dibanding DDL PENUH database-nya, untuk rekap trial threshold (poin 35).
    Pembanding sama untuk GraphRAG dan Baseline, jadi angkanya bisa dibandingkan langsung.

    Field per query: tables_sent, tables_db, cols_sent, cols_db, schema_tokens,
    schema_tokens_db, prompt_tokens_k0, prompt_tokens_k0_db.

    - tables_pct / cols_pct = rata-rata per query dari sent / db: setiap query berbobot
      sama, berapa pun ukuran database-nya ("satu query melihat x% dari database-nya").
    - schema_red_db / prompt_red_db = 1 - sum(sent) / sum(db): rasio total, sama seperti
      reduksi terhadap Baseline. Karena itu 1 - cols_pct tidak harus sama dengan reduksi kolom.
    """
    def col(key):
        return [r[key] for r in rows]

    return {
        "tables_sent": statistics.mean(col("tables_sent")),
        "tables_db": statistics.mean(col("tables_db")),
        "tables_pct": statistics.mean(r["tables_sent"] / r["tables_db"] for r in rows),
        "cols_pct": statistics.mean(r["cols_sent"] / r["cols_db"] for r in rows),
        "schema_tokens_db": statistics.mean(col("schema_tokens_db")),
        "schema_red_db": total_reduction(col("schema_tokens"), col("schema_tokens_db")),
        "prompt_tokens_k0_db": statistics.mean(col("prompt_tokens_k0_db")),
        "prompt_red_db": total_reduction(col("prompt_tokens_k0"), col("prompt_tokens_k0_db")),
    }
