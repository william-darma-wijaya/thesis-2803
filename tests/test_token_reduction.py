"""
Tes lokal reduksi kolom/token (IMPLEMENTATION_DECISIONS.md poin 33), tanpa Spider dan tanpa model:
skema sintetis 1 DB (3 tabel), tokenizer palsu (1 token per kata).

    python tests/test_token_reduction.py
"""

import sys
from pathlib import Path

import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments import result_tables as rt
from src.metrics import token_reduction as tr
from src.retrieval.baseline import build_table_schema_context, table_nodes_to_schema
from src.retrieval.retrieval import build_schema_context, column_nodes_to_schema
from src.utils.prompt_stats import columns_per_table, prompt_stats

DB = "shop"
TABLES = {
    "customer": [("id", "number", True), ("name", "text", False), ("city", "text", False), ("age", "number", False)],
    "orders": [("id", "number", True), ("customer_id", "number", False), ("amount", "number", False)],
    "product": [("id", "number", True), ("title", "text", False)],
}


class WordTokenizer:
    def encode(self, text):
        return text.split()


def column_graph():
    g = nx.Graph()
    for t, cols in TABLES.items():
        for c, typ, pk in cols:
            g.add_node(f"{DB}.{t}.{c}", type="column", database=DB, table=t, column=c, column_type=typ, is_pk=pk)
        g.add_node(f"{DB}.{t}.*", type="column", database=DB, table=t, column="*", column_type="text", is_pk=False)
    g.add_edge(f"{DB}.orders.customer_id", f"{DB}.customer.id", relation="foreign_key")
    return g


def table_graph():
    g = nx.Graph()
    for t, cols in TABLES.items():
        g.add_node(t, type="table", database=DB, table=t,
                   columns=[{"Column": c, "Type": typ, "is_pk": "Yes" if pk else "No"} for c, typ, pk in cols])
    g.add_edge("orders", "customer", relation="foreign_key",
               fks=[{"from_table": "orders", "from_col": "customer_id", "to_col": "id"}])
    return g


def entry(qid, schema, stats, t_in_extra, t_out, ex, difficulty="easy"):
    # T_in = schema tokens + a fixed non-schema part, as in a real prompt
    return {"query_id": qid, "difficulty": difficulty, "predicted_schema": schema,
            "token_input": stats["schema_tokens"] + t_in_extra, "token_output": t_out,
            "ex_result": ex, "esm_result": ex, **stats}


def main():
    tok = WordTokenizer()
    cg, tg = column_graph(), table_graph()
    table_cols = columns_per_table(cg)
    assert table_cols[(DB, "customer")] == 4 and table_cols[(DB, "orders")] == 3, table_cols  # `*` not counted

    # Query 1: Baseline sends customer + orders in full (7 cols); GraphRAG sends 3 columns of the same tables.
    b_nodes = ["customer", "orders"]
    b_schema = table_nodes_to_schema(tg, b_nodes)
    b_stats = prompt_stats(b_schema, DB, table_cols, build_table_schema_context(tg, b_nodes), tok)
    g_nodes = [f"{DB}.customer.name", f"{DB}.customer.id", f"{DB}.orders.customer_id"]
    g_schema = column_nodes_to_schema(cg, g_nodes)
    g_stats = prompt_stats(g_schema, DB, table_cols, build_schema_context(cg, g_nodes), tok)
    assert len(b_schema) == 7 and b_stats["n_columns_tables_full"] == 7 and b_stats["n_columns_db"] == 9
    assert len(g_schema) == 3 and g_stats["n_columns_tables_full"] == 7
    assert 0 < g_stats["schema_tokens"] < b_stats["schema_tokens"]

    # Query 2: GraphRAG pulls in a table Baseline did not pick (product): table effect negative.
    b2_nodes = ["customer"]
    b2_schema = table_nodes_to_schema(tg, b2_nodes)
    b2_stats = prompt_stats(b2_schema, DB, table_cols, build_table_schema_context(tg, b2_nodes), tok)
    g2_nodes = [f"{DB}.customer.city", f"{DB}.product.title"]
    g2_schema = column_nodes_to_schema(cg, g2_nodes)
    g2_stats = prompt_stats(g2_schema, DB, table_cols, build_schema_context(cg, g2_nodes), tok)

    FIXED = 100  # same non-schema prompt part for both pipelines (same template, question, few-shot)
    baseline = [entry("dev_0001", b_schema, b_stats, FIXED, 20, 1),
                entry("dev_0002", b2_schema, b2_stats, FIXED, 20, 0, "hard")]
    graphrag = [entry("dev_0002", g2_schema, g2_stats, FIXED, 25, 1, "hard"),   # order differs on purpose
                entry("dev_0001", g_schema, g_stats, FIXED, 20, 1)]

    pairs = tr.pair_logs(baseline, graphrag)
    assert [b["query_id"] for b, _ in pairs] == [g["query_id"] for _, g in pairs] == ["dev_0001", "dev_0002"]

    s = tr.reduction_summary(pairs, alpha=3.0)
    assert abs(s["columns"]["reduction"] - (1 - (3 + 2) / (7 + 4))) < 1e-9
    assert s["non_schema_tokens"]["reduction"] == 0           # shared part cancels exactly
    sb, sg = s["schema_tokens"]["baseline_mean"] * 2, s["schema_tokens"]["graphrag_mean"] * 2
    tin_b = sb + 2 * FIXED
    # prompt reduction = schema reduction x Baseline schema share (fixed part identical)
    assert abs(s["t_in"]["reduction"] - s["schema_tokens"]["reduction"] * sb / tin_b) < 1e-9
    assert s["t_in"]["reduction"] < s["schema_tokens"]["reduction"]

    dec = tr.column_decomposition(pairs)
    assert abs((1 - dec["total"]) - (1 - dec["table_selection"]) * (1 - dec["column_pruning"])) < 1e-9
    assert dec["table_selection"] < 0                       # (7 + 6) > (7 + 4): GraphRAG touched more table width
    assert tr.full_schema_rate(baseline) == 0 and tr.median_reduction([1], [0]) is None

    tables = rt.token_reduction(baseline, graphrag, 3.0) + [rt.token_reduction_by_difficulty(baseline, graphrag)]
    for t in tables:
        rt.formatted(t)                                     # renders with the notebook/Excel formats
    by_d = tables[-1].set_index("Difficulty")
    assert set(by_d.index) == {"Easy", "Hard", "All"} and by_d.loc["All", "Δ EX"] == 50.0

    # Logs written before poin 33 (no schema_tokens): no crash, schema rows simply absent.
    old = [{k: v for k, v in e.items() if k not in ("schema_tokens", "n_columns_db", "n_columns_tables_full")}
           for e in baseline + graphrag]
    s_old = tr.reduction_summary(tr.pair_logs(old[:2], old[2:]), 3.0)
    assert "schema_tokens" not in s_old and "columns" in s_old and tr.column_decomposition(tr.pair_logs(old[:2], old[2:])) is None
    rt.token_reduction(old[:2], old[2:], 3.0)
    print("OK: token reduction tests passed")


if __name__ == "__main__":
    main()
