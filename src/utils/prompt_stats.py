"""
prompt_stats.py — per-query size of the schema context, measured the same way for
GraphRAG and Baseline (kode custom skripsi, bukan SPIDER). Lihat
context/IMPLEMENTATION_DECISIONS.md poin 33 untuk definisi apple-to-apple.

Columns are counted from `predicted_schema` ("table.column", lowercase, no `*`),
which for both pipelines is exactly the set of columns written into the DDL:
GraphRAG renders only its retrieved column nodes, Baseline renders every column
of its selected tables. Tokens use the generation tokenizer on the raw string,
like T_in (generate_sql_with_token_count() also tokenizes the raw prompt).
"""

from collections import Counter

import networkx as nx


def columns_per_table(col_graph: nx.Graph) -> Counter:
    """(db_id, table lowercase) -> number of columns, from the column-level graph."""
    seen = {
        (d["database"], d["table"].lower(), d["column"].lower())
        for _, d in col_graph.nodes(data=True)
        if d.get("type") == "column" and d.get("column") != "*"
    }
    return Counter((db, t) for db, t, _ in seen)


def prompt_stats(predicted_schema: list[str], db_id: str, table_cols: Counter,
                 schema_context: str, tokenizer) -> dict:
    """
    Fields added to a raw_logs / ablation entry:
        schema_tokens        : tokens of the DDL block alone (T_in minus this = the
                               part both pipelines share: task, few-shot, question,
                               instructions)
        n_columns_db         : columns of the whole database (prompt == whole DB
                               when n_columns == n_columns_db)
        n_columns_tables_full: columns of the prompt's tables if they were sent in
                               full. Equals n_columns for Baseline; for GraphRAG it
                               splits the column reduction into table selection vs
                               column pruning.
    """
    tables = {s.split(".", 1)[0] for s in predicted_schema}
    return {
        "schema_tokens": len(tokenizer.encode(schema_context)) if schema_context else 0,
        "n_columns_db": sum(n for (db, _), n in table_cols.items() if db == db_id),
        "n_columns_tables_full": sum(table_cols.get((db_id, t), 0) for t in tables),
    }
