"""
Schema processing and graph construction for the Spider dataset.
"""

import json
import logging
from pathlib import Path

import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


def load_spider_schema(json_path: Path) -> pd.DataFrame:
    """
    Parse Spider's tables.json into a flat DataFrame with one row per column.

    Columns: Database, Table, Column, Type, PK, FK_Relation
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = []
    for db in data:
        db_id = db["db_id"]
        table_names = db["table_names_original"]
        column_names = db["column_names_original"]
        column_types = db["column_types"]
        primary_keys = set(db["primary_keys"])
        fk_map = {fk[0]: fk[1] for fk in db["foreign_keys"]}

        for col_idx, (table_idx, col_name) in enumerate(column_names):
            table_name = table_names[table_idx] if table_idx != -1 else "ALL"

            fk_relation = "-"
            if col_idx in fk_map:
                target_idx = fk_map[col_idx]
                t_table_idx, t_col_name = column_names[target_idx]
                fk_relation = f"{table_names[t_table_idx]}.{t_col_name}"

            records.append(
                {
                    "Database": db_id,
                    "Table": table_name,
                    "Column": col_name,
                    "Type": column_types[col_idx],
                    "PK": "Yes" if col_idx in primary_keys else "No",
                    "FK_Relation": fk_relation,
                }
            )

    return pd.DataFrame(records)


def build_schema_graph(df: pd.DataFrame) -> nx.Graph:
    """
    Build a graph where each node is a (database, table, column) triple.

    Edges encode two relationships:
      - belongs_to_pk : non-PK column → PK column of the same table
      - foreign_key   : FK column → referenced PK column
    """
    G = nx.Graph()
    pk_map: dict[str, dict[str, str]] = {}  # db -> table -> pk_node_id

    df_clean = df[df["Column"] != "*"].copy()

    # --- Pass 1: add nodes ---
    for _, row in df_clean.iterrows():
        db, table, col = (
            str(row["Database"]).strip(),
            str(row["Table"]).strip(),
            str(row["Column"]).strip(),
        )
        node_id = f"{db}.{table}.{col}"
        is_pk = row["PK"] == "Yes"

        G.add_node(
            node_id,
            type="column",
            database=db,
            table=table,
            column=col,
            column_type=str(row["Type"]).strip(),
            is_pk=is_pk,
        )

        if db not in pk_map:
            pk_map[db] = {}
        if is_pk:
            pk_map[db][table] = node_id

    # --- Pass 2: add edges ---
    for _, row in df_clean.iterrows():
        db, table, col = (
            str(row["Database"]).strip(),
            str(row["Table"]).strip(),
            str(row["Column"]).strip(),
        )
        node_id = f"{db}.{table}.{col}"
        fk_rel = str(row["FK_Relation"]).strip()

        # Intra-table edge: column → its table's PK
        if table in pk_map.get(db, {}) and node_id != pk_map[db][table]:
            G.add_edge(node_id, pk_map[db][table], relation="belongs_to_pk")

        # FK edge: column → referenced PK
        if fk_rel != "-":
            target_node_id = f"{db}.{fk_rel}"
            if target_node_id in G:
                G.add_edge(node_id, target_node_id, relation="foreign_key")

    logger.info("Graph built: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G
