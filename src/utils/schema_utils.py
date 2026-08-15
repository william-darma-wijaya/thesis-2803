"""
schema_utils.py

Helper untuk load & validasi schema database (dipakai di src/metrics/sla.py terutama,
tapi juga dipakai metrics lain untuk validasi table.column).

Referensi: EVALUATION_ANALYSIS_GUIDE.md Bagian 1.1 (SLA), poin (b) dan (c) langkah 2-3.
"""

from pathlib import Path
from typing import Dict, List

from src.core.schema import load_spider_schema


def load_db_schema(schema_path: str) -> Dict[str, List[str]]:
    """
    Load schema semua database dari tables.json, return dict:
    {db_id: ["table.column", "table.column", ...]}.

    Reuse src.core.schema.load_spider_schema() daripada re-parse tables.json dari
    nol — fungsi itu sudah menghasilkan DataFrame satu baris per kolom dengan
    kolom Database/Table/Column. Yang perlu ditambahkan di sini cuma dua hal yang
    load_spider_schema() SENGAJA tidak lakukan (karena dipakai juga oleh graph
    builder yang butuh baris "*" tetap ada untuk keperluan lain):
      1. Buang baris wildcard "*" (Column == "*", biasanya row pertama tiap
         database, dengan Table="ALL") — itu bukan kolom asli, guide 1.1 poin (c)
         eksplisit minta di-exclude dari schema linking.
      2. Gabungkan Table+Column jadi satu string "table.column", di-lowercase
         supaya konsisten dengan aturan case-insensitive di guide 1.1 poin (e).
    """
    schema_df = load_spider_schema(Path(schema_path))

    # Exclude kolom wildcard "*" sebelum dibentuk jadi "table.column" (lihat
    # docstring di atas — load_spider_schema() tidak melakukan ini sendiri).
    real_columns = schema_df[schema_df["Column"] != "*"]

    db_schema: Dict[str, List[str]] = {}
    for db_id, group in real_columns.groupby("Database"):
        db_schema[db_id] = [
            f"{table.lower()}.{column.lower()}"
            for table, column in zip(group["Table"], group["Column"])
        ]
    return db_schema


def is_valid_schema_element(db_id: str, table_column: str, db_schema: Dict[str, List[str]]) -> bool:
    """
    Cek apakah "table.column" valid ada di db_schema[db_id]. Case-insensitive
    (selalu .lower() dulu sebelum bandingkan — lihat guide 1.1 poin (e)).
    """
    if db_id not in db_schema:
        return False
    return table_column.lower() in db_schema[db_id]


def get_table_names(db_id: str, db_schema: Dict[str, List[str]]) -> List[str]:
    """
    Ambil semua nama tabel unik dari db_schema[db_id] (dipakai untuk perbandingan
    table-level di SLA, guide 1.1 poin (c) langkah 4).
    """
    if db_id not in db_schema:
        return []
    # Tiap entry di db_schema[db_id] sudah dalam bentuk "table.column" (lowercase,
    # lihat load_db_schema di atas) -- ambil bagian sebelum "." pertama saja.
    # split(".", 1) supaya kolom yang namanya mengandung "." (jarang, tapi
    # mungkin di beberapa DB Spider) tidak ikut kepotong table name-nya.
    tables = {entry.split(".", 1)[0] for entry in db_schema[db_id]}
    return sorted(tables)
