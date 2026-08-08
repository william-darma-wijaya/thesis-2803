"""
sla.py — Schema Linking Accuracy

Referensi: EVALUATION_ANALYSIS_GUIDE.md Bagian 1.1 (baca dulu sebelum implementasi).

PRIOR ART SUDAH ADA di codebase (bukan `recall_get_table()` dari percakapan lama —
itu tidak ditemukan di repo ini, kemungkinan referensinya sudah berubah nama):
- `src.retrieval.retrieval.evaluate_schema_linking()` dan `_parse_gold_elements()`
  sudah menghitung recall/precision retrieved-vs-gold untuk GraphRAG (column-level).
- `src.retrieval.baseline.evaluate_table_linking()` versi table-level-nya.
Keduanya dipakai LIVE di pipeline.py/ablation.py untuk print progress, TAPI:
  1. Keduanya mencampur table+column jadi SATU set (tidak dipisah table-level vs
     column-level seperti diminta guide 1.1 poin (c) langkah 4).
  2. Ground truth di-ekstrak dengan token-matching terhadap known schema names
     (lihat _parse_gold_elements di retrieval.py), BUKAN dengan SQL parser
     (guide poin (b) menyaratkan parse gold SQL ke table.column secara terstruktur).
  3. Precision/recall dari fungsi itu TIDAK divalidasi terhadap db_schema secara
     eksplisit (guide poin (e): validasi ground truth vs db_schema wajib).
extract_ground_truth_schema() di bawah ini SENGAJA ditulis dari nol pakai SQL
parser resmi SPIDER (bukan reuse _parse_gold_elements) supaya ketiga poin di atas
terpenuhi persis sesuai guide.

Formula:
    Precision = |ground_truth ∩ predicted| / |predicted|
    Recall    = |ground_truth ∩ predicted| / |ground_truth|
    F1        = 2 * (Precision * Recall) / (Precision + Recall)

Dihitung di DUA level terpisah: table-level dan column-level.

STATUS: metrik SLA lengkap diimplementasi -- extract_ground_truth_schema()
(gold_schema, pakai SQL parser resmi SPIDER) DAN compute_sla()/aggregate_sla()
(precision/recall/F1, table-level & column-level terpisah, macro-average per
query). Dipanggil per-query dari data/raw_logs/*.json (gold_schema +
predicted_schema sudah tersimpan di situ) saat Dimensi 5 (Bottleneck Retrieval
vs Generation, src/dimensions/dim5_bottleneck.py) diimplementasi -- dim5 itu
sendiri masih stub, tapi metrik yang dibutuhkannya sudah siap.
"""

import sys
from pathlib import Path
from typing import Dict, List, Set
from dataclasses import dataclass

# external/spider_eval/ ada di root project, bukan di src/ -- bootstrap sys.path
# supaya "from process_sql import ..." bisa resolve. Pola yang sama dipakai di
# semua entry point src/experiments/*.py (sys.path.insert(0, parents[2])).
_SPIDER_EVAL_DIR = Path(__file__).resolve().parents[2] / "external" / "spider_eval"
if str(_SPIDER_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_SPIDER_EVAL_DIR))

from process_sql import Schema, get_sql  # noqa: E402  (import after sys.path bootstrap, intentional)

from src.utils.schema_utils import is_valid_schema_element


@dataclass
class SLAResult:
    precision: float
    recall: float
    f1: float


def _schema_dict_for_spider_parser(db_id: str, db_schema: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    process_sql.Schema() butuh input berbentuk {table_name: [column_name, ...]},
    BUKAN {db_id: ["table.column", ...]} seperti format db_schema kita. Fungsi ini
    cuma mengubah bentuk (tidak butuh koneksi SQLite sama sekali -- makanya
    extract_ground_truth_schema() di bawah tidak perlu parameter db_path).
    """
    schema: Dict[str, List[str]] = {}
    for table_column in db_schema.get(db_id, []):
        table, _, column = table_column.partition(".")
        schema.setdefault(table, []).append(column)
    return schema


def _collect_col_ids(sql: dict, out: Set[str]) -> None:
    """
    Jalan rekursif ke seluruh bagian sql dict hasil process_sql.get_sql(), kumpulkan
    setiap "col_id" (string identifier kolom ala Spider, format "__table.column__"
    atau "__all__" untuk "*") yang muncul di SELECT/WHERE/GROUP BY/HAVING/ORDER BY/
    FROM(join conditions)/nested subquery/INTERSECT/UNION/EXCEPT.

    Struktur sql dict (didokumentasikan di header process_sql.py):
        col_unit  = (agg_id, col_id, isDistinct)
        val_unit  = (unit_op, col_unit1, col_unit2)      # col_unit2 bisa None
        val       = number | string | sql(dict) | col_unit  (lihat parse_value())
        cond_unit = (not_op, op_id, val_unit, val1, val2)
        condition = [cond_unit, 'and'/'or', cond_unit, ...]
        table_unit = (table_type, table_id | sql(dict))   # table_type: 'table_unit'|'sql'
        sql = {'select', 'from', 'where', 'groupBy', 'having', 'orderBy',
               'limit', 'intersect', 'union', 'except'}

    Ini re-implementasi kecil yang sengaja ditulis di sini (bukan dari
    process_sql.py, yang tidak menyediakan walker seperti ini secara publik) --
    lihat catatan limitasi "SELECT *" di extract_ground_truth_schema().
    """
    if sql is None:
        return

    def visit_col_unit(col_unit):
        if col_unit is None:
            return
        _agg_id, col_id, _is_distinct = col_unit
        out.add(col_id)

    def visit_val_unit(val_unit):
        if val_unit is None:
            return
        _unit_op, col_unit1, col_unit2 = val_unit
        visit_col_unit(col_unit1)
        visit_col_unit(col_unit2)

    def visit_value(val):
        # val bisa berupa float/str literal (bukan schema reference -- skip),
        # sql dict bersarang (subquery -- recurse), atau col_unit tuple (untuk
        # perbandingan kolom-ke-kolom semacam "WHERE a.x = b.y" -- lihat
        # parse_value() di process_sql.py, fallback branch-nya parse_col_unit()).
        if isinstance(val, dict):
            _collect_col_ids(val, out)
        elif isinstance(val, tuple):
            visit_col_unit(val)

    def visit_condition(condition):
        for item in condition:
            if item in ("and", "or"):
                continue
            _not_op, _op_id, val_unit, val1, val2 = item
            visit_val_unit(val_unit)
            visit_value(val1)
            visit_value(val2)

    # SELECT: (isDistinct, [(agg_id, val_unit), ...])
    _is_distinct, val_units = sql["select"]
    for _agg_id, val_unit in val_units:
        visit_val_unit(val_unit)

    # FROM: {'table_units': [table_unit, ...], 'conds': condition}  (conds = ON clauses)
    for table_type, unit in sql["from"]["table_units"]:
        if table_type == "sql":
            _collect_col_ids(unit, out)
        # table_type == "table_unit": unit cuma table_id ("__table__"), bukan
        # kolom -- table yang di-FROM tanpa kolom terpilih tidak menghasilkan
        # entry "table.column" (lihat limitasi SELECT * di bawah).
    visit_condition(sql["from"]["conds"])

    visit_condition(sql["where"])
    visit_condition(sql["having"])

    for col_unit in sql["groupBy"]:
        visit_col_unit(col_unit)

    if sql["orderBy"]:
        _direction, val_units = sql["orderBy"]
        for val_unit in val_units:
            visit_val_unit(val_unit)

    for key in ("intersect", "union", "except"):
        _collect_col_ids(sql.get(key), out)


def _col_id_to_table_column(col_id: str) -> str | None:
    """
    Konversi Spider col_id ("__table.column__") jadi "table.column". Return None
    untuk hal yang BUKAN kolom asli:
      - "__all__" -- ini "*", tidak bisa diterjemahkan ke kolom spesifik tanpa
        expand manual (lihat limitasi di extract_ground_truth_schema()).
      - id yang tidak mengandung "." setelah di-strip -- itu table_id murni
        ("__table__", bukan "__table.column__"), muncul kalau parser secara
        internal mereferensikan tabel tanpa kolom (jarang lolos ke sini karena
        col_unit selalu berisi col_id kolom, tapi dijaga untuk safety).
    """
    if col_id == "__all__":
        return None
    if col_id.startswith("__") and col_id.endswith("__"):
        inner = col_id[2:-2]
        if "." in inner:
            return inner
    return None


def extract_ground_truth_schema(gold_sql: str, db_id: str, db_schema: Dict[str, List[str]]) -> Set[str]:
    """
    Parse gold_sql pakai SQL parser resmi SPIDER (process_sql.py), ekstrak semua
    "table.column" yang dipakai (SELECT, WHERE, JOIN, GROUP BY, ORDER BY, HAVING,
    subquery, INTERSECT/UNION/EXCEPT), lalu validasi tiap elemen terhadap
    db_schema[db_id] (guide 1.1 poin (c) langkah 1-2, poin (e)).

    ⚠️ LIMITASI YANG DIKETAHUI (bukan bug, keterbatasan struktural parser):
    "SELECT * FROM t" tidak menghasilkan entry apapun dari klausa SELECT itu
    sendiri -- Spider parser merepresentasikan "*" sebagai satu id ("__all__"),
    bukan expand ke semua kolom tabel t. Kalau query itu juga punya WHERE/JOIN/dst
    yang mereferensikan kolom spesifik, kolom itu tetap ketangkap seperti biasa;
    yang hilang HANYA kontribusi dari "*" itu sendiri. Expand otomatis "*" ke
    semua kolom FROM-table adalah keputusan desain (apakah representatif untuk
    ground truth SLA atau tidak), bukan sesuatu yang diasumsikan sendiri di sini.
    """
    # Bangun Schema() dari db_schema kita, bukan dari koneksi SQLite -- lihat
    # _schema_dict_for_spider_parser(). Ini juga berarti fungsi ini tidak butuh
    # akses ke file .sqlite sama sekali, cukup db_schema yang sudah di-load.
    schema = Schema(_schema_dict_for_spider_parser(db_id, db_schema))
    parsed = get_sql(schema, gold_sql)

    col_ids: Set[str] = set()
    _collect_col_ids(parsed, col_ids)

    candidates = {
        table_column
        for col_id in col_ids
        if (table_column := _col_id_to_table_column(col_id)) is not None
    }

    # Validasi wajib terhadap db_schema (guide 1.1 poin (e)) -- proses parsing
    # SQL kadang menangkap alias yang secara kebetulan valid secara sintaks tapi
    # bukan schema asli; buang elemen yang tidak lolos validasi.
    return {
        table_column
        for table_column in candidates
        if is_valid_schema_element(db_id, table_column, db_schema)
    }


def compute_sla(
    ground_truth: Set[str],
    predicted: Set[str],
    level: str = "column",
) -> SLAResult:
    """
    Hitung precision, recall, f1 di level tertentu ("table" atau "column").

    Referensi guide 1.1 poin (c) langkah 4-5.

    PENTING soal validasi (guide 1.1 poin (c) langkah 2-3): `ground_truth` dan
    `predicted` di sini HARUS SUDAH tervalidasi terhadap db_schema oleh caller
    SEBELUM dipanggil ke sini -- compute_sla() sendiri tidak melakukan validasi
    apapun, cuma set-operation murni. Kalau sumbernya `extract_ground_truth_schema()`
    (di atas), itu sudah tervalidasi. Kalau sumbernya predicted_schema di
    data/raw_logs/*.json, itu juga sudah valid by construction (dibangun langsung
    dari node schema graph di src/experiments/pipeline.py, bukan dari parsing SQL
    predicted yang bisa berisi halusinasi kolom).

    Edge case pembagian oleh nol (predicted/ground_truth kosong): fallback 0.0,
    persis seperti disebutkan di guide -- BUKAN convention "1.0 = perfect by
    convention" yang dipakai src.retrieval.retrieval.evaluate_schema_linking()
    untuk kasus serupa (fungsi itu beda tujuan/tidak diacu guide untuk SLA).
    """
    if level == "table":
        # Ambil bagian sebelum "." pertama dari tiap elemen "table.column".
        # split(".", 1) (bukan split(".")) supaya kolom yang namanya sendiri
        # mengandung "." (jarang, tapi mungkin) tidak ikut kepotong table-nya --
        # pola yang sama dipakai schema_utils.get_table_names().
        ground_truth = {elem.split(".", 1)[0] for elem in ground_truth}
        predicted = {elem.split(".", 1)[0] for elem in predicted}
    elif level != "column":
        raise ValueError(f"level harus 'table' atau 'column', bukan {level!r}")

    true_positive = len(ground_truth & predicted)
    precision = true_positive / len(predicted) if predicted else 0.0
    recall = true_positive / len(ground_truth) if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return SLAResult(precision=precision, recall=recall, f1=f1)


def aggregate_sla(results: List[SLAResult]) -> SLAResult:
    """
    Agregasi across semua query: mean precision, mean recall, mean f1
    (macro-average per query, BUKAN micro-average — lihat guide 1.1 poin (c) langkah 6).

    Macro-average di sini artinya: precision/recall/f1 dihitung PER QUERY dulu
    (oleh compute_sla di atas), baru di-rata-rata across query. Ini BEDA dari
    micro-average (jumlahkan semua TP/predicted/ground_truth dulu across semua
    query, baru hitung satu precision/recall/f1 dari total itu) -- guide 1.1
    poin (c) langkah 6 eksplisit minta macro-average, jadi caller WAJIB memanggil
    compute_sla() sekali per query dan kumpulkan hasilnya ke list ini, bukan
    menggabung semua ground_truth/predicted jadi satu set besar lalu compute_sla
    sekali untuk semuanya (itu akan jadi micro-average, salah).
    """
    if not results:
        # Guide tidak menyebutkan kasus ini secara eksplisit -- dataset kosong
        # berarti tidak ada query yang dievaluasi sama sekali, bukan "semua
        # query gagal" (itu kasus lain). 0.0 dipilih supaya tidak diam-diam
        # menyembunyikan bug pemanggil (mis. lupa isi list sebelum agregasi)
        # di balik angka yang terlihat valid.
        return SLAResult(precision=0.0, recall=0.0, f1=0.0)

    n = len(results)
    return SLAResult(
        precision=sum(r.precision for r in results) / n,
        recall=sum(r.recall for r in results) / n,
        f1=sum(r.f1 for r in results) / n,
    )
