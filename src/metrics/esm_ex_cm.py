"""
esm_ex_cm.py — Exact Set Match, Execution Accuracy, Component Match

Referensi: EVALUATION_ANALYSIS_GUIDE.md Bagian 1.2 (ESM), 1.3 (EX), 1.4 (CM).

PENTING: Ketiga metric ini pakai parser & evaluator resmi SPIDER
(external/spider_eval/evaluation.py + process_sql.py), diimport in-process
(bukan subprocess seperti yang dipakai pipeline.py/ablation.py/baseline.py untuk
skor agregat) supaya bisa dapat breakdown per-query per-klausa.

CATATAN MAPPING CLAUSES -> official partial_scores (SUDAH diverifikasi baca
langsung Evaluator.eval_partial_match() di evaluation.py, bukan tebakan):
    guide CLAUSES     -> official partial_scores key   -> catatan
    "select"          -> "select"                      -> termasuk AGG (bukan "select(no AGG)")
    "where"           -> "where"                        -> termasuk OP (bukan "where(no OP)")
    "group_by"        -> "group(no Having)"             -> nama official membingungkan: ini GROUP BY, BUKAN "no group"
    "having"          -> "group"                        -> nama official membingungkan: ini HAVING, dipanggil dari eval_having()
    "order_by"        -> "order"                        -> 1:1
    "keywords"         -> "keywords"                     -> 1:1
    "from"            -> (tidak ada di partial_scores)  -> dihitung manual di _from_clause_match(), replika
                                                            persis logic yang dipakai Evaluator.eval_exact_match()
                                                            sendiri untuk cek FROM/table set (evaluation.py:386-389)
    "union"/"intersect"/"except" -> (tidak ada, hanya ada gabungan "IUEN") -> dihitung manual per operator
                                                            di _set_op_clause_match(), reuse eval_exact_match()
                                                            secara rekursif pada sub-query masing-masing operator
    "union_all"       -> TIDAK DIDUKUNG                  -> Spider SQL_OPS di process_sql.py cuma punya
                                                            ('intersect','union','except') -- parser TIDAK
                                                            membedakan UNION vs UNION ALL secara struktural,
                                                            jadi tidak ada cara menghitung ini dari parser resmi.
                                                            Diisi None (bukan 0/1) supaya keterbatasan ini
                                                            eksplisit di data, bukan angka palsu.

acc/rec/f1 dari get_scores() di evaluation.py SUDAH biner (0 atau 1, bukan
pecahan -- lihat get_scores(): return (0,0,0) kalau pred_total != label_total,
return (1,1,1) kalau count == pred_total == label_total, else (0,0,0)). Itu
persis definisi "match" guide 1.4: exact set match per klausa, bukan skor
parsial. Termasuk kasus klausa kosong di kedua sisi (pred_total=label_total=0
-> count==pred_total -> (1,1,1)), yang otomatis memenuhi aturan guide 1.4 poin
(c): "klausa tidak muncul di keduanya -> match otomatis (1)" tanpa perlu
di-special-case manual.
"""

import copy
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# external/spider_eval/ ada di root project, bukan di src/ -- bootstrap sys.path
# supaya "from evaluation import ..." / "from process_sql import ..." resolve.
_SPIDER_EVAL_DIR = Path(__file__).resolve().parents[2] / "external" / "spider_eval"
if str(_SPIDER_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_SPIDER_EVAL_DIR))

from evaluation import (  # noqa: E402  (import after sys.path bootstrap, intentional)
    Evaluator,
    eval_exec_match,
    build_foreign_key_map_from_json,
    build_valid_col_units,
    rebuild_sql_val,
    rebuild_sql_col,
)
from process_sql import Schema, get_schema, get_sql  # noqa: E402

from src.utils.sql_normalize import normalize_join_keywords_for_parsing

# Lowercase snake_case dipertahankan sengaja (guide menulis nama klausa uppercase,
# mis. "SELECT"/"GROUP BY") -- konsisten dengan field lain di raw_logs yang semua
# lowercase snake_case. Keputusan didokumentasikan di
# context/IMPLEMENTATION_DECISIONS.md poin 7.
CLAUSES = [
    "select", "from", "where", "group_by", "order_by", "having",
    "keywords", "union", "union_all", "intersect", "except",
]

# Map dari nama klausa guide ke key partial_scores resmi (lihat catatan mapping
# di docstring modul). Klausa yang tidak ada di dict ini (from, union,
# intersect, except, union_all) dihitung terpisah, bukan dari partial_scores.
_PARTIAL_SCORE_KEY = {
    "select": "select",
    "where": "where",
    "group_by": "group(no Having)",
    "having": "group",
    "order_by": "order",
    "keywords": "keywords",
}

# sql dict kosong dipakai kalau predicted_sql gagal di-parse (bukan SQL valid
# menurut grammar Spider) -- persis salinan fallback yang dipakai official
# evaluate() (evaluation.py baris ~516-533) supaya predicted SQL yang tidak
# valid dievaluasi dengan cara yang sama seperti evaluasi resmi (ESM/CM = 0
# terhadap gold manapun, bukan exception).
def _empty_sql() -> dict:
    return {
        "except": None,
        "from": {"conds": [], "table_units": []},
        "groupBy": [],
        "having": [],
        "intersect": None,
        "limit": None,
        "orderBy": [],
        "select": [False, []],
        "union": None,
        "where": [],
    }


@dataclass
class QueryEvalResult:
    esm: int  # 0 atau 1
    ex: int  # 0 atau 1
    cm_per_clause: Dict[str, Optional[int]]  # {"select": 1, "where": 0, "union_all": None, ...}
    difficulty: str  # "easy" | "medium" | "hard" | "extra" (dari Evaluator.eval_hardness pada GOLD sql)
    # ^ "extra", BUKAN "extra_hard" seperti tertulis di guide Bagian 2 -- ikut
    # konvensi resmi SPIDER apa adanya, keputusan didokumentasikan di
    # context/IMPLEMENTATION_DECISIONS.md poin 6.


def build_kmaps(tables_json_path: str) -> dict:
    """
    Bangun foreign-key map untuk SEMUA database di tables.json sekaligus.

    PENTING SOAL PERFORMA: panggil fungsi ini SEKALI untuk seluruh evaluation run
    (bukan sekali per query, bukan sekali per db_id) -- build_foreign_key_map_from_json
    selalu parse SELURUH tables.json terlepas dari db mana yang dibutuhkan, jadi
    memanggilnya ribuan kali (sekali per query di dev set) akan re-parse file yang
    sama ribuan kali untuk hasil yang identik. Hasilnya (dict {db_id: kmap}) aman
    dipakai ulang untuk semua query di semua database.
    """
    return build_foreign_key_map_from_json(tables_json_path)


def build_schema_for_db(db_path: str) -> Schema:
    """
    Bangun Schema() Spider untuk SATU database SQLite (baca struktur tabel/kolom
    lewat PRAGMA table_info). Sama seperti build_kmaps(), sebaiknya di-cache per
    db_id oleh caller (mis. sekali per db_id di dev set) daripada dipanggil ulang
    setiap query -- tiap panggilan buka koneksi SQLite baru.
    """
    return Schema(get_schema(db_path))


def _from_clause_match(pred_sql: dict, gold_sql: dict) -> int:
    """
    Replika PERSIS logic FROM/table-set check yang dipakai Evaluator.eval_exact_match()
    sendiri (evaluation.py baris ~386-389) -- supaya nilai cm_per_clause["from"]
    konsisten dengan apa yang sebenarnya menentukan hasil `esm` (bukan definisi
    "from match" yang berbeda/independen).
    """
    gold_tables = gold_sql["from"]["table_units"]
    pred_tables = pred_sql["from"]["table_units"]
    if len(gold_tables) == 0:
        return 1
    return int(sorted(pred_tables) == sorted(gold_tables))


def _set_op_clause_match(pred_sub: Optional[dict], gold_sub: Optional[dict], evaluator: Evaluator) -> int:
    """
    Guide minta union/intersect/except dinilai sebagai klausa terpisah, tapi
    official evaluator cuma punya satu skor gabungan "IUEN" (Intersect/Union/
    Except/Nesting) di partial_scores -- tidak bisa dipecah balik ke 3 klausa
    independen dari situ. Jadi dihitung manual: kalau gold DAN predicted
    sama-sama tidak pakai operator ini -> match otomatis (1, konsisten dengan
    aturan guide 1.4 untuk klausa yang tidak muncul di keduanya). Kalau cuma
    salah satu yang pakai -> tidak match (0). Kalau keduanya pakai -> recurse
    eval_exact_match() pada sub-query masing-masing (intersect/union/except
    sendiri berupa sql dict penuh, jadi valid untuk exact-match check yang sama).

    PENTING (lihat context/IMPLEMENTATION_DECISIONS.md poin 17): pred_sub/
    gold_sub HARUS berupa salinan independen (deep copy) yang diambil SEBELUM
    eval_exact_match() utama dipanggil di _evaluate_single_query_inner(), BUKAN
    hasil `pred_sql.get(key)`/`gold_sql.get(key)` yang diambil SESUDAHNYA.
    eval_exact_match() memutasi in-place semua list di dalam dict yang dioper
    kepadanya -- termasuk merambat ke sub-dict union/intersect/except bersarang
    di dalam p_sql/g_sql -- jadi mengambil sub-query SETELAH panggilan utama
    akan memberikan data yang sudah "termakan", menghasilkan false-negative
    (predicted_sql == gold_sql persis tapi tetap dilaporkan tidak match) untuk
    SEMUA query yang pakai UNION/INTERSECT/EXCEPT. Diverifikasi ini bukan
    skenario langka: 38/40 query INTERSECT dan 29/31 query EXCEPT di Spider dev
    set asli salah skor 0 sebelum fix ini.

    `evaluator` di sini adalah instance yang SAMA yang dipakai untuk cek ESM
    utama di evaluate_single_query() -- aman dipakai ulang di sini karena
    evaluator.partial_scores (side effect dari eval_exact_match) sudah
    diambil/disalin ke variabel lokal SEBELUM fungsi ini dipanggil.
    """
    if gold_sub is None and pred_sub is None:
        return 1
    if gold_sub is None or pred_sub is None:
        return 0
    return int(evaluator.eval_exact_match(pred_sub, gold_sub))


def _evaluate_single_query_inner(
    predicted_sql: str,
    gold_sql: str,
    db_path: str,
    schema: Schema,
    kmap: dict,
) -> QueryEvalResult:
    evaluator = Evaluator()  # instance baru per query -- eval_exact_match() menyimpan
                              # state (self.partial_scores) sebagai side effect, jadi
                              # tiap query butuh instance sendiri supaya tidak ada
                              # kebocoran state antar query.

    # normalize_join_keywords_for_parsing() HANYA mempengaruhi salinan yang
    # di-parse di sini -- gold_sql/predicted_sql ASLI (tidak dinormalisasi)
    # tetap yang dioper ke eval_exec_match() di bawah (baris ~251). Lihat
    # src/utils/sql_normalize.py dan context/IMPLEMENTATION_DECISIONS.md
    # poin 11 untuk kenapa pemisahan ini wajib (LEFT/RIGHT/FULL JOIN mengubah
    # hasil eksekusi, execution_safe_only=False di sini aman justru KARENA
    # hasil normalisasi ini tidak pernah dieksekusi).
    g_sql = get_sql(schema, normalize_join_keywords_for_parsing(gold_sql))
    difficulty = evaluator.eval_hardness(g_sql)

    try:
        p_sql = get_sql(schema, normalize_join_keywords_for_parsing(predicted_sql))
    except Exception:
        # predicted SQL dari LLM sangat mungkin tidak valid secara grammar Spider
        # (beda dengan "tidak valid secara SQLite" yang ditangani terpisah di EX).
        # Fallback ke sql kosong -- persis pola yang dipakai official evaluate().
        # (Celah JOIN-keyword -- LEFT/RIGHT/FULL/INNER/CROSS JOIN -- sudah
        # ditangani lewat normalize_join_keywords_for_parsing() di atas;
        # fallback ini sekarang murni untuk SQL yang benar-benar tidak valid
        # secara grammar Spider, bukan untuk kasus join-type lagi.)
        p_sql = _empty_sql()

    # Normalisasi nilai literal & alias kolom lewat foreign key (mis. kolom yang
    # sama secara semantik tapi direferensikan lewat FK-linked column name yang
    # berbeda) -- WAJIB supaya hasil numeriknya konsisten dengan angka yang
    # dilaporkan Spider official evaluation.py (skip langkah ini akan
    # menghasilkan ESM/EX yang sedikit berbeda dari yang official).
    g_valid_col_units = build_valid_col_units(g_sql["from"]["table_units"], schema)
    g_sql = rebuild_sql_val(g_sql)
    g_sql = rebuild_sql_col(g_valid_col_units, g_sql, kmap)
    p_valid_col_units = build_valid_col_units(p_sql["from"]["table_units"], schema)
    p_sql = rebuild_sql_val(p_sql)
    p_sql = rebuild_sql_col(p_valid_col_units, p_sql, kmap)

    # Ambil salinan independen (deep copy) dari sub-query union/intersect/except
    # SEKARANG, SEBELUM eval_exact_match() utama dipanggil di bawah -- fungsi itu
    # memutasi in-place p_sql/g_sql, termasuk merambat ke sub-dict bersarang ini.
    # Kalau _set_op_clause_match() nanti mengambil sub-query dari p_sql/g_sql
    # yang SAMA setelah termutasi, hasilnya false-negative walau predicted_sql
    # == gold_sql persis. Lihat context/IMPLEMENTATION_DECISIONS.md poin 17 dan
    # docstring _set_op_clause_match() untuk detail lengkap + bukti empiris.
    p_set_ops = {op: copy.deepcopy(p_sql.get(op)) for op in ("union", "intersect", "except")}
    g_set_ops = {op: copy.deepcopy(g_sql.get(op)) for op in ("union", "intersect", "except")}

    # URUTAN INI PENTING -- EX harus dihitung SEBELUM ESM, bukan sesudahnya.
    # eval_exact_match() (lewat eval_partial_match() -> eval_sel()/eval_where()/dst)
    # MEMBUANG elemen yang sudah match dari list di dalam g_sql/p_sql secara
    # in-place (mis. eval_sel() literally memanggil `label_sel.remove(unit)` pada
    # list yang SAMA dengan g_sql['select'][1], bukan copy). Kalau eval_exec_match()
    # dipanggil SESUDAH eval_exact_match(), val_units yang dibacanya untuk
    # membangun result-column mapping sudah terlanjur kosong/berkurang, dan EX
    # akan salah (false negative) walau predicted_sql == gold_sql persis. Official
    # evaluate() di evaluation.py juga memanggil eval_exec_match() SEBELUM
    # evaluator.eval_exact_match() untuk alasan yang sama (lihat evaluation.py
    # baris ~546-560) -- urutan ini BUKAN kebetulan, harus diikuti persis.
    #
    # ✅ KEPUTUSAN PENELITI (didokumentasikan, bukan asumsi Claude): guide 1.3(c)
    # langkah 3 menulis EX seharusnya order-INSENSITIVE (bandingkan sebagai set
    # of tuples) KECUALI gold SQL punya ORDER BY eksplisit. eval_exec_match() DI
    # BAWAH ini TIDAK melakukan itu -- res_map() di evaluation.py (baris ~630-635)
    # membangun list per kolom SELECT langsung dari urutan cursor.fetchall(), lalu
    # membandingkan dict berisi list itu (order-sensitive selalu, apapun isi
    # gold['orderBy']). Ini bukan pilihan implementasi di sini -- itu memang
    # perilaku asli eval_exec_match() resmi SPIDER (sudah diverifikasi baca
    # source langsung, tidak ada sorted()/set() di mana pun di fungsi itu).
    # Guide sendiri mensyaratkan DUA hal yang berkontradiksi: "reuse parser resmi
    # SPIDER" (1.3 poin b) DAN "order-insensitive kecuali ORDER BY" (1.3 poin c
    # langkah 3) -- resmi SPIDER tidak melakukan yang kedua. Peneliti memutuskan
    # (2026-08-08) untuk TETAP pakai eval_exec_match() resmi apa adanya, supaya
    # angka EX tetap directly comparable dengan "Spider EX" yang dilaporkan
    # paper/sistem lain -- menerima keterbatasan bahwa dua query yang secara
    # relasional setara tapi dikembalikan SQLite dengan urutan baris fisik
    # berbeda (mis. tidak ada ORDER BY, join order beda) bisa ke-score EX=0.
    # src/utils/sql_execution.py's compare_execution_results() (yang akan
    # meng-implementasi order-insensitivity literal sesuai guide) SENGAJA
    # dibiarkan sebagai stub, TIDAK dipakai di sini -- jangan wire itu tanpa
    # instruksi baru dari peneliti.
    ex = 1 if eval_exec_match(db_path, predicted_sql, gold_sql, p_sql, g_sql) else 0

    esm = evaluator.eval_exact_match(p_sql, g_sql)
    partial_scores = evaluator.partial_scores  # side effect dari eval_exact_match di atas

    cm_per_clause: Dict[str, Optional[int]] = {}
    for clause, official_key in _PARTIAL_SCORE_KEY.items():
        # acc/rec/f1 sudah biner (lihat docstring modul) -- pakai 'f1' karena
        # itu yang dipakai eval_exact_match() sendiri untuk keputusan akhir.
        cm_per_clause[clause] = int(partial_scores[official_key]["f1"])
    cm_per_clause["from"] = _from_clause_match(p_sql, g_sql)
    cm_per_clause["union"] = _set_op_clause_match(p_set_ops["union"], g_set_ops["union"], evaluator)
    cm_per_clause["intersect"] = _set_op_clause_match(p_set_ops["intersect"], g_set_ops["intersect"], evaluator)
    cm_per_clause["except"] = _set_op_clause_match(p_set_ops["except"], g_set_ops["except"], evaluator)
    cm_per_clause["union_all"] = None  # tidak didukung parser resmi -- lihat docstring modul

    return QueryEvalResult(esm=int(esm), ex=ex, cm_per_clause=cm_per_clause, difficulty=difficulty)


def evaluate_single_query(
    predicted_sql: str,
    gold_sql: str,
    db_id: str,
    db_path: str,
    schema: Schema,
    kmap: dict,
) -> QueryEvalResult:
    """
    Hitung ESM, EX, difficulty, dan CM per klausa untuk SATU query.

    `schema` dan `kmap` HARUS dibangun sebelumnya lewat build_schema_for_db()
    (sekali per db_id) dan build_kmaps() (sekali untuk seluruh run) -- lihat
    catatan performa di kedua fungsi itu. db_id dioper ke sini cuma untuk
    logging kalau terjadi kegagalan (kmap sendiri sudah harus di-index oleh
    caller sebelum dipanggil, mis. `kmap = kmaps[db_id]`).

    Referensi guide 1.2 poin (c), 1.3 poin (c), 1.4 poin (c).
    """
    try:
        return _evaluate_single_query_inner(predicted_sql, gold_sql, db_path, schema, kmap)
    except Exception:
        # Official evaluator punya beberapa edge case yang tidak dia tangani
        # sendiri dengan aman (mis. eval_exact_match men-sort table_units yang
        # bisa berisi nested sql dict kalau FROM punya subquery -- dict tidak
        # bisa dibandingkan "<" di Python 3, akan throw TypeError). Predicted
        # SQL dari LLM juga bisa memicu kasus yang tidak diantisipasi parser di
        # luar try/except parsing predicted_sql di atas. Daripada satu query
        # aneh menghentikan seluruh proses generate raw_logs, treat sebagai
        # kegagalan evaluasi total (semua metrik = 0) -- perluasan dari prinsip
        # "predicted SQL invalid -> EX=0" yang sudah WAJIB di guide 1.3 poin (e).
        # LOGGED (bukan silent) supaya kegagalan yang sering / bukan karena
        # predicted SQL jelek (mis. bug di schema/db_id) tetap kelihatan.
        logger.warning(
            "evaluate_single_query gagal untuk db_id=%s, predicted_sql=%r: %s",
            db_id, predicted_sql[:100], repr(sys.exc_info()[1]),
        )
        return QueryEvalResult(
            esm=0, ex=0,
            # "union_all" TETAP None di sini, bukan 0 -- lihat docstring
            # aggregate_cm(): union_all SELALU None (parser resmi tidak bisa
            # bedakan UNION vs UNION ALL, bukan "biasanya gagal"). Menulis 0 di
            # fallback total-failure ini akan mencemari agregat union_all
            # dengan query yang gagal karena alasan LAIN sama sekali (bukan
            # soal union_all), keputusan didokumentasikan di
            # context/IMPLEMENTATION_DECISIONS.md poin 13.
            cm_per_clause={clause: (None if clause == "union_all" else 0) for clause in CLAUSES},
            difficulty="unknown",
        )


def aggregate_esm(results: List[QueryEvalResult]) -> float:
    """ESM = (jumlah query dengan esm=1) / total query * 100%"""
    if not results:
        return 0.0
    return sum(r.esm for r in results) / len(results) * 100


def aggregate_ex(results: List[QueryEvalResult]) -> float:
    """EX = (jumlah query dengan ex=1) / total query * 100%"""
    if not results:
        return 0.0
    return sum(r.ex for r in results) / len(results) * 100


def aggregate_cm(results: List[QueryEvalResult]) -> Dict[str, Optional[float]]:
    """
    Return dict {clause: akurasi_persen} untuk tiap klausa di CLAUSES.
    JANGAN digabung jadi satu angka rata-rata (lihat guide 1.4 poin (e)).

    "union_all" akan selalu None di sini (bukan 0.0) karena tidak ada satupun
    query yang punya nilai cm_per_clause["union_all"] terisi angka -- lihat
    docstring modul soal keterbatasan parser resmi untuk operator ini.
    """
    out: Dict[str, Optional[float]] = {}
    for clause in CLAUSES:
        values = [
            r.cm_per_clause[clause]
            for r in results
            if r.cm_per_clause.get(clause) is not None
        ]
        out[clause] = (sum(values) / len(values) * 100) if values else None
    return out
