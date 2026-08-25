"""
sql_normalize.py

Single source of truth untuk menutup celah parser resmi SPIDER
(external/spider_eval/process_sql.py) yang tidak mengenali join-type keyword
apapun selain 'join' polos.

KONTEKS: process_sql.py's JOIN_KEYWORDS = ('join', 'on', 'as') (baris 32) --
tidak ada 'inner'/'cross'/'left'/'right'/'full'/'outer'. parse_from() (baris
~388) cuma skip token kalau PERSIS 'join'; token lain di depan 'join' (mis.
'left') malah dicoba diparse sebagai nama tabel/alias oleh parse_table_unit(),
lalu crash (KeyError: 'inner'/'cross'/'left'/'right'/'full' -- sudah
diverifikasi langsung menjalankan get_sql() terhadap kedelapan varian join).

AMAN dinormalisasi untuk PARSING karena struktur hasil parse
'from': {'table_units': [...], 'conds': condition} (process_sql.py baris 15)
TIDAK PERNAH merekam join type sama sekali -- ESM/CM tidak kehilangan
informasi apapun yang sebenarnya mereka bandingkan.

TIDAK semua join type aman dinormalisasi untuk EKSEKUSI, karena itu alasan ada
dua tier (lihat execution_safe_only di bawah):
  - INNER JOIN / CROSS JOIN -> 100% setara hasil eksekusi dengan JOIN polos.
  - LEFT/RIGHT/FULL[ OUTER] JOIN -> TIDAK setara -- baris NULL-padded yang
    dipertahankan join-join ini beneran mengubah result set kalau
    dieksekusi sebagai JOIN biasa.

Referensi: context/IMPLEMENTATION_DECISIONS.md poin 11 (keputusan lengkap +
kenapa dua tier ini perlu), menyelesaikan open item A yang sebelumnya belum
diputuskan.

Dipakai oleh:
  - src/metrics/esm_ex_cm.py (jalur in-process) -- lewat
    normalize_join_keywords_for_parsing() langsung, execution_safe_only=False
    default, karena string mentah untuk eksekusi (eval_exec_match()) TETAP
    dioper terpisah, tidak pernah lewat fungsi ini.
  - src/experiments/pipeline.py, src/experiments/ablation.py,
    notebooks/eval_pipeline.ipynb Bagian 10 (jalur subprocess, CLI resmi
    evaluation.py) -- lewat normalize_sql_file_for_parsing(), WAJIB
    execution_safe_only=True untuk --etype exec (CLI resmi memakai string yang
    SAMA untuk parsing maupun eksekusi, tidak ada pemisahan seperti jalur
    in-process -- lihat evaluation.py's evaluate(), baris ~501-547), boleh
    False untuk --etype match (tidak pernah mengeksekusi apapun, dikonfirmasi
    baris ~546: eval_exec_match cuma dipanggil kalau etype in ["all","exec"]).
"""

import re
from pathlib import Path
from typing import Union

# Execution-neutral: INNER JOIN / CROSS JOIN menghasilkan result set IDENTIK
# dengan JOIN polos. Aman dinormalisasi bahkan di teks yang akan benar-benar
# dieksekusi.
_EXEC_SAFE_JOIN_RE = re.compile(r"\b(?:INNER|CROSS)\s+JOIN\b", re.IGNORECASE)

# TIDAK execution-neutral: LEFT/RIGHT/FULL[ OUTER] JOIN (dan bare OUTER JOIN)
# mempertahankan baris NULL-padded yang tidak match -- menormalisasi ini di
# teks yang dieksekusi akan mengubah hasil query sungguhan. Hanya aman kalau
# salinan hasil normalisasi CUMA dipakai untuk parsing, string asli tetap yang
# dieksekusi.
_EXEC_UNSAFE_JOIN_RE = re.compile(
    r"\b(?:(?:LEFT|RIGHT|FULL)(?:\s+OUTER)?|OUTER)\s+JOIN\b", re.IGNORECASE
)


def normalize_join_keywords_for_parsing(sql: str, execution_safe_only: bool = False) -> str:
    """
    Ganti join-type keyword yang tidak dikenali process_sql.py's JOIN_KEYWORDS
    jadi 'JOIN' polos, supaya get_sql() tidak crash.

    execution_safe_only=True  -> HANYA normalisasi INNER/CROSS JOIN (aman
        dipakai bahkan di teks yang akan benar-benar dieksekusi/
        cursor.execute()).
    execution_safe_only=False (default) -> normalisasi SEMUA join type
        termasuk LEFT/RIGHT/FULL[ OUTER]/OUTER JOIN. HANYA aman untuk teks
        yang CUMA dipakai untuk parsing (get_sql()), TIDAK PERNAH untuk teks
        yang akan dieksekusi.

    Word-boundary (\\b) + keharusan token join-type diikuti langsung oleh
    'JOIN' membuat regex ini tidak mengenai nama kolom/tabel yang kebetulan
    mirip (mis. `SELECT t.full, t.outer_id FROM ...` -- 'full'/'outer' di
    sana tidak diikuti kata 'JOIN', jadi tidak match).
    """
    sql = _EXEC_SAFE_JOIN_RE.sub("JOIN", sql)
    if not execution_safe_only:
        sql = _EXEC_UNSAFE_JOIN_RE.sub("JOIN", sql)
    return sql


def normalize_sql_file_for_parsing(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    *,
    has_db_id_suffix: bool,
    execution_safe_only: bool = False,
) -> Path:
    """
    Baca file satu-SQL-per-baris (predictions.txt: SQL polos; dev_gold.sql:
    "SQL<TAB>db_id", format resmi Spider -- lihat evaluation.py's evaluate(),
    `g_str, db = g` baris ~503), normalisasi tiap baris lewat
    normalize_join_keywords_for_parsing(), tulis ke output_path.

    Dipakai HANYA untuk file yang dioper sebagai --gold/--pred ke
    evaluation.py CLI (jalur subprocess). Baris kosong dilewati (tidak ditulis
    ulang) -- konsisten dengan cara evaluation.py's evaluate() sendiri
    memfilter baris kosong (`if len(l.strip()) > 0`).

    PENTING: untuk --etype exec, WAJIB execution_safe_only=True -- lihat
    docstring modul.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if has_db_id_suffix:
                sql, _, db_id = line.rpartition("\t")
                f_out.write(
                    f"{normalize_join_keywords_for_parsing(sql, execution_safe_only)}\t{db_id}\n"
                )
            else:
                f_out.write(normalize_join_keywords_for_parsing(line, execution_safe_only) + "\n")

    return output_path
