"""
sql_execution.py

Helper untuk eksekusi SQL secara aman di SQLite (dipakai untuk metric EX).

Referensi: EVALUATION_ANALYSIS_GUIDE.md Bagian 1.3 (Execution Accuracy), poin (c) dan (e).

PENTING: SQLite WAJIB, bukan PostgreSQL/Supabase. Evaluasi resmi SPIDER pakai SQLite.

CATATAN: external/spider_eval/evaluation.py (root project, sudah di-download)
punya `eval_exec_match(db, p_str, g_str, pred, gold)` (baris 614) yang JUGA
melakukan eksekusi + compare, sudah try-except di eksekusi predicted-nya. Bedanya:
fungsi official itu membandingkan hasil per SELECT-column lewat `val_units`
(struktur SELECT hasil parse process_sql.py), bukan set-of-tuples generik seperti
di compare_execution_results() di bawah.

✅ KEPUTUSAN PENELITI (2026-08-08): src/metrics/esm_ex_cm.py SUDAH pakai
eval_exec_match() resmi SPIDER untuk EX, bukan file ini. Ini SENGAJA, bukan
belum sempat — eval_exec_match() SELALU order-sensitive (bandingkan list
per-kolom langsung dari urutan cursor.fetchall(), tidak ada sorted()/set()
sama sekali di source resminya), BEDA dari algoritma di guide 1.3(c) langkah 3
yang minta order-insensitive KECUALI gold SQL punya ORDER BY eksplisit — dan
`compare_execution_results()` di bawah ini didesain justru untuk
mengimplementasikan behavior guide itu (lihat parameter `order_sensitive`).
Peneliti memilih TETAP pakai eval_exec_match() resmi (angka EX comparable
dengan "Spider EX" yang dilaporkan paper/sistem lain), menerima false-negative
risk dari kasus row order beda tapi hasil relasional setara. Karena itu,
`execute_sql()` dan `compare_execution_results()` di bawah TIDAK dipakai di
manapun di codebase — JANGAN wire ini ke esm_ex_cm.py atau pipeline manapun
tanpa instruksi baru dari peneliti yang membalik keputusan di atas.
"""

import sqlite3
from typing import List, Tuple, Optional


def execute_sql(db_path: str, sql: str) -> Optional[List[Tuple]]:
    """
    Eksekusi satu query SQL di database SQLite pada db_path.
    Return hasil sebagai list of tuples, atau None kalau query gagal
    (syntax error, kolom tidak ada, dsb).

    WAJIB dibungkus try-except (guide 1.3 poin (e)) — jangan biarkan exception
    merambat ke atas dan crash pipeline. Predicted SQL dari LLM sangat mungkin invalid.

    TODO:
    - Buka koneksi sqlite3 ke db_path
    - Jalankan sql via cursor.execute()
    - fetchall() hasilnya
    - Kalau ada exception (sqlite3.Error atau exception lain), return None
    - Tutup koneksi di finally block
    """
    raise NotImplementedError


def compare_execution_results(
    result_predicted: Optional[List[Tuple]],
    result_gold: Optional[List[Tuple]],
    order_sensitive: bool = False,
) -> int:
    """
    Bandingkan dua hasil eksekusi SQL, return 1 (match) atau 0 (tidak match).

    Referensi guide 1.3 poin (c) langkah 3:
    - Kalau result_predicted None (query gagal dieksekusi) -> return 0
    - Kalau order_sensitive=False (default), bandingkan sebagai SET of tuples
      supaya urutan baris tidak masalah
    - Kalau order_sensitive=True (gold SQL punya ORDER BY eksplisit), bandingkan
      sebagai LIST berurutan

    TODO: implementasikan logic perbandingan sesuai catatan di atas.
    """
    raise NotImplementedError
