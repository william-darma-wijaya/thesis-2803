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
risk dari kasus row order beda tapi hasil relasional setara.

UPDATE (2026-09-06): execute_sql() dan compare_execution_results() di bawah
SEKARANG diimplementasi penuh, TAPI dipakai HANYA oleh diagnostic counter
read-only di src/experiments/pipeline.py's run_comparison() yang MENGHITUNG
(bukan mengubah) seberapa sering EX resmi dan perbandingan order-insensitive
tidak sepakat, per kondisi (GraphRAG vs Baseline), untuk mengukur besar
confound order-sensitivity. Nilai ex_result di raw_logs TETAP 100% dari
eval_exec_match() resmi -- keputusan 2026-08-08 di atas TIDAK dibalik. JANGAN
wire compare_execution_results() ke esm_ex_cm.py atau menjadikannya sumber
ex_result/ex di manapun tanpa instruksi baru peneliti yang eksplisit membalik
keputusan itu.
"""

import sqlite3
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Optional


def execute_sql(db_path: str, sql: str, timeout: float = 30.0) -> Optional[List[Tuple]]:
    """
    Eksekusi satu query SQL di database SQLite pada db_path.
    Return hasil sebagai list of tuples, atau None kalau query gagal
    (syntax error, kolom tidak ada, timeout, dsb).

    Dibungkus try-except menyeluruh (guide 1.3 poin (e)) -- exception tidak
    boleh merambat ke atas. Predicted SQL dari LLM sangat mungkin invalid.
    Koneksi dibuka read-only supaya query tidak bisa memodifikasi database.
    """
    conn = None
    try:
        # as_uri() menghasilkan file URI yang benar di Windows & POSIX
        # (file:///C:/... vs file:///kaggle/...); string f"file:{path}" mentah
        # tidak valid di Windows (backslash + drive letter).
        try:
            uri = Path(db_path).resolve().as_uri() + "?mode=ro"
            conn = sqlite3.connect(uri, uri=True, timeout=timeout)
        except Exception:
            # Fallback: koneksi biasa (mis. path tidak absolut / as_uri gagal).
            conn = sqlite3.connect(db_path, timeout=timeout)
        # Spider DB punya byte non-UTF-8 di beberapa kolom teks; ikuti toleransi
        # evaluator resmi daripada me-raise.
        conn.text_factory = lambda b: b.decode("utf-8", errors="ignore")
        cur = conn.cursor()
        cur.execute(sql)
        return cur.fetchall()
    except Exception:
        return None
    finally:
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass


def compare_execution_results(
    result_predicted: Optional[List[Tuple]],
    result_gold: Optional[List[Tuple]],
    order_sensitive: bool = False,
) -> int:
    """
    Bandingkan dua hasil eksekusi SQL, return 1 (match) atau 0 (tidak match).

    Referensi guide 1.3 poin (c) langkah 3:
    - Kalau salah satu None (query gagal dieksekusi) -> return 0
    - order_sensitive=False (default): bandingkan sebagai MULTISET of rows
      (Counter) -- urutan baris tidak masalah, tapi jumlah baris duplikat tetap
      dihitung
    - order_sensitive=True (mis. gold SQL punya ORDER BY eksplisit): bandingkan
      sebagai LIST berurutan

    Dipakai HANYA oleh diagnostic counter di run_comparison() -- bukan sumber
    kebenaran EX (lihat catatan header modul).
    """
    if result_predicted is None or result_gold is None:
        return 0
    if order_sensitive:
        return int(list(result_predicted) == list(result_gold))
    try:
        return int(
            Counter(map(tuple, result_predicted)) == Counter(map(tuple, result_gold))
        )
    except TypeError:
        return int(list(result_predicted) == list(result_gold))
