"""
dim2_structure.py — Dimensi 2: Akurasi Struktur SQL

Referensi: EVALUATION_ANALYSIS_GUIDE.md Bagian 3, "Dimensi 2".
Menjawab RM2. Metrik: ESM, EX (dibaca BERPASANGAN, bukan terpisah).

Tabel interpretasi WAJIB (jangan buat sendiri):
| EX naik, ESM turun | EX naik, ESM naik | ESM naik, EX turun | keduanya rendah -> lanjut Dimensi 5 |
"""

from typing import List, Dict


def run_dimension_2(baseline_logs: List[dict], graphrag_logs: List[dict]) -> dict:
    """
    TODO:
    1. Agregasi ESM & EX untuk baseline & graphrag (overall + per difficulty)
    2. Tampilkan berdampingan dalam satu tabel
    3. Klasifikasikan ke salah satu dari 4 kombinasi (lihat guide Bagian 3
       Dimensi 2 untuk tabel lengkapnya):
       - EX naik & ESM turun -> "penghematan token tidak merusak logika semantic"
       - ESM naik & EX turun -> "struktur ditiru tapi gagal eksekusi"
       - EX naik & ESM naik  -> "kondisi paling ideal"
       - EX rendah & ESM rendah -> "kegagalan menyeluruh, lanjut ke Dimensi 5"
    4. Return dict berisi tabel + label kombinasi + catatan "lanjut ke Dimensi 5"
       kalau kondisi terakhir terpenuhi
    """
    raise NotImplementedError
