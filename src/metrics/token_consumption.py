"""
token_consumption.py — Token Consumption (T)

Referensi: EVALUATION_ANALYSIS_GUIDE.md Bagian 1.5, sub-bagian "Token Consumption".

DESAIN (diputuskan 2026-08-15, lihat context/IMPLEMENTATION_DECISIONS.md poin 9):
modul ini TIDAK menokenisasi teks sendiri. T_in/T_out sudah dihitung sebagai
integer di titik generasi — `generate_sql_with_token_count()` di
src/generation/generation.py memakai tokenizer Qwen2.5-Coder-7B-Instruct yang
SUDAH dimuat (bukan instance baru), dan hasilnya (`token_input`/`token_output`)
sudah tersimpan sebagai int di data/raw_logs/*.json dan outputs/tables/
ablation_results.csv. Memuat tokenizer kedua di sini via
AutoTokenizer.from_pretrained akan redundan dan tidak pernah dipanggil siapa pun
di pipeline nyata. compute_token_consumption() karena itu murni aritmetika atas
int yang sudah ada, bukan re-encode teks mentah.

PENTING: μ=3 dari proposal (subbab 3.8.2.5), jangan diubah tanpa instruksi
eksplisit peneliti (lihat guide 1.5 poin (e)).
"""

import statistics
from typing import List

MU = 3.0  # HARUS sama dengan PipelineConfig.token_output_weight di src/core/config.py
          # — jangan diubah di sini saja tanpa ikut mengubah config.py, dan sebaliknya.


def compute_token_consumption(token_input: int, token_output: int, mu: float = MU) -> int:
    """
    T = T_in + mu * T_out

    Referensi guide 1.5 poin (c) langkah 3. T_in/T_out di sini adalah COUNT
    yang sudah dihitung sebelumnya (lihat catatan DESAIN di atas) — bukan teks.
    Pola aritmetika sama persis dengan yang sudah dipakai inline di
    src/experiments/ablation.py (`n_t = int(n_in + alpha * n_out)`).
    """
    return int(token_input + mu * token_output)


def aggregate_token_consumption(token_list: List[int]) -> dict:
    """
    Return {"mean": ..., "median": ...} dari list token consumption per query.
    Kedua statistik WAJIB dilaporkan (lihat guide Bagian 4 poin 3 dan Bagian 3 Dimensi 1).

    Caller (src/dimensions/dim1_efficiency.py) bertanggung jawab memecah
    token_list per difficulty level sebelum memanggil ini kalau butuh breakdown
    per-difficulty — fungsi ini sengaja tetap generik atas satu populasi list.
    """
    if not token_list:
        raise ValueError("token_list kosong — tidak ada data untuk diagregasi")
    return {
        "mean": statistics.mean(token_list),
        "median": statistics.median(token_list),
    }
