"""
token_consumption.py — Token Consumption (T)

Referensi: EVALUATION_ANALYSIS_GUIDE.md Bagian 1.5, sub-bagian "Token Consumption".

PENTING: tokenizer WAJIB Qwen2.5-Coder-7B-Instruct, jangan pakai tiktoken atau
tokenizer lain (lihat guide 1.5 poin (e)).

RESOLVED (sebelumnya ada konflik μ vs α — sudah diputuskan peneliti): guide fix
μ=3 dari proposal (subbab 3.8.2.5). `PipelineConfig.token_output_weight` di
src/core/config.py sudah diupdate ke 3.0 supaya konsisten — itu SEKARANG jadi
satu-satunya sumber kebenaran untuk konstanta ini (lihat aturan "jangan hardcode
nilai, selalu referensikan dari cfg" di CLAUDE.md root). Saat implementasi
compute_token_consumption() di bawah, JANGAN hardcode MU lokal — ambil dari
`cfg.token_output_weight` yang dioper oleh caller (src/dimensions/dim1_efficiency.py),
supaya kalau nilainya berubah lagi di masa depan cukup diubah satu tempat.
Belum ada data ablation nyata yang tersimpan di outputs/ (masih kosong per saat
resolusi ini ditulis), jadi tidak ada data lama yang perlu di-generate ulang.
"""

from typing import List
from dataclasses import dataclass

MU = 3  # HARUS sama dengan PipelineConfig.token_output_weight di src/core/config.py
         # (lihat catatan RESOLVED di atas) — jangan diubah di sini saja tanpa
         # ikut mengubah config.py, dan sebaliknya.

_tokenizer = None  # lazy-loaded singleton


def get_tokenizer():
    """
    Load tokenizer Qwen2.5-Coder-7B-Instruct sekali saja (singleton pattern
    supaya tidak reload berkali-kali tiap query).

    TODO:
    - from transformers import AutoTokenizer
    - AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")
    - simpan ke _tokenizer global, return
    """
    raise NotImplementedError


def count_tokens(text: str) -> int:
    """
    Hitung jumlah token dari suatu teks pakai tokenizer Qwen2.5-Coder-7B-Instruct.
    TODO: tokenizer = get_tokenizer(); return len(tokenizer.encode(text))
    """
    raise NotImplementedError


def compute_token_consumption(prompt_text: str, raw_output_text: str, mu: int = MU) -> int:
    """
    T = T_in + mu * T_out

    Referensi guide 1.5 poin (c) langkah 1-3.

    - prompt_text: SELURUH prompt yang dikirim ke LLM (schema DDL + few-shot + NL question)
    - raw_output_text: output LLM SEBELUM post-processing (lihat guide 1.5 poin (e),
      jangan hitung dari hasil SETELAH post-processing)

    TODO:
    - t_in = count_tokens(prompt_text)
    - t_out = count_tokens(raw_output_text)
    - return t_in + mu * t_out
    """
    raise NotImplementedError


def aggregate_token_consumption(token_list: List[int]) -> dict:
    """
    Return {"mean": ..., "median": ...} dari list token consumption per query.
    Kedua statistik WAJIB dilaporkan (lihat guide Bagian 4 poin 3 dan Bagian 3 Dimensi 1).

    TODO: pakai statistics.mean() dan statistics.median() atau numpy
    """
    raise NotImplementedError
