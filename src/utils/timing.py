"""
timing.py — pencatat latensi per tahap (kode custom skripsi, bukan bagian SPIDER).

step() dipasang di dalam fungsi pipeline dan tidak melakukan apa pun kecuali sedang ada
collect(), jadi ablation.py, sweep.py, dan pemanggil lain tidak berubah. Hanya mencatat
waktu: prompt, SQL, dan metrik tidak terpengaruh. Lihat IMPLEMENTATION_DECISIONS.md poin 36.
"""

import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

RETRIEVAL_STEPS = ("retrieval_search", "build_ddl")
END_TO_END_STEPS = RETRIEVAL_STEPS + ("few_shot", "build_prompt", "prompt_tokenize", "llm_generate", "sql_clean")
RUN_FILENAME = "latency_run.json"

# Waktu tahap sekali-jalan (setup, evaluator resmi, dimensi). Dijalankan ulang = ditimpa.
RUN: dict = {}
_query_steps = None
_run_start = None


def _sync_gpu() -> None:
    """Operasi GPU bisa asinkron; tunggu selesai supaya waktunya masuk tahap yang benar."""
    torch = sys.modules.get("torch")
    if torch is not None and torch.cuda.is_available():
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)


@contextmanager
def collect():
    """Kumpulkan semua step() di dalam blok ini ke satu dict {tahap: detik}."""
    global _query_steps
    previous, _query_steps = _query_steps, {}
    try:
        yield _query_steps
    finally:
        _query_steps = previous


@contextmanager
def step(name: str):
    """Satu tahap per pertanyaan. Tanpa collect() yang aktif, tidak melakukan apa pun."""
    steps = _query_steps
    if steps is None:
        yield
        return
    _sync_gpu()
    start = time.perf_counter()
    try:
        yield
    finally:
        _sync_gpu()
        steps[name] = steps.get(name, 0.0) + time.perf_counter() - start


@contextmanager
def run_step(name: str):
    """Satu tahap sekali-jalan, disimpan ke RUN. Bisa dipakai sebagai `with` atau dekorator."""
    global _run_start
    _sync_gpu()
    start = time.perf_counter()
    if _run_start is None:
        _run_start = start
    try:
        yield
    finally:
        _sync_gpu()
        RUN[name] = time.perf_counter() - start


def reset_run() -> None:
    """Dipanggil di awal run CLI, karena `--models all` menjalankan beberapa model dalam satu proses."""
    global _run_start
    RUN.clear()
    _run_start = None


def finish_query(steps: dict | None) -> dict | None:
    """Tambahkan retrieval_total dan end_to_end. Pencatatan token dan proxy tidak ikut dihitung."""
    if steps is None:
        return None
    return {
        **steps,
        "retrieval_total": sum(steps.get(s, 0.0) for s in RETRIEVAL_STEPS),
        "end_to_end": sum(steps.get(s, 0.0) for s in END_TO_END_STEPS),
    }


def gpu_name() -> str | None:
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return None
    return f"{torch.cuda.device_count()} x {torch.cuda.get_device_name(0)}"


def _peak_vram_gb() -> float | None:
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_available():
        return None
    return sum(torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count())) / 1024 ** 3


def save_run(tables_dir: Path) -> Path:
    """Tulis waktu sekali-jalan, total waktu, GPU, dan puncak VRAM ke latency_run.json."""
    path = Path(tables_dir) / RUN_FILENAME
    elapsed = None if _run_start is None else time.perf_counter() - _run_start
    path.write_text(json.dumps({"steps": RUN, "elapsed_s": elapsed, "gpu": gpu_name(),
                                "peak_vram_gb": _peak_vram_gb()}, indent=2), encoding="utf-8")
    return path
