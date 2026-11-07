"""
sampling.py — sampel dev set stratified per tingkat kesulitan (kode custom skripsi).

Keputusan peneliti 2026-11-03 (IMPLEMENTATION_DECISIONS.md poin 34): "20%" = 20% dari
TIAP tingkat kesulitan, supaya proporsi easy/medium/hard/extra sampel sama dengan dev
set penuh. Satu fungsi untuk semua tempat yang mengambil sampel (trial SLA, sweep.py,
ablation.py, pipeline.py, baseline.py, eval_pipeline.ipynb), jadi dengan ratio + seed
yang sama semuanya memakai query yang SAMA.

Difficulty = Evaluator.eval_hardness() resmi SPIDER pada field `sql` (gold SQL yang
sudah di-parse oleh pembuat Spider) di dev.json — label yang sama dengan `difficulty`
di raw_logs, tanpa perlu membuka database.
"""

from collections import Counter, defaultdict

import numpy as np

from src.metrics.esm_ex_cm import Evaluator

LEVELS = ["easy", "medium", "hard", "extra"]


def difficulty_of(items: list[dict]) -> list[str]:
    ev = Evaluator()
    return [ev.eval_hardness(item["sql"]) for item in items]


def stratified_indices(items: list[dict], ratio: float, seed: int) -> list[int]:
    """
    Indeks sampel (urut naik). Per tingkat: round(ratio x jumlah), minimal 1, diambil
    acak tanpa pengembalian dengan RNG per tingkat (seed, urutan tingkat) — jadi sampel
    satu tingkat tidak berubah kalau tingkat lain berubah. ratio >= 1 -> semua indeks.
    """
    if ratio >= 1.0:
        return list(range(len(items)))
    groups: dict[str, list[int]] = defaultdict(list)
    for i, level in enumerate(difficulty_of(items)):
        groups[level].append(i)
    out = []
    for j, level in enumerate(LEVELS + sorted(set(groups) - set(LEVELS))):
        idx = groups.get(level, [])
        if not idx:
            continue
        n = min(len(idx), max(1, round(ratio * len(idx))))
        out += np.random.default_rng([seed, j]).choice(idx, size=n, replace=False).tolist()
    return sorted(out)


def stratified_sample(items: list[dict], ratio: float, seed: int) -> list[dict]:
    return [items[i] for i in stratified_indices(items, ratio, seed)]


def distribution_table(items: list[dict], indices: list[int]) -> list[dict]:
    """Jumlah & proporsi per tingkat: dev penuh vs sampel (bahan justifikasi sampling bias)."""
    levels = difficulty_of(items)
    full, samp = Counter(levels), Counter(levels[i] for i in indices)
    return [{"difficulty": lv, "full_n": full[lv], "full_share": full[lv] / len(items),
             "sample_n": samp[lv], "sample_share": samp[lv] / len(indices) if indices else 0.0}
            for lv in LEVELS if full[lv]]
