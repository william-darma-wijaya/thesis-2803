"""
raw_logs.py — helper generik untuk operasi atas raw_logs (data/raw_logs/*.json),
dipakai lintas src/dimensions/*.py.

DIFFICULTY_LEVELS dan group_by_difficulty() awalnya ditulis lokal di
dim1_efficiency.py (dengan catatan "extract nanti kalau dim2 butuh pola yang
sama, bukan sekarang"). dim2_structure.py sekarang butuh pola grouping yang
identik, jadi diextract ke sini -- single source of truth untuk grouping
per-difficulty di seluruh src/dimensions/.

Label difficulty ikuti konvensi resmi SPIDER: "easy"/"medium"/"hard"/"extra"
(BUKAN "extra_hard" seperti tertulis di guide Bagian 2 -- lihat
context/IMPLEMENTATION_DECISIONS.md poin 6).
"""

from typing import Dict, List

DIFFICULTY_LEVELS = ["easy", "medium", "hard", "extra"]


def group_by_difficulty(logs: List[dict]) -> Dict[str, List[dict]]:
    """Kelompokkan raw_logs entries per difficulty level. Entry dengan
    difficulty di luar DIFFICULTY_LEVELS (seharusnya tidak pernah terjadi,
    lihat catatan di atas) diam-diam di-skip, bukan crash."""
    groups: Dict[str, List[dict]] = {level: [] for level in DIFFICULTY_LEVELS}
    for entry in logs:
        d = entry.get("difficulty")
        if d in groups:
            groups[d].append(entry)
    return groups
