"""
Tes lokal sampling stratified per difficulty (IMPLEMENTATION_DECISIONS.md poin 34), tanpa Spider:
SQL di-parse dengan parser resmi SPIDER pada skema mainan, lalu eval_hardness() resmi memberi labelnya.

    python tests/test_sampling.py
"""

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import re

from src.metrics.esm_ex_cm import Schema, get_sql
from src.utils.sampling import difficulty_of, distribution_table, stratified_indices, stratified_sample
import process_sql  # noqa: E402  (importable once esm_ex_cm has put external/spider_eval on sys.path)

try:
    process_sql.word_tokenize("SELECT a FROM b")
except LookupError:  # nltk 'punkt' data not installed locally (Kaggle setup downloads it)
    process_sql.word_tokenize = lambda s: re.findall(r"[\w.]+|[^\w\s]", s)   # keeps T1.name whole, like nltk

SCHEMA = Schema({"singer": ["singer_id", "name", "age", "country"], "concert": ["concert_id", "singer_id", "year"]})
QUERIES = {
    "easy": "SELECT count(*) FROM singer",
    "medium": "SELECT name FROM singer WHERE age > 20 ORDER BY age",
    "hard": "SELECT name FROM singer WHERE age > 20 INTERSECT SELECT name FROM singer WHERE age < 40",
    "extra": "SELECT T1.name, count(*) FROM singer AS T1 JOIN concert AS T2 ON T1.singer_id = T2.singer_id "
             "WHERE T1.age > 20 GROUP BY T1.name HAVING count(*) > 1 ORDER BY count(*) DESC LIMIT 3",
}
SIZES = {"easy": 25, "medium": 43, "hard": 17, "extra": 15}   # ~ proporsi dev Spider


def main():
    items = [{"question": f"{lv} {i}", "sql": get_sql(SCHEMA, QUERIES[lv]), "level": lv}
             for lv, n in SIZES.items() for i in range(n)]
    labels = difficulty_of(items)
    assert labels == [it["level"] for it in items], Counter(zip(labels, [it["level"] for it in items]))

    idx = stratified_indices(items, 0.2, seed=42)
    got = Counter(items[i]["level"] for i in idx)
    assert got == {lv: max(1, round(0.2 * n)) for lv, n in SIZES.items()}, got   # 5 / 9 / 3 / 3
    assert idx == sorted(idx) and len(set(idx)) == len(idx)
    assert idx == stratified_indices(items, 0.2, seed=42)                          # reproducible
    assert idx != stratified_indices(items, 0.2, seed=7)
    assert stratified_indices(items, 1.0, seed=42) == list(range(len(items)))
    assert len(stratified_sample(items, 0.01, seed=42)) == 4                        # min 1 per level

    rows = {r["difficulty"]: r for r in distribution_table(items, idx)}
    assert rows["medium"]["full_n"] == 43 and rows["medium"]["sample_n"] == 9
    assert all(abs(r["full_share"] - r["sample_share"]) < 0.03 for r in rows.values())
    print("OK: stratified sampling tests passed", dict(got))


if __name__ == "__main__":
    main()
