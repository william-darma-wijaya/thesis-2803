"""
Tes lokal k DAIL-SQL yang bisa diganti, tanpa kode DAIL, Spider, atau LLM: prompt dan generate palsu.
k = 0 hanya menjalankan satu pass (pass 1 akan mengulang prompt yang sama); k > 0 tetap dua pass.

    python tests/test_dail_k_shot.py
"""

import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.retrieval import dail as dail_mod


class FakePrompt:
    def __init__(self, name):
        self.name, self.calls = name, 0

    def format(self, target, **kwargs):
        self.calls += 1
        return {"prompt": f"{self.name} prompt", "n_examples": 0 if self.name == "zero" else 9}


def make(k: int):
    cfg = SimpleNamespace(dail_k_shot=k, dail_max_seq_len=4096, dail_max_ans_len=200, dail_scope_factor=100)
    item = {"db_id": "db", "question": "how many singers?", "query": "SELECT count(*) FROM singer"}
    pre, final = FakePrompt("pre"), FakePrompt("zero" if k == 0 else "final")
    fake_dail = {"process_duplication": lambda s: s, "utils": SimpleNamespace(sql2skeleton=lambda sql, schema: "select _")}
    base = dail_mod.DailBaseline(cfg, fake_dail, None, pre, final, {dail_mod.question_key(item): dict(item)},
                                 {"db": {}}, None)
    return base, item, pre, final


class FakeTokenizer:
    def encode(self, text):
        return text.split()


def main():
    dail_mod._generate = lambda prompt, llm, tokenizer, cfg: ("count(*) FROM singer", 5)

    base, item, pre, final = make(0)
    out = dail_mod.run_single_dail(item, base, None, FakeTokenizer())
    assert pre.calls == 0 and final.calls == 1                       # satu pass saja
    assert list(out["passes"]) == ["pass2"] and out["pre_sql"] is None
    assert (out["n_in"], out["n_out"]) == (2, 5)                     # tidak dihitung dua kali
    assert out["pred"] == "SELECT count(*) FROM singer"

    base, item, pre, final = make(9)
    out = dail_mod.run_single_dail(item, base, None, FakeTokenizer())
    assert pre.calls == 1 and final.calls == 1                       # arsitektur DAIL asli tetap dua pass
    assert list(out["passes"]) == ["pass1", "pass2"] and (out["n_in"], out["n_out"]) == (4, 10)
    print("OK: DAIL k-shot tests passed")


if __name__ == "__main__":
    main()
