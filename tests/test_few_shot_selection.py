"""
Local check for skeleton few-shot selection (IMPLEMENTATION_DECISIONS.md point 29).
A bag-of-words fake replaces BGE-M3; needs data/spider_data/ (gitignored, local only).

    USE_TF=0 python tests/test_few_shot_selection.py      (or: python -m pytest tests/)
"""

import csv
import json
import os
import sys
import tempfile
import zlib
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("USE_TF", "0")  # local transformers otherwise trips over TensorFlow

import torch  # noqa: E402

from src.core.config import PipelineConfig  # noqa: E402
from src.experiments import ablation  # noqa: E402
from src.generation.few_shot import (  # noqa: E402
    FewShotExample, build_few_shot_index, format_few_shot_block, retrieve_few_shot_examples,
)
from src.generation.generation import build_prompt  # noqa: E402
from src.skeleton import compare_ab  # noqa: E402

SPIDER = Path("data/spider_data")
DEV_Q = "How many singers do we have?"  # dev.json, concert_singer


class FakeEncoder:
    def encode(self, texts, convert_to_tensor=True, **_):
        single = isinstance(texts, str)
        out = torch.zeros(1 if single else len(texts), 512)
        for i, text in enumerate([texts] if single else texts):
            for w in text.lower().split():
                out[i, zlib.crc32(w.encode()) % 512] += 1
        return out[0] if single else out


ENC = FakeEncoder()


@lru_cache(maxsize=None)
def setup():
    q_cfg = PipelineConfig(data_path=SPIDER)
    s_cfg = replace(q_cfg, few_shot_selection="skeleton")
    return q_cfg, build_few_shot_index(q_cfg.train_json, ENC, q_cfg), s_cfg, build_few_shot_index(s_cfg.train_json, ENC, s_cfg)


def raises(exc, fn, *args, match=""):
    try:
        fn(*args)
    except exc as e:
        assert match in str(e), e
        return
    raise AssertionError(f"{exc.__name__} not raised")


def test_question_mode_is_unchanged():
    q_cfg, q_idx, _, _ = setup()
    legacy = build_few_shot_index(q_cfg.train_json, ENC)  # old call, no cfg
    assert legacy.selection == "question" and torch.equal(legacy.embeddings, q_idx.embeddings)
    assert retrieve_few_shot_examples(DEV_Q, legacy, ENC, q_cfg) == retrieve_few_shot_examples(DEV_Q, q_idx, ENC, q_cfg)


def test_skeleton_mode_picks_other_examples_but_prompt_keeps_originals():
    q_cfg, q_idx, s_cfg, s_idx = setup()
    differ = 0
    for question in list(s_idx.query_skeletons)[:20]:
        by_question = retrieve_few_shot_examples(question, q_idx, ENC, q_cfg)
        by_skeleton = retrieve_few_shot_examples(question, s_idx, ENC, s_cfg)
        assert len(by_skeleton) == s_cfg.few_shot_k
        assert all(ex in s_idx.examples for ex in by_skeleton)  # original train question + SQL
        assert "[MASK]" not in format_few_shot_block(by_skeleton)
        differ += by_skeleton != by_question
    assert differ >= 10, differ  # 48/50 with this fake encoder; same-topic hits can coincide


def test_mismatch_and_unknown_question_fail_loudly():
    q_cfg, q_idx, s_cfg, s_idx = setup()
    raises(ValueError, retrieve_few_shot_examples, DEV_Q, q_idx, ENC, s_cfg, match="selection")
    raises(KeyError, retrieve_few_shot_examples, "Not a Spider dev question?", s_idx, ENC, s_cfg, match="masking.py")
    raises(ValueError, lambda: PipelineConfig(few_shot_selection="random"), match="few_shot_selection")


def test_ablation_runs_per_selection_do_not_overwrite():
    q_cfg = PipelineConfig()
    s_cfg = replace(q_cfg, few_shot_selection="skeleton")

    def result(ex):
        return ablation.AblationResult(3, "graphrag", ex, 0, 0, 0, 0, 0, 0, 10, Path("p"), Path("q"))

    original = ablation.RESULTS_CSV
    with tempfile.TemporaryDirectory() as tmp:
        ablation.RESULTS_CSV = Path(tmp) / "ablation_results.csv"
        try:
            ablation._save_results([result(50.0)], q_cfg, "graphrag")
            ablation._save_results([result(60.0)], s_cfg, "graphrag")
            ablation._save_results([result(55.0)], q_cfg, "graphrag")  # rerun replaces only its own rows
            with open(ablation.RESULTS_CSV, newline="", encoding="utf-8") as f:
                rows = {r["few_shot_selection"]: float(r["avg_ex"]) for r in csv.DictReader(f)}
            ex_files = {p.name for p in Path(tmp).glob("ablation_ex_per_k_*.json")}
        finally:
            ablation.RESULTS_CSV = original
    assert rows == {"question": 55.0, "skeleton": 60.0}, rows
    assert ex_files == {"ablation_ex_per_k_graphrag.json", "ablation_ex_per_k_graphrag_skeleton.json"}, ex_files
    assert ablation._run_tag("baseline", q_cfg) == "baseline"
    assert ablation._run_tag("baseline", s_cfg) == "baseline_skeleton"


def test_leak_detection_and_ab_report():
    question = "Which singers are from France?"
    schema = "CREATE TABLE singer (\n  singer_id INT PRIMARY KEY,\n  name TEXT,\n  country TEXT\n);"
    shots = format_few_shot_block([FewShotExample(
        "Names of journalists from England?", 'SELECT name FROM journalist WHERE nationality = "England"', "news_report")])
    prompt = build_prompt(question, schema, {}, shots)

    def row(pred, ex):
        return {"question": question, "prompt": prompt, "pred_sql": pred, "ex": ex, "esm": ex,
                "tokens_in": 100, "token_consumption": 130, "difficulty": "easy"}

    leaky = row('SELECT name FROM singer WHERE nationality = "England"', 0)
    clean = row('SELECT T1.name FROM singer AS T1 WHERE T1.country = "France"', 1)
    invented = row("SELECT age FROM singer", 0)
    assert compare_ab.leak(leaky) == ({"nationality"}, {"nationality"}, {"england"})
    assert compare_ab.leak(clean) == (set(), set(), set())
    assert compare_ab.leak(invented) == ({"age"}, set(), set())

    with tempfile.TemporaryDirectory() as tmp:
        for name, r in (("ablation_graphrag_prompts_k3.jsonl", leaky),
                        ("ablation_graphrag_skeleton_prompts_k3.jsonl", clean)):
            (Path(tmp) / name).write_text(json.dumps(r) + "\n", encoding="utf-8")
        out = compare_ab.compare("graphrag", 3, Path(tmp))
    assert out["ex"]["only_skeleton"] == 1 and out["ex"]["only_question"] == 0
    assert out["leak: name only in examples"] == {"question": 100.0, "skeleton": 0.0}


if __name__ == "__main__":
    if not (SPIDER / "train_spider.json").exists():
        sys.exit("needs data/spider_data/ (local Spider copy)")
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
