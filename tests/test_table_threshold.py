"""
Tes lokal threshold tabel per pipeline (IMPLEMENTATION_DECISIONS.md poin 43), tanpa Spider dan tanpa model:
embedder palsu memberi skor tabel tetap 1.00 / 0.62 / 0.30.

    python tests/test_table_threshold.py
"""

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import PipelineConfig, add_config_override_args, apply_config_overrides
from src.retrieval.baseline import TableSchemaIndex, semantic_linking_table_level
from src.retrieval.retrieval import SchemaIndex, retrieve_candidate_tables


class FakeEmbedder:
    """Setiap n-gram -> vektor [1, 0], jadi skor tabel = komponen pertama embedding tabel."""
    def encode(self, texts, convert_to_tensor=True):
        return torch.tensor([[1.0, 0.0]] * len(texts))


def _vec(score: float) -> list:
    return [score, math.sqrt(1 - score ** 2)]


TABLES = ["a", "b", "c"]
EMB = torch.tensor([_vec(1.0), _vec(0.62), _vec(0.30)])


def one_stage(cfg) -> list:
    idx = TableSchemaIndex("db", [f"db.{t}" for t in TABLES], EMB)
    return [n.split(".")[1] for n in semantic_linking_table_level(idx, "singer name", FakeEmbedder(), cfg)]


def two_stage(cfg) -> list:
    idx = SchemaIndex("db", TABLES, EMB, [], [], [], None)
    return retrieve_candidate_tables("singer name", idx, FakeEmbedder(), cfg)


def main():
    cfg = PipelineConfig()
    assert (cfg.table_similarity_threshold, cfg.baseline_table_threshold(), cfg.value_link_numbers) == (0.60, 0.65, True)
    assert two_stage(cfg) == ["a", "b"]            # 0.62 >= 0.60
    assert one_stage(cfg) == ["a"]                 # 0.62 <  0.65

    follow = PipelineConfig(baseline_table_similarity_threshold=None)
    assert follow.baseline_table_threshold() == 0.60 and one_stage(follow) == ["a", "b"]

    p = argparse.ArgumentParser()
    add_config_override_args(p)
    c = apply_config_overrides(PipelineConfig(), p.parse_args(["--baseline-table-threshold", "0.7",
                                                               "--value-link-numbers", "false"]))
    assert c.baseline_table_threshold() == 0.7 and c.value_link_numbers is False
    print("OK: table threshold tests passed")


if __name__ == "__main__":
    main()
