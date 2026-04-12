"""
Configuration for GraphRAG Text-to-SQL Pipeline.
"""

import torch
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class PipelineConfig:
    # --- Paths ---
    data_path: Path = Path("/kaggle/input/datasets/alrette/spiderdataset/spider_data")
    predictions_file: Path = Path("predictions.txt")

    @property
    def tables_json(self) -> Path:
        return self.data_path / "tables.json"

    @property
    def dev_json(self) -> Path:
        return self.data_path / "dev.json"

    @property
    def db_dir(self) -> Path:
        return self.data_path / "database"

    @property
    def gold_sql(self) -> Path:
        p = self.data_path / "dev_gold.sql"
        return p if p.exists() else self.data_path / "dev_gold"

    @property
    def train_json(self) -> Path:
        return self.data_path / "train_spider.json"

    # --- Models ---
    embedding_model: str = "BAAI/bge-m3"
    llm_model: str = "Qwen/Qwen2.5-Coder-14B-Instruct"

    # --- Quantization ---
    load_in_4bit: bool = True
    bnb_double_quant: bool = True
    bnb_quant_type: str = "nf4"
    bnb_compute_dtype: torch.dtype = torch.bfloat16

    # --- Schema Linking ---
    semantic_similarity_threshold: float = 0.35
    max_ngram: int = 3

    # Top-k: keep only the k best column matches per query n-gram.
    # Set to 0 to disable (falls back to threshold-only filtering).
    top_k_columns: int = 5

    # Two-stage retrieval: first select candidate tables, then restrict
    # column retrieval to only those tables.
    # Set to 0 to disable (all tables are considered in one pass).
    top_k_tables: int = 3

    # --- Few-shot ---
    # Number of similar training examples injected into the prompt.
    # Set to 0 to disable few-shot entirely.
    few_shot_k: int = 3
    # If True, examples from the same database are ranked first among top-k.
    few_shot_same_db_first: bool = True

    # --- Generation ---
    max_new_tokens: int = 200
    temperature: float = 0.0

    # --- Mode ---
    use_full_schema_bypass: bool = False

    # --- Reproducibility ---
    seed: int = 42
