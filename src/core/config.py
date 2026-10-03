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
    predictions_file: Path = Path("outputs/predictions/predictions.txt")

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
    llm_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct"
    # Stage 3 of the ablation study: every model here is run (GraphRAG + Baseline)
    # with the best top-k / few-shot config from stages 1-2.
    # `python src/experiments/pipeline.py --baseline --models all`
    llm_model_frame: list[str] = field(default_factory=lambda: [
        "Qwen/Qwen2.5-Coder-1.5B-Instruct",
        "Qwen/Qwen2.5-Coder-3B-Instruct",
        "Qwen/Qwen2.5-Coder-7B-Instruct",
        "Qwen/Qwen2.5-Coder-14B-Instruct",
    ])

    # --- Quantization ---
    load_in_4bit: bool = True
    bnb_double_quant: bool = True
    bnb_quant_type: str = "nf4"
    bnb_compute_dtype: torch.dtype = torch.float16

    # --- Schema Linking ---
    # Minimum cosine similarity (best n-gram match) for a column to be kept.
    # Uncalibrated for BGE-M3 -- do not trust 0.35 blindly (sweep.py --thresholds).
    semantic_similarity_threshold: float = 0.35
    # Stage 1 counterpart: minimum cosine similarity (best n-gram match) for a TABLE
    # to become a candidate. 0.0 = disabled (only top_k_tables_pct limits tables).
    # Tuned separately from the column threshold (the score scales differ).
    table_similarity_threshold: float = 0.0
    max_ngram: int = 3

    # Top-k is a FRACTION in (0, 1], not an absolute count, so the budget scales
    # with database size. Converted per query by retrieval.k_from_pct():
    #   tables  : ceil(pct x number of tables in the database)
    #   columns : ceil(pct x number of columns in the Stage-1 candidate tables)
    # Always at least 1. Set to 0 to disable (all tables / no column cap).
    # The column count is a cap, not a target: fewer columns passing the
    # similarity threshold -> fewer are kept.
    top_k_tables_pct: float = 0.6
    top_k_columns_pct: float = 0.6
    # Baseline has no column stage, so only its table fraction is tuned
    # (separately from GraphRAG's, by the same SLA sweep).
    baseline_top_k_tables_pct: float = 0.6

    # Stage 1 of the ablation study (sweep.py, no LLM, ranked by SLA):
    # GraphRAG = top_k_frame x top_k_frame (tables x columns), Baseline = top_k_frame.
    top_k_frame: list[float] = field(default_factory=lambda: [0.4, 0.5, 0.6, 0.8])

    # --- Few-shot ---
    # Number of similar training examples injected into the prompt. Used by
    # BOTH GraphRAG and Baseline -- the two arms always get the same k.
    # Set to 0 to disable few-shot entirely.
    few_shot_k: int = 3
    # Stage 2 of the ablation study (ablation.py, ranked by EX).
    few_shot_frame: list[int] = field(default_factory=lambda: [0, 1, 3, 5])

    # --- Generation ---
    max_new_tokens: int = 256
    temperature: float = 0.0

    # --- Mode ---
    use_full_schema_bypass: bool = False

    # --- Token consumption ---
    # α in T = T_in + α × T_out (called μ in the thesis proposal, subbab 3.8.2.5).
    # Fixed at 3.0 per proposal — do not change without an explicit decision from
    # the researcher (see context/EVALUATION_ANALYSIS_GUIDE.md Bagian 1.5).
    token_output_weight: float = 3.0

    # --- Reproducibility ---
    seed: int = 42

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Catch old absolute top-k values (e.g. 3) left in the pct fields."""
        for name in ("top_k_tables_pct", "top_k_columns_pct", "baseline_top_k_tables_pct"):
            value = getattr(self, name)
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be a fraction in [0, 1] (0 = disabled), got {value!r}")
        if any(not 0 < p <= 1 for p in self.top_k_frame):
            raise ValueError(f"top_k_frame values must be in (0, 1], got {self.top_k_frame!r}")
        if self.few_shot_k < 0 or any(k < 0 for k in self.few_shot_frame):
            raise ValueError("few_shot_k / few_shot_frame must be >= 0")


# CLI overrides shared by pipeline.py and ablation.py, so a Kaggle run can use
# the stage 1/2 winners before they are committed to the defaults above.
_CLI_OVERRIDES = (
    ("--tables-pct", "top_k_tables_pct", float),
    ("--columns-pct", "top_k_columns_pct", float),
    ("--baseline-tables-pct", "baseline_top_k_tables_pct", float),
    ("--threshold", "semantic_similarity_threshold", float),
    ("--table-threshold", "table_similarity_threshold", float),
    ("--few-shot-k", "few_shot_k", int),
    ("--llm-model", "llm_model", str),
)


def add_config_override_args(parser) -> None:
    for flag, attr, typ in _CLI_OVERRIDES:
        parser.add_argument(flag, dest=attr, type=typ, default=None,
                            help=f"Override config.py {attr}.")


def apply_config_overrides(cfg: PipelineConfig, args) -> PipelineConfig:
    for _, attr, _ in _CLI_OVERRIDES:
        value = getattr(args, attr, None)
        if value is not None:
            setattr(cfg, attr, value)
    cfg.validate()
    return cfg
