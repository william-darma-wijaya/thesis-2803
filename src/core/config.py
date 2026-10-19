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
    # How stopwords are handled when the question is split into n-grams (GraphRAG
    # Stage 1 + 2; Baseline embeds the whole question and ignores this):
    #   "none"   : keep every n-gram (original behaviour)
    #   "remove" : drop stopwords from the word list FIRST, then build n-grams
    #              (creates adjacency that is not in the question)
    #   "edge"   : build n-grams from the original word order, then drop every
    #              n-gram that STARTS or ENDS with a stopword (inner stopwords stay:
    #              "number of singers" survives, "of singers" does not)
    # Deliberately NOT the NLTK list: it contains words that are meaningful in
    # Spider column names (from, to, no, not, all, before, after, between ...).
    query_stopword_mode: str = "none"
    query_stopwords: list[str] = field(default_factory=lambda: [
        "what", "which", "who", "whom", "whose", "how", "when", "where",
        "show", "list", "give", "find", "tell", "return", "display", "me",
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "do", "does", "did",
        "of", "for", "with", "and", "or", "that", "this", "these", "those",
        "have", "has", "had", "their", "its", "it", "there",
    ])

    # Which nodes of the traversal paths go into the prompt (GraphRAG):
    #   "full"    : every node on every shortest path between anchors (original).
    #               Paths run through primary keys (belongs_to_pk edges), so two
    #               anchors of ONE table pull in that table's PK even without a join
    #   "fk_only" : anchors + only the two endpoints of each foreign_key edge on
    #               the paths, i.e. the JOIN columns. Same paths, fewer nodes
    traversal_mode: str = "full"
    # Tables without a primary key (junction tables such as pets_1.Has_Pet) get no
    # belongs_to_pk edges, so their columns are isolated and traversal cannot cross
    # them (Student -> Has_Pet -> Pets has no path). True = connect the columns of
    # each PK-less table to each other ("same_table" edges) in the TRAVERSAL graph
    # only; the schema graph from schema.py and the DDL stay unchanged.
    connect_no_pk_tables: bool = False
    # Embedding text for Stage 1/2: False = humanized ORIGINAL names ("stuid");
    # True = Spider's annotated natural names from tables.json ("student id").
    # SQL, DDL and node ids always use the original names.
    embed_natural_names: bool = False

    # --- Value linking (GraphRAG, exact match; src/retrieval/value_linking.py) ---
    # Question n-grams that exactly equal a cell value ("asia") add that column
    # (country.Continent) as an extra traversal anchor. Index built once per DB.
    use_value_linking: bool = False
    value_max_rows_per_column: int = 2000    # SELECT DISTINCT ... LIMIT N per column
    value_min_chars: int = 3                 # shorter values are ignored
    value_max_chars: int = 50                # longer values ignored; column skipped if mean length exceeds it
    value_max_numeric_ratio: float = 0.5     # column skipped if more values than this are pure numbers
    value_max_columns_per_value: int = 3     # value in more columns than this = too common, ignored
    value_max_linked_columns: int = 5        # max columns added per question
    # Numbers ("played in 2013" -> matches.year). Off by default; needs use_value_linking.
    value_link_numbers: bool = False
    value_max_columns_per_number: int = 2    # stricter than text: small numbers sit in many columns
    value_number_skip_after: list[str] = field(default_factory=lambda: [
        "top", "first", "last", "limit",       # "top 3" is a LIMIT, not a cell value
    ])

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
    # What the cosine similarity compares when picking examples:
    #   "question" -- raw question text (picks examples on the same topic)
    #   "skeleton" -- de-semanticized question from src/skeleton/masking.py
    #                 (picks examples with the same intent / SQL structure)
    # The prompt always shows the ORIGINAL question + SQL of each example.
    # Stays "question" until the EX A/B (IMPLEMENTATION_DECISIONS.md point 29).
    few_shot_selection: str = "question"
    skeleton_dir: Path = Path("data/skeletons")

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
        if self.query_stopword_mode not in ("none", "remove", "edge"):
            raise ValueError(f"query_stopword_mode must be none|remove|edge, got {self.query_stopword_mode!r}")
        if self.traversal_mode not in ("full", "fk_only"):
            raise ValueError(f"traversal_mode must be full|fk_only, got {self.traversal_mode!r}")
        if self.few_shot_selection not in ("question", "skeleton"):
            raise ValueError(f"few_shot_selection must be 'question' or 'skeleton', got {self.few_shot_selection!r}")


# CLI overrides shared by pipeline.py and ablation.py, so a Kaggle run can use
# the stage 1/2 winners before they are committed to the defaults above.
_CLI_OVERRIDES = (
    ("--tables-pct", "top_k_tables_pct", float),
    ("--columns-pct", "top_k_columns_pct", float),
    ("--baseline-tables-pct", "baseline_top_k_tables_pct", float),
    ("--threshold", "semantic_similarity_threshold", float),
    ("--table-threshold", "table_similarity_threshold", float),
    ("--stopword-mode", "query_stopword_mode", str),
    ("--traversal-mode", "traversal_mode", str),
    ("--few-shot-k", "few_shot_k", int),
    ("--few-shot-selection", "few_shot_selection", str),
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
