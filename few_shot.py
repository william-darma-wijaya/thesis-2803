"""
Few-shot example retrieval for the GraphRAG Text-to-SQL pipeline.

Strategy: similarity-based retrieval.
  - At startup, encode ALL training questions with the same BGE-M3 embedding model.
  - At inference, find the top-k training questions most similar to the current
    question using cosine similarity.
  - Return their (question, gold_sql) pairs as few-shot examples in the prompt.

Optionally, examples from the same database are prioritised (same-DB-first mode),
since those examples are guaranteed to share at least some schema vocabulary.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util

from config import PipelineConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FewShotExample:
    question: str
    sql: str
    db_id: str


@dataclass
class FewShotIndex:
    """Precomputed embedding index over the entire training set."""
    examples: list[FewShotExample]          # parallel to embeddings
    embeddings: torch.Tensor                # shape (n_train, d)


# ---------------------------------------------------------------------------
# Index construction  (called ONCE at startup)
# ---------------------------------------------------------------------------

def build_few_shot_index(
    train_json_path: Path,
    embed_model: SentenceTransformer,
) -> FewShotIndex:
    """
    Load the Spider training set and encode every question.

    This is the only expensive step — it runs once and the result is reused
    for every dev question.
    """
    with open(train_json_path, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    examples = [
        FewShotExample(
            question=item["question"],
            sql=item["query"],
            db_id=item["db_id"],
        )
        for item in train_data
    ]

    logger.info("Encoding %d training questions for few-shot index …", len(examples))
    embeddings = embed_model.encode(
        [e.question for e in examples],
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=256,
    )

    logger.info("Few-shot index ready.")
    return FewShotIndex(examples=examples, embeddings=embeddings)


# ---------------------------------------------------------------------------
# Retrieval  (called per question)
# ---------------------------------------------------------------------------

def retrieve_few_shot_examples(
    question: str,
    db_id: str,
    index: FewShotIndex,
    embed_model: SentenceTransformer,
    cfg: PipelineConfig,
) -> list[FewShotExample]:
    """
    Return the top-k most similar training examples for `question`.

    If cfg.few_shot_same_db_first is True, examples from the same database
    are ranked first among the top-k candidates, giving the LLM schema-familiar
    examples when available.

    The current question's own database is NOT excluded — Spider train/dev splits
    share database IDs, so same-DB examples are often the most relevant.
    """
    if cfg.few_shot_k == 0 or index is None:
        return []

    query_emb = embed_model.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, index.embeddings)[0]  # (n_train,)

    # Retrieve a larger candidate pool first if same-DB-first is enabled,
    # so we have enough candidates to rerank after partitioning by DB.
    # Edge case: if the DB has fewer training examples than k, the same-DB
    # partition will be exhausted and we pad from other DBs — this is fine
    # and expected. The pool cap ensures we never request more than available.
    pool_size = cfg.few_shot_k * 4 if cfg.few_shot_same_db_first else cfg.few_shot_k
    pool_size = min(pool_size, len(index.examples))

    top_indices = torch.topk(scores, pool_size).indices.tolist()
    candidates = [index.examples[i] for i in top_indices]

    if cfg.few_shot_same_db_first:
        # Stable rerank: same-DB examples float to the top, order within
        # each group is preserved (i.e. still sorted by similarity score).
        same_db = [e for e in candidates if e.db_id == db_id]
        other_db = [e for e in candidates if e.db_id != db_id]
        candidates = same_db + other_db

    return candidates[: cfg.few_shot_k]


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_few_shot_block(examples: list[FewShotExample]) -> str:
    """
    Render few-shot examples as a clean prompt block.

    Format per example:
        -- Example N
        -- Q: <question>
        SELECT ...
    """
    if not examples:
        return ""

    lines = ["### Examples\n"]
    for i, ex in enumerate(examples, start=1):
        lines.append(f"-- Example {i}")
        lines.append(f"-- Q: {ex.question}")
        lines.append(ex.sql.strip())
        lines.append("")  # blank line between examples

    return "\n".join(lines) + "\n"
