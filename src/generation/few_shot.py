"""
Few-shot example retrieval for the GraphRAG Text-to-SQL pipeline.

Strategy: similarity-based retrieval.
  - At startup, encode ALL training questions with the same BGE-M3 embedding model.
  - At inference, find the top-k training questions most similar to the current
    question using cosine similarity.
  - Return their (question, gold_sql) pairs as few-shot examples in the prompt.

Ranking murni by cosine similarity — tidak ada prioritas berdasarkan db_id.
(Mode "same-DB-first" pernah ada, dihapus 2026-09-12 karena provably dead code
untuk Spider: train/dev disjoint total, lihat IMPLEMENTATION_DECISIONS.md poin 22.)
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer, util

from src.core.config import PipelineConfig

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
    index: FewShotIndex,
    embed_model: SentenceTransformer,
    cfg: PipelineConfig,
) -> list[FewShotExample]:
    """
    Return the top-k most similar training examples for `question`, ranked
    purely by cosine similarity.

    REMOVED (2026-09-12, lihat context/IMPLEMENTATION_DECISIONS.md poin 22):
    dulu ada mode "same-DB-first" yang mempromosikan contoh dari database yang
    sama ke atas top-k, dengan asumsi "Spider train/dev splits share database
    IDs". Asumsi itu salah — Spider itu cross-domain benchmark, train (140 DB)
    dan dev (20 DB) sengaja disjoint total (diverifikasi langsung: 0 overlap).
    Jadi cabang same-DB itu PROVABLY selalu kosong untuk evaluasi Spider dev
    standar manapun juga — dead code yang tidak pernah bisa menyala, bukan cuma
    kebetulan nol di satu run. Dihapus, bukan cuma didokumentasikan, supaya
    tidak ada over-fetch pool (`k*4`) yang costly tanpa efek apapun.
    """
    if cfg.few_shot_k == 0 or index is None:
        return []

    query_emb = embed_model.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, index.embeddings)[0]  # (n_train,)

    top_indices = torch.topk(scores, cfg.few_shot_k).indices.tolist()
    return [index.examples[i] for i in top_indices]


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

def format_few_shot_block(examples: list[FewShotExample]) -> str:
    """
    Render few-shot examples as a clean prompt block.

    Format per example:
        -- Example N
        -- Q: <question>
        -- A:
        SELECT ...
    """
    if not examples:
        return ""

    lines = ["### Examples\n"]
    for i, ex in enumerate(examples, start=1):
        lines.append(f"-- Example {i}")
        lines.append(f"-- Q: {ex.question}")
        lines.append("-- A:")
        lines.append(ex.sql.strip())
        lines.append("")  # blank line between examples

    return "\n".join(lines) + "\n"
