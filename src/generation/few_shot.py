"""
Few-shot example retrieval for the GraphRAG Text-to-SQL pipeline.

Strategy: similarity-based retrieval.
  - At startup, encode ALL training questions with the same BGE-M3 embedding model.
  - At inference, find the top-k training questions most similar to the current
    question using cosine similarity.
  - Return their (question, gold_sql) pairs as few-shot examples in the prompt.

With cfg.few_shot_selection="skeleton" the similarity is computed on
de-semanticized questions (data/skeletons/, src/skeleton/masking.py) instead,
so examples are picked by intent rather than topic. The returned examples, and
so the prompt, still carry the ORIGINAL question + SQL.

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
    selection: str = "question"             # what the embeddings encode (cfg.few_shot_selection)
    query_skeletons: dict[str, str] | None = None  # target question -> skeleton, for "skeleton"


# ---------------------------------------------------------------------------
# Index construction  (called ONCE at startup)
# ---------------------------------------------------------------------------

def build_few_shot_index(
    train_json_path: Path,
    embed_model: SentenceTransformer,
    cfg: PipelineConfig | None = None,
) -> FewShotIndex:
    """
    Load the Spider training set and encode every question (or its skeleton,
    when cfg.few_shot_selection == "skeleton"; no cfg = "question").

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

    selection = cfg.few_shot_selection if cfg else "question"
    texts, query_skeletons = [e.question for e in examples], None
    if selection == "skeleton":
        train_sk = json.loads((cfg.skeleton_dir / "train.json").read_text(encoding="utf-8"))
        if [r["question"] for r in train_sk] != texts:
            raise ValueError(f"{cfg.skeleton_dir / 'train.json'} is out of sync with {train_json_path}; "
                             "rerun src/skeleton/masking.py")
        texts = [r["skeleton"] for r in train_sk]
        dev_sk = json.loads((cfg.skeleton_dir / "dev.json").read_text(encoding="utf-8"))
        query_skeletons = {r["question"]: r["skeleton"] for r in dev_sk}

    logger.info("Encoding %d training %ss for few-shot index …", len(examples), selection)
    embeddings = embed_model.encode(
        texts,
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=256,
    )

    logger.info("Few-shot index ready (selection=%s).", selection)
    return FewShotIndex(examples=examples, embeddings=embeddings,
                        selection=selection, query_skeletons=query_skeletons)


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
    if index.selection != cfg.few_shot_selection:
        raise ValueError(f"few-shot index was built with selection={index.selection!r} but "
                         f"cfg.few_shot_selection={cfg.few_shot_selection!r}; pass cfg to build_few_shot_index()")

    query = question
    if index.selection == "skeleton":
        if question not in index.query_skeletons:
            raise KeyError(f"No skeleton for question {question!r} in {cfg.skeleton_dir / 'dev.json'}; "
                           "rerun src/skeleton/masking.py")
        query = index.query_skeletons[question]

    query_emb = embed_model.encode(query, convert_to_tensor=True)
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


def build_few_shot_block(
    question: str,
    index: FewShotIndex | None,
    embed_model: SentenceTransformer,
    cfg: PipelineConfig,
) -> str:
    """
    Retrieve + format in one call. Used by BOTH GraphRAG (run_single) and
    Baseline (run_single_baseline), so the two arms always get the same
    cfg.few_shot_k examples for the same question. "" when k=0 or no index.
    """
    if index is None or cfg.few_shot_k <= 0:
        return ""
    return format_few_shot_block(retrieve_few_shot_examples(question, index, embed_model, cfg))
