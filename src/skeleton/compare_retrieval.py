"""
No-LLM check for skeleton retrieval (IMPLEMENTATION_DECISIONS.md point 29, step 1):
are few-shot examples retrieved by SKELETON similarity closer in SQL structure
to the gold SQL than examples retrieved by raw QUESTION similarity (what
few_shot.py does today)? If not, skeleton retrieval is not worth a GPU run.

For every dev question, top-k train examples are retrieved both ways with the
same model (cfg.embedding_model, BGE-M3) and compared with the dev gold SQL:
  struct_jaccard : Jaccard of SQL keyword features (JOIN, GROUP BY, ORDER BY,
                   LIMIT, COUNT, subquery, EXCEPT, ...) -- custom proxy
  exact_struct   : share of the k examples with exactly the same feature set
  same_hardness  : share with the same Spider hardness (official SPIDER
                   Evaluator.eval_hardness on the parsed `sql` in the json)
Paired Wilcoxon signed-rank test over dev questions, skeleton vs question.

Kaggle (GPU):  python src/skeleton/compare_retrieval.py
Output: printed report + outputs/tables/skeleton_retrieval_check.json
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "external" / "spider_eval"))

import torch  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

from evaluation import Evaluator  # noqa: E402  (official SPIDER)
from src.core.config import PipelineConfig  # noqa: E402

FEATURES = {name: re.compile(pattern, re.IGNORECASE | re.DOTALL) for name, pattern in {
    "join": r"\bjoin\b", "where": r"\bwhere\b", "group_by": r"\bgroup\s+by\b",
    "having": r"\bhaving\b", "order_by": r"\border\s+by\b", "limit": r"\blimit\b",
    "desc": r"\bdesc\b", "distinct": r"\bdistinct\b", "like": r"\blike\b",
    "between": r"\bbetween\b", "not_in": r"\bnot\s+in\b", "or": r"\bor\b",
    "count": r"\bcount\s*\(", "avg": r"\bavg\s*\(", "sum": r"\bsum\s*\(",
    "max": r"\bmax\s*\(", "min": r"\bmin\s*\(", "subquery": r"\(\s*select\b",
    "union": r"\bunion\b", "intersect": r"\bintersect\b", "except": r"\bexcept\b",
}.items()}

# Guo et al. (2023) Fig. 1. Q1 is in Spider train (theme_gallery); Q2/Q3 are not.
# Skeletons precomputed with masking.py (theme_gallery / concert_singer schemas).
FIG1 = [
    ("What are the themes and years for exhibitions, sorted by ticket price descending?",
     "What are the [MASK] and [MASK] for [MASK], sorted by [MASK] [MASK] descending?"),
    ("What are the names and ages for every singer in descending order of age?",
     "What are the [MASK] and [MASK] for every [MASK] in descending order of [MASK]?"),
    ("Where are the themes for exhibitions, and how much is the ticket price?",
     "Where are the [MASK] for [MASK], and how much is the [MASK] [MASK]?"),
]


def features(sql: str) -> frozenset[str]:
    return frozenset(name for name, pattern in FEATURES.items() if pattern.search(sql))


def jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


METRICS = {
    "struct_jaccard": lambda d, t: jaccard(d["features"], t["features"]),
    "exact_struct": lambda d, t: float(d["features"] == t["features"]),
    "same_hardness": lambda d, t: float(d["hardness"] == t["hardness"]),
}


def mean(xs) -> float:
    return sum(xs) / len(xs)


def load_split(skeleton_path: Path, spider_path: Path, evaluator: Evaluator) -> list[dict]:
    rows = json.loads(skeleton_path.read_text(encoding="utf-8"))
    spider = json.loads(spider_path.read_text(encoding="utf-8"))
    if [r["question"] for r in rows] != [x["question"] for x in spider]:
        raise SystemExit(f"{skeleton_path} is out of sync with {spider_path}; rerun src/skeleton/masking.py")
    for r, x in zip(rows, spider):
        r["hardness"] = evaluator.eval_hardness(x["sql"])
        r["features"] = features(r["query"])
    return rows


def main() -> None:
    cfg = PipelineConfig()
    ap = argparse.ArgumentParser(description="Skeleton vs question similarity for few-shot retrieval (no LLM).")
    ap.add_argument("--spider-dir", type=Path, default=cfg.data_path)
    ap.add_argument("--skeleton-dir", type=Path, default=Path("data/skeletons"))
    ap.add_argument("--k", type=int, nargs="+", default=[k for k in cfg.few_shot_frame if k > 0])
    ap.add_argument("--out", type=Path, default=Path("outputs/tables/skeleton_retrieval_check.json"))
    args = ap.parse_args()

    evaluator = Evaluator()
    train = load_split(args.skeleton_dir / "train.json", args.spider_dir / "train_spider.json", evaluator)
    dev = load_split(args.skeleton_dir / "dev.json", args.spider_dir / "dev.json", evaluator)

    model = SentenceTransformer(cfg.embedding_model)

    def embed(texts: list[str]) -> torch.Tensor:
        return model.encode(texts, convert_to_tensor=True, normalize_embeddings=True,
                            batch_size=256, show_progress_bar=False)

    top = {}
    for field in ("question", "skeleton"):
        scores = embed([r[field] for r in dev]) @ embed([r[field] for r in train]).T
        top[field] = torch.topk(scores, max(args.k)).indices.tolist()

    def per_question(field: str, k: int, metric) -> list[float]:
        return [mean([metric(d, train[i]) for i in idx[:k]]) for d, idx in zip(dev, top[field])]

    report = {"embedding_model": cfg.embedding_model, "n_dev": len(dev), "n_train": len(train), "by_k": {}}
    print(f"\n=== Skeleton vs question retrieval  ({len(dev)} dev x {len(train)} train, {cfg.embedding_model})")
    print(f"{'k':>2}  {'metric':<15} {'question':>9} {'skeleton':>9} {'delta':>7} {'wilcoxon p':>11}")
    for k in args.k:
        row = {}
        for name, metric in METRICS.items():
            q, s = per_question("question", k, metric), per_question("skeleton", k, metric)
            p = wilcoxon(s, q).pvalue if any(a != b for a, b in zip(s, q)) else 1.0
            row[name] = {"question": mean(q), "skeleton": mean(s), "p": p}
            print(f"{k:>2}  {name:<15} {mean(q):>9.3f} {mean(s):>9.3f} {mean(s) - mean(q):>+7.3f} {p:>11.2g}")
        row["overlap"] = mean([len(set(a[:k]) & set(b[:k])) / k for a, b in zip(top["question"], top["skeleton"])])
        print(f"{k:>2}  {'same examples':<15} {row['overlap']:>9.1%}  (share of top-k picked by both)")
        report["by_k"][k] = row

    k = 3 if 3 in args.k else args.k[0]
    q_all = per_question("question", k, METRICS["struct_jaccard"])
    s_all = per_question("skeleton", k, METRICS["struct_jaccard"])
    report["struct_jaccard_by_hardness"] = {}
    print(f"\nstruct_jaccard by dev hardness (k={k}):")
    for level in ("easy", "medium", "hard", "extra"):
        sub = [i for i, d in enumerate(dev) if d["hardness"] == level]
        q, s = mean([q_all[i] for i in sub]), mean([s_all[i] for i in sub])
        report["struct_jaccard_by_hardness"][level] = {"n": len(sub), "question": q, "skeleton": s}
        print(f"  {level:<7} n={len(sub):<4} question={q:.3f} skeleton={s:.3f} delta={s - q:+.3f}")

    eq, es = embed([q for q, _ in FIG1]), embed([s for _, s in FIG1])
    sim_q, sim_s = eq @ eq.T, es @ es.T
    report["fig1"] = {"question": {"q1_q2": sim_q[0, 1].item(), "q1_q3": sim_q[0, 2].item()},
                      "skeleton": {"q1_q2": sim_s[0, 1].item(), "q1_q3": sim_s[0, 2].item()}}
    print("\nFig. 1 (Q1~Q2 same intent ORDER BY, Q1~Q3 same topic): want Q1~Q2 > Q1~Q3 for skeleton")
    print(f"  question: Q1~Q2={sim_q[0, 1]:.4f}  Q1~Q3={sim_q[0, 2]:.4f}")
    print(f"  skeleton: Q1~Q2={sim_s[0, 1]:.4f}  Q1~Q3={sim_s[0, 2]:.4f}")

    differ = [i for i in range(len(dev)) if top["question"][i][0] != top["skeleton"][i][0]]
    print(f"\nTop-1 differs for {len(differ)}/{len(dev)} dev questions. Random examples:")
    for i in random.Random(42).sample(differ, min(5, len(differ))):
        d = dev[i]
        print(f"\n  DEV  [{d['hardness']}] {d['question']}\n       {d['query']}")
        for field in ("question", "skeleton"):
            t = train[top[field][i][0]]
            print(f"  {field.upper():<8} J={jaccard(d['features'], t['features']):.2f} {t['question']}\n       {t['query']}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
