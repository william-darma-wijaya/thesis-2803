"""
Step 3 of IMPLEMENTATION_DECISIONS.md point 29: A/B of few-shot selection
("question" vs "skeleton") from two ablation.py runs on the same dev questions.

  EX / ESM : paired, exact McNemar test (binomial on the discordant questions)
  leak     : prediction uses a name that is NOT in its own schema context or
             question but IS in the few-shot examples -- the problem that
             started point 29. "unknown" = not in schema/question/examples at
             all (invented, or a retrieval miss), reported for contrast.
  value    : prediction copies a quoted literal of an example that is not in
             its own question.

Custom code for this thesis (not SPIDER). Reads
  outputs/predictions/ablation_{mode}_prompts_k{k}.jsonl           (question)
  outputs/predictions/ablation_{mode}_skeleton_prompts_k{k}.jsonl  (skeleton)

    python src/skeleton/compare_ab.py          # k=3, every mode with both runs
"""

import argparse
import json
import re
from pathlib import Path

from scipy.stats import binomtest

SQL_WORDS = frozenset("""
    select from where group by order having limit offset distinct as on join inner left right
    outer cross natural and or not in like between is null exists asc desc union intersect
    except all count sum avg min max case when then else end cast
""".split())
_STRING = re.compile(r"'[^']*'|\"[^\"]*\"")
_WORD = re.compile(r"[A-Za-z_]\w*")
_ALIAS = re.compile(r"t\d+")


def words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text)}


def literals(text: str) -> set[str]:
    return {s[1:-1].strip().lower() for s in _STRING.findall(text)} - {""}


def leak(row: dict) -> tuple[set[str], set[str], set[str]]:
    """(unknown names, names leaked from examples, values leaked from examples) of one prediction."""
    prompt = row["prompt"]
    schema = prompt.split("### Database Schema", 1)[-1].split("### Examples", 1)[0].split("### Question", 1)[0]
    examples = prompt.split("### Examples", 1)[1].split("### Question", 1)[0] if "### Examples" in prompt else ""
    names = {w for w in words(_STRING.sub(" ", row["pred_sql"])) if w not in SQL_WORDS and not _ALIAS.fullmatch(w)}
    unknown = names - words(schema) - words(row["question"])
    question = row["question"].lower()
    values = {v for v in literals(row["pred_sql"]) & literals(examples) if v not in question}
    return unknown, unknown & words(examples), values


def mean(xs) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def compare(mode: str, k: int, pred_dir: Path) -> dict | None:
    paths = {sel: pred_dir / f"ablation_{mode}{'' if sel == 'question' else '_skeleton'}_prompts_k{k}.jsonl"
             for sel in ("question", "skeleton")}
    if not all(p.exists() for p in paths.values()):
        print(f"\n[{mode}] skipped: needs {paths['question'].name} and {paths['skeleton'].name}")
        return None
    runs = {sel: {(r := json.loads(line))["question"]: r for line in p.read_text(encoding="utf-8").splitlines()}
            for sel, p in paths.items()}
    shared = [q for q in runs["question"] if q in runs["skeleton"]]
    if len(shared) != len(runs["question"]) or len(shared) != len(runs["skeleton"]):
        print(f"[{mode}] WARNING: runs are on different questions; comparing the {len(shared)} shared ones")
    q_rows = [runs["question"][q] for q in shared]
    s_rows = [runs["skeleton"][q] for q in shared]

    out = {"n": len(shared), "k": k}
    print(f"\n=== {mode}  (k={k}, {len(shared)} paired dev questions)")
    print(f"  {'':<34} {'question':>9} {'skeleton':>9} {'delta':>7}   McNemar")
    for metric in ("ex", "esm"):
        a, b = [float(r[metric]) for r in q_rows], [float(r[metric]) for r in s_rows]
        only_q = sum(x > y for x, y in zip(a, b))
        only_s = sum(y > x for x, y in zip(a, b))
        p = binomtest(only_s, only_q + only_s, 0.5).pvalue if only_q + only_s else 1.0
        out[metric] = {"question": 100 * mean(a), "skeleton": 100 * mean(b), "only_question": only_q,
                       "only_skeleton": only_s, "p": p}
        print(f"  {metric.upper():<34} {100 * mean(a):>8.2f}% {100 * mean(b):>8.2f}% {100 * (mean(b) - mean(a)):>+7.2f}"
              f"   p={p:.3g} (right only with question={only_q}, only with skeleton={only_s})")

    for name, field in (("T_in (prompt tokens)", "tokens_in"), ("T (consumption)", "token_consumption")):
        a, b = mean([r[field] for r in q_rows]), mean([r[field] for r in s_rows])
        out[field] = {"question": a, "skeleton": b}
        print(f"  {name:<34} {a:>9.1f} {b:>9.1f} {b - a:>+7.1f}")

    leaks = {"question": [leak(r) for r in q_rows], "skeleton": [leak(r) for r in s_rows]}
    rates = {
        "leak: name only in examples": lambda unknown, leaked, values: bool(leaked),
        "leak: value copied from examples": lambda unknown, leaked, values: bool(values),
        "unknown name (nowhere in prompt)": lambda unknown, leaked, values: bool(unknown - leaked),
    }
    for label, rate in rates.items():
        a, b = mean([rate(*x) for x in leaks["question"]]), mean([rate(*x) for x in leaks["skeleton"]])
        out[label] = {"question": 100 * a, "skeleton": 100 * b}
        print(f"  {label:<34} {100 * a:>8.2f}% {100 * b:>8.2f}% {100 * (b - a):>+7.2f}")

    print("  EX by difficulty:")
    out["ex_by_difficulty"] = {}
    for level in ("easy", "medium", "hard", "extra"):
        idx = [i for i, r in enumerate(q_rows) if r.get("difficulty") == level]
        a, b = mean([float(q_rows[i]["ex"]) for i in idx]), mean([float(s_rows[i]["ex"]) for i in idx])
        out["ex_by_difficulty"][level] = {"n": len(idx), "question": 100 * a, "skeleton": 100 * b}
        print(f"    {level:<7} n={len(idx):<4} question={100 * a:6.2f}%  skeleton={100 * b:6.2f}%  delta={100 * (b - a):+.2f}")

    examples = [(q, s, lq[1]) for q, s, lq, ls in zip(q_rows, s_rows, leaks["question"], leaks["skeleton"])
                if lq[1] and not ls[1]][:3]
    if examples:
        print("  Leaks with question selection that skeleton selection avoided:")
        for q, s, names in examples:
            print(f"    Q: {q['question']}\n       question-sel: {q['pred_sql']}   leaked={sorted(names)}"
                  f"\n       skeleton-sel: {s['pred_sql']}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="A/B of few-shot selection: question vs skeleton.")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--pred-dir", type=Path, default=Path("outputs/predictions"))
    ap.add_argument("--out", type=Path, default=Path("outputs/tables/skeleton_ab.json"))
    args = ap.parse_args()

    report = {mode: r for mode in ("graphrag", "baseline") if (r := compare(mode, args.k, args.pred_dir))}
    if report:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
