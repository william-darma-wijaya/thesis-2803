"""
De-semanticization (Guo et al., 2023, arXiv:2304.13301), simplified: mask the
domain words of a question so only its intent skeleton is left.

    What are the names of the singers who are not French?
 -> What are the [MASK] of the [MASK] who are not [MASK]?

Custom code for this thesis (not from the paper, not from SPIDER). A token is
masked when it is
  1. a number (POS NUM or digits),
  2. a value: inside a spaCy entity and a NOUN/PROPN, or a nationality (NORP),
  3. a content word whose lemma is a word of a table/column name of the
     question's OWN database (word-by-word, no synonyms),
  4. inside quotes ('Fail', "Homeland Security"). Not in the paper: it stands
     in for the database-value matching we skip.
INTENT_WORDS are never masked by rules 1-3, even when a column is named after them.

The skeleton is only for RETRIEVING few-shot examples; the LLM still gets the
original question. Precomputed locally so Kaggle does not need spaCy:

    python src/skeleton/masking.py      ->  data/skeletons/{train,dev}.json
"""

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import spacy

MASK = "[MASK]"
CONTENT_POS = {"NOUN", "PROPN", "ADJ", "VERB", "NUM"}
ENTITY_POS = {"NOUN", "PROPN"}
# Words that decide the SQL operator. Spider has columns like Number_of_...,
# Total, Order_ID, so plain lemma matching would erase them from ~13% of
# questions. Checked against both the word and its lemma ("highest" -> "high").
# have/be: tables like Has_Pet, Is_male; "have ... but do not have" means EXCEPT.
INTENT_WORDS = frozenset("""
    number total count average avg sum maximum max minimum min
    most least highest lowest largest smallest order distinct different have be
""".split())
# A single quote only opens/closes a value when not glued to a word (students' names).
_QUOTED = re.compile(r"\"[^\"]+\"|(?<!\w)'[^']+'(?!\w)")


def load_nlp():
    nlp = spacy.load("en_core_web_sm")
    # spaCy splits "id" like the contraction "I'd" (-> "i", "d"), which breaks
    # the most common column word in Spider.
    nlp.tokenizer.rules = {k: v for k, v in nlp.tokenizer.rules.items() if k.lower() != "id"}
    return nlp


def schema_words(db: dict, nlp) -> set[str]:
    """Lemmas of every word in the database's table and column names (tables.json)."""
    names = db["table_names"] + [name for _, name in db["column_names"][1:]]  # [0] is "*"
    return {t.lemma_.lower() for doc in nlp.pipe(names) for t in doc}


def mask_question(doc, words: set[str]) -> str:
    """Skeleton of a spaCy-parsed question, given its database's schema_words()."""
    quoted = [(m.start() + 1, m.end() - 1) for m in _QUOTED.finditer(doc.text)]
    out = []
    for t in doc:
        lemma = t.lemma_.lower()
        masked = any(a <= t.idx < b for a, b in quoted) or (
            t.lower_ not in INTENT_WORDS and lemma not in INTENT_WORDS and (
                t.pos_ == "NUM" or t.is_digit
                or (t.ent_type_ and t.pos_ in ENTITY_POS)
                or t.ent_type_ == "NORP"
                or (t.pos_ in CONTENT_POS and lemma in words)
            )
        )
        out.append((MASK if masked else t.text) + t.whitespace_)
    return "".join(out)


def _self_check(nlp, dbs: dict) -> None:
    words = schema_words(dbs["concert_singer"], nlp)
    got = mask_question(nlp("What are the names of the singers who are not French?"), words)
    assert got == "What are the [MASK] of the [MASK] who are not [MASK]?", got
    # intent word survives even when a column is named after it
    got = mask_question(nlp("What is the total number of singers aged 30?"), words | {"number", "total"})
    assert got == "What is the total number of [MASK] [MASK] [MASK]?", got
    # "id" stays one token; quoted value masked whole; possessive quote is not a value
    got = mask_question(nlp("Show the id of the singers' song named 'Love Me Do'."), words)
    assert got == "Show the [MASK] of the [MASK]' [MASK] [MASK] '[MASK] [MASK] [MASK]'.", got


def main() -> None:
    ap = argparse.ArgumentParser(description="Mask Spider questions into skeletons (de-semanticization).")
    ap.add_argument("--spider-dir", type=Path, default=Path("data/spider_data"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/skeletons"))
    ap.add_argument("--show", type=int, default=15, help="random examples to print per split")
    args = ap.parse_args()

    nlp = load_nlp()
    dbs = {d["db_id"]: d for d in json.loads((args.spider_dir / "tables.json").read_text(encoding="utf-8"))}
    _self_check(nlp, dbs)
    words = {db_id: schema_words(db, nlp) for db_id, db in dbs.items()}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, fname in (("train", "train_spider.json"), ("dev", "dev.json")):
        data = json.loads((args.spider_dir / fname).read_text(encoding="utf-8"))
        rows, unmasked_nouns = [], Counter()
        for x, doc in zip(data, nlp.pipe(x["question"] for x in data)):
            skeleton = mask_question(doc, words[x["db_id"]])
            rows.append({"db_id": x["db_id"], "question": x["question"], "skeleton": skeleton, "query": x["query"]})
            unmasked_nouns.update(t.lemma_.lower() for t in doc if t.pos_ == "NOUN"
                                  and t.lemma_.lower() not in words[x["db_id"]] | INTENT_WORDS)

        out = args.out_dir / f"{split}.json"
        out.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")

        n_masks = [r["skeleton"].count(MASK) for r in rows]
        print(f"\n=== {split}: {len(rows)} questions -> {out}")
        print(f"  avg [MASK]/question: {sum(n_masks) / len(rows):.2f}   "
              f"no [MASK] at all: {sum(n == 0 for n in n_masks)} ({sum(n == 0 for n in n_masks) / len(rows):.1%})")
        print(f"  most common nouns left unmasked (not in own schema): {unmasked_nouns.most_common(15)}")
        for r in random.Random(42).sample(rows, min(args.show, len(rows))):
            print(f"  [{r['db_id']}] {r['question']}\n  {' ' * (len(r['db_id']) + 3)}{r['skeleton']}")


if __name__ == "__main__":
    main()
