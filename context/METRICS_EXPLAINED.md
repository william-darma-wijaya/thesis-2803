# Metrics — Explained Simply

> Companion to `EVALUATION_ANALYSIS_GUIDE.md` — that file has the formulas, algorithms, and thresholds. This file explains *what the numbers actually mean*, one metric at a time, in plain terms. Currently covers: **SLA**. The rest (ESM, EX, CM, Token Consumption, TEP, QVT) will be added here later as we get to each one.
>
> Throughout: 🔧 marks something that's the **actual official SPIDER benchmark source code** (from `external/spider_eval/`, downloaded from the official `taoyds/spider` GitHub repo) — not written for this thesis. ✍️ marks code **written specifically for this thesis** (usually wrapping or extending the 🔧 pieces). See `CLAUDE.md` → "Atribusi kode: SPIDER resmi vs custom" for the full rule.

## SLA (Schema Linking Accuracy)

> Code: `src/metrics/sla.py`. Formulas/algorithm: guide Section 1.1.
>
> ✍️ **SLA is not an official SPIDER metric at all** — SPIDER's own benchmark doesn't score schema retrieval, only final SQL correctness. SLA is specific to this thesis's proposal. The one 🔧 official piece it borrows is SPIDER's SQL parser (`process_sql.py`'s `Schema`/`get_sql()`) — reused so gold SQL is read the *exact* same way the official ESM/EX metrics read it, not a second, possibly-inconsistent parser.

### What SLA checks

Before the LLM ever writes a single line of SQL, the retrieval step has to go find the right `table.column`s to hand it. **SLA checks whether retrieval grabbed the correct pieces — before generation even starts.**

Think of it as a treasure hunt. There's a shopping list of exactly which columns are needed to answer the question correctly (the **gold schema** — figured out by reading the correct SQL answer). The retrieval pipeline goes and grabs its own pile of columns, guessing from the question (the **predicted schema**). SLA just compares the two piles.

### How it's calculated, step by step

**Step 1 — Build the correct checklist.** Read the gold SQL (the human-written correct answer) and figure out every `table.column` it actually uses. This is *not* done by scanning the question text for words that look like column names — that would get fooled by coincidences. Instead, the gold SQL is fed to SPIDER's own official parser 🔧 (`process_sql.py`'s `Schema` + `get_sql()` — literally the same parser the official ESM/EX metrics use), which turns it into a structured tree instead of a plain string. Then a custom walker ✍️, written for this thesis, goes through that whole tree (`SELECT`, `WHERE`, `JOIN`, `GROUP BY`, even columns hiding inside a sub-question) and pulls out every column genuinely being used — SPIDER's own script doesn't expose a ready-made tool for this, so this part had to be written from scratch. Result: the gold schema, a set like `{students.name, students.age}`.

*Even a plain `SELECT name FROM students` (no table name written anywhere) still ends up as `students.name` on the checklist* — the parser 🔧 already knows `students` is the only table around, so it fully qualifies the column itself. This matters because the predicted pile (Step 2) is always fully qualified too — if gold weren't, they'd never match.

**Step 2 — Grab the predicted pile.** This part's already done by the time SLA runs — it's just whatever columns the retrieval step (GraphRAG or Baseline) picked out and handed to the LLM for that question.

**Step 3 — Compare the two piles.** Overlap the two sets to get Precision, Recall, and F1 (see below).

**Step 4 — Do Step 3 twice.** Once comparing whole tables, once comparing exact columns (see "Two zoom levels" below).

**Step 5 — Repeat for every question, then average.** One question's score isn't the final grade — every question in the dataset gets its own Precision/Recall/F1, and the reported SLA is the average across all of them (see "Averaging" below).

### Precision — "of everything you grabbed, how much did you actually need?"

```
correct_grabs = gold_schema ∩ predicted_schema
precision = |correct_grabs| / |predicted_schema|
```

Example: gold needs `{name, age, student_id}`. Predicted grabbed `{name, age, gpa, student_id}`.

**Precision = 3/4 = 75%** — the robot grabbed `gpa` for no reason. Oops, that hurts precision a little.

### Recall — "of everything you needed, how much did you actually grab?"

```
recall = |correct_grabs| / |gold_schema|
```

Same example: gold needed 3 items, predicted got all 3.

**Recall = 3/3 = 100%** — the robot got everything it needed. Perfect recall, even with that one useless extra grab.

### Why recall matters way more than precision here

**Grabbing one extra useless card is messy but not fatal. Forgetting a card you needed is fatal** — the LLM can't write a correct answer with a column it was never shown. That's why the guide treats recall as the headline number for SLA, and precision as secondary.

### F1 — one number that balances both

```
F1 = 2 · (precision × recall) / (precision + recall)
```

F1 = 2·(0.75×1.0)/(0.75+1.0) = **0.857**. Handy for a quick glance, but SLA should always be reported with precision and recall shown separately too — F1 alone hides *which* of the two is the problem.

### Two zoom levels: table vs. column

- **Table level** — did retrieval at least go into the right *folder* (e.g. `students`)? Forgiving — doesn't care which exact column inside.
- **Column level** — did retrieval grab the exact right *card* inside the folder (`name`, not `gpa`)? Strict — the real test.

It's the exact same two piles both times — table level just chops each `table.column` down to `table` first (so `students.name` and `students.age` both become just `students`), then runs the identical Precision/Recall/F1 math. That's also why table level always looks more forgiving: found the right folder but grabbed the wrong cards inside it → column level punishes you, table level doesn't even notice.

Both are computed and reported separately — never mixed into one number.

### Averaging across the whole dataset

Every question gets its own precision/recall/F1. The final score is the **average of those per-question scores** (macro-average) — not "pool everything together first, then compute one big score" (micro-average). Otherwise a handful of column-heavy queries could quietly dominate the result.

### Where it fits in the bigger picture

SLA is diagnostic, not a final grade. It pairs with EX in **Dimension 5 (Bottleneck: Retrieval vs. Generation)**: high SLA F1 but low EX means retrieval found the right stuff and generation is where it broke. Low SLA F1 means retrieval itself is the bottleneck — the LLM never had a fair shot.
