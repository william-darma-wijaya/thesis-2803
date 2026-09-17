# Metrics — Explained Simply

> Companion to `EVALUATION_ANALYSIS_GUIDE.md` — that file has the formulas, algorithms, and thresholds. This file explains *what the numbers actually mean*, one metric at a time, in plain terms. Currently covers: **SLA, Token Consumption, TEP, QVT**. The rest (ESM, EX, CM) will be added here later as we get to each one.
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

---

## Token Consumption (T)

> Code: `src/metrics/token_consumption.py`. Formula: guide Section 1.5.
>
> ✍️ **100% custom to this thesis, no 🔧 official SPIDER piece involved** — SPIDER's benchmark doesn't measure inference cost at all, only SQL correctness. Unlike SLA (which at least reuses SPIDER's parser), this metric borrows nothing from `external/spider_eval/`.

### What it checks

This one isn't about being *right* — it's about being *expensive*. Every question sent to the LLM costs something to process: reading the prompt costs tokens, and writing the answer costs tokens too. Token Consumption adds those up into one number per query, so GraphRAG's "smaller, pruned schema" claim can actually be checked against a number instead of just eyeballed.

### The two halves

- **T_in** — everything fed *into* the model for that question: the schema (as `CREATE TABLE` DDL), the few-shot examples (if any), and the natural-language question itself. This is the "reading" cost.
- **T_out** — everything the model *writes back*: the raw generated SQL, counted **before** any cleanup happens (before alias-stripping, `CAST` removal, etc. — see `generation.py`'s SQL cleaner). Counting the *cleaned* SQL instead would be cheating the number down — the model already did the work of producing the messy version, so that's the real cost paid.

Both counts use the actual **Qwen2.5-Coder-7B-Instruct tokenizer** — the same one doing the real inference — not a generic stand-in like `tiktoken`. Different tokenizers slice text into different-sized pieces, so a "token" from the wrong tokenizer wouldn't correspond to a real unit of cost for this model.

### The formula

```
T = T_in + μ · T_out          (μ = 3, fixed — from proposal subbab 3.8.2.5)
```

Output tokens count **3× heavier** than input tokens. Why the imbalance: generating text is a slow, one-token-at-a-time process (the model has to run a full forward pass for every single output token, in sequence). Reading the prompt, by contrast, happens largely in parallel in one pass (prefill). So a query that makes the model *write more* is punished harder in this formula than one that just hands it a longer schema to *read* — which matches how these costs actually behave in real inference.

### Where the numbers actually come from (implementation note)

`compute_token_consumption()` doesn't tokenize any text itself — it's pure arithmetic, `T_in + μ·T_out`, over integers that already exist. The real counting happens once, at the moment of generation: `generate_sql_with_token_count()` (in `src/generation/generation.py`) uses the LLM's already-loaded tokenizer to count `T_out` as the model generates, and the caller counts `T_in` the same way right next to it. Those integers get written straight into `data/raw_logs/*.json` (`token_input`/`token_output`) and `outputs/tables/ablation_results.csv`. So by the time this metric module sees a query, the tokenizing work is already done — it just adds the two numbers together with the right weight.

### Reporting: mean *and* median

Both are required, not just one. Token counts per query tend to be **skewed** — most queries are short, but a handful of gnarly multi-join questions can be much longer, and those drag the mean up. The median shows what a "typical" query costs; the mean shows the average including the expensive outliers. Reporting both (and broken down per difficulty level: Easy/Medium/Hard/Extra) gives a fuller picture than either alone.

### Where it fits in the bigger picture

Token Consumption is the raw ingredient for two things: **Dimension 1 (Efficiency)**, where GraphRAG's `T` is compared straight against Baseline's `T` to see if the pruning actually saves anything, and **TEP** (below), which asks a sharper question than "is it cheaper" — it asks "is it cheaper *without giving up accuracy*."

---

## TEP (Token Elasticity of Performance)

> Code: `src/metrics/tep.py`. Formula: guide Section 1.5, sub-bagian "TEP". Interpretation table: guide Section 3, "Dimensi 1".
>
> ✍️ **100% custom to this thesis** — this is the novelty metric of the whole research, adapted from an economics concept (see below). No SPIDER involvement at all; it's computed entirely from numbers Token Consumption and EX already produced.

### The idea, borrowed from economics

Economists have a concept called *price elasticity of demand*: if a product's price goes up by 10% and people buy 5% less of it, the "elasticity" is `-5%/10% = -0.5`. It's a single number that answers "how sensitive is one thing to a change in another thing?" — not just "did it change," but "how efficiently did it change relative to the thing that moved it."

TEP asks the same style of question about this research: **if GraphRAG changes token cost by some percentage, how much does accuracy (EX) change in response, proportionally?**

```
TEP_G = (ΔEX_G / EX_B) / (ΔT_G / T_B)

ΔEX_G = EX_G − EX_B     (accuracy change, GraphRAG vs Baseline)
ΔT_G  = T_G − T_B       (token cost change, GraphRAG vs Baseline)
```

Both halves are *percentage* changes (relative to the Baseline value), not raw differences — that's what makes this an elasticity instead of a plain subtraction. It's also why EX_B and EX_G **must** be in the same scale (both percent, or both 0–1) — mixing scales would silently wreck the ratio.

### Reading the sign — the whole point of the metric

The sign of TEP tells you whether accuracy and token cost moved **together** or **apart**:

- **TEP < 0 — moved in opposite directions.** The hoped-for outcome: GraphRAG's token cost went down *while* accuracy went up (or, symmetrically, cost went up while accuracy went up even more — either way, you're not paying for your gain). *"Skenario ideal: GraphRAG meningkatkan EX dan menurunkan Token Consumption dibanding baseline."*
- **TEP ≈ 0 — barely moved either way.** Within a small band (±0.05, agreed with the researcher on 2026-08-15 — the guide itself never pins an exact number here, unlike its other thresholds). Token savings happened without meaningfully hurting or helping accuracy. *"Efisiensi token tercapai tanpa trade-off penurunan EX yang signifikan."*
- **TEP > 0 — moved together.** Usually the unwelcome case here: token cost went down *and* accuracy went down with it — the pruning cost you something. *"Ada trade-off: penurunan token disertai penurunan EX."*

### Worked example (straight from the guide)

`EX_B = 60%, EX_G = 68%, T_B = 2000, T_G = 1400`

```
ΔEX_G = 68 − 60 = 8
ΔT_G  = 1400 − 2000 = −600
TEP_G = (8/60) / (−600/2000) = 0.1333 / (−0.3) = −0.444
```

TEP < 0 → GraphRAG got *more* accurate *and* cheaper at the same time. This is the number the implementation was checked against directly (`compute_tep(60, 68, 2000, 1400)` reproduces `-0.444` exactly).

### Why division-by-zero is treated as an error, not a workaround

If `T_G` and `T_B` come out exactly equal (`ΔT_G = 0`), or `EX_B` / `T_B` is `0`, the formula divides by zero. The guide doesn't say what TEP should mean in that situation, so `compute_tep()` raises a clear error instead of quietly returning `inf`/`nan` — a query that hit that edge case is a signal something's off with the run, not a number to interpret.

### Where it fits in the bigger picture

TEP is the payoff metric of **Dimension 1 (Efficiency)** — it's computed *last*, only after Token Consumption and EX have both already been reported side by side for Baseline and GraphRAG (guide's required output order: token table → EX table → TEP). It's the number that turns "GraphRAG uses fewer tokens" and "GraphRAG has different accuracy" into one combined verdict on whether the trade was worth it.

---

## QVT (Query Variance Testing)

> Code: `src/metrics/qvt.py`. Formula: guide Section 1.6.
>
> ✍️ **100% custom to this thesis** — SPIDER's benchmark evaluates one fixed question per gold SQL, it has no concept of "the same question asked differently." QVT and the whole idea of paraphrase robustness testing is specific to this thesis's proposal.
>
> ⚠️ **Implemented but not yet usable end-to-end** — see the "What's still missing" note near the bottom before assuming this metric can be run today.

### What it checks

Every other metric so far asks "did the system get *this exact question* right?" QVT asks a different question: **if you ask for the same thing in a different way, does the system keep giving you a working answer — or does it get lucky on one phrasing and fall apart on the next?**

Think of it like quizzing someone on the same fact five different ways. Getting it right once could just mean they memorized *that specific wording*. Getting it right across all five phrasings means they actually understood what was being asked.

### Where the "different phrasings" come from

SPIDER's dev set has exactly one NL question per gold SQL — there's no built-in paraphrase data. So QVT needs a **separate, hand-prepared dataset**: for a chosen gold SQL, someone writes (or generates) several reworded versions of the same question, runs each one through the pipeline, and checks whether the result is still correct. That data lives in `data/qvt_variations/` — see the "still missing" note below, this is the part that doesn't exist yet.

"Correct," for QVT specifically, is measured with **EX** (Execution Accuracy) — not ESM. This was a deliberate choice (`IMPLEMENTATION_DECISIONS.md` poin 10): QVT cares whether the *functional answer* stayed right, not whether the SQL kept the exact same shape — and paraphrasing a question is exactly the kind of thing that nudges an LLM toward a differently-structured-but-still-correct query (e.g. a subquery instead of a join). Grading that with a structure-sensitive metric like ESM would make a perfectly robust system look unstable just because its SQL style shifted.

### The two-layer score

**Layer 1 — per gold SQL.** For one gold SQL with `m` paraphrased variations, count how many of those `m` came back correct:

```
score_i = (number of variations that scored EX=1) / m
```

Example: a gold SQL has 5 paraphrases, 3 of them produce correct SQL → `score = 3/5 = 0.6`.

**Layer 2 — across the whole dataset.** Average the layer-1 scores across every gold SQL that has one:

```
QVT = mean(score_i)  — averaged only over gold SQLs that passed the filter below
```

### The filter — the step everyone forgets

Here's the part the guide calls out as the single most common implementation mistake: **if every single paraphrase of a gold SQL fails, that gold SQL doesn't get a score of 0 — it gets thrown out of the average entirely.**

Why: a gold SQL where *all* variations failed usually means the pipeline never had a shot at that question at all (wrong schema retrieved, question too hard, etc.) — that's a different kind of failure than "got some phrasings right, some wrong," and averaging it in as a flat 0 would conflate "somewhat inconsistent" with "completely unable," dragging the whole QVT score down for the wrong reason. So it's dropped, not zeroed.

Worked example straight from the guide: gold SQL `Q_2` has 4 paraphrases, all 4 fail → `Q_2` is **excluded** from the QVT average — not counted as `score_2 = 0`.

### Reading the practical effect of the filter

This makes the `M` in the formula (the count you divide by) **not the total number of gold SQLs in the dataset** — it's only the ones that passed the filter. So QVT technically answers "**among gold SQLs the system can answer at all, how consistently does it answer them across phrasings**" — it's not a measure of raw success rate (that's what EX/ESM on the un-paraphrased dev set already covers).

### Implementation note: how the code enforces this

`compute_qvt_per_query()` does layer 1: it counts correct variations for one gold SQL, and if that count is `0`, it returns the result marked `included=False` instead of `score=0.0` — that flag is what keeps it out of the average. `aggregate_qvt()` does layer 2: it takes the mean of only the `included=True` scores. If literally every gold SQL in the batch gets filtered out (every paraphrase of every question failed), it raises an error instead of silently reporting `QVT = 0` — that situation means something is badly wrong with the run, not that QVT is a real, meaningful zero.

### What's still missing (as of 2026-09-14)

Both pieces of code are now in place: `src/experiments/build_qvt_variations.py` builds `data/qvt_variations/` automatically from the 470 natural paraphrase pairs already present in SPIDER's `dev.json` (`IMPLEMENTATION_DECISIONS.md` poin 21 — no manual paraphrase writing needed), and `dim4_robustness.py`'s `run_dimension_4()` does the aggregation, the `ΔQVT = QVT_G − QVT_B` delta, the ±2% interpretation, and the QVT×ESM cross-tab (poin 23).

The only thing left is **data**: `data/qvt_variations/` stays empty until `data/raw_logs/*.json` exists, and that needs a real `python src/experiments/pipeline.py --baseline --sample 1.0` run on Kaggle (GPU). That's the same blocker every other dimension shares, not something specific to QVT any more. One requirement is specific to QVT though: the run **must** be full dev set (`--sample 1.0`), because `query_id` is positional — a subsampled run can't be mapped back to `dev.json` rows safely. `build_qvt_variations.py` checks this and refuses rather than producing a silently wrong pairing.

### Where it fits in the bigger picture

QVT is the headline metric of **Dimension 4 (Robustness & Consistency)**, always read alongside ESM (per gold SQL: is a query that's *inconsistent* across phrasings also one that scores badly on ESM in general?). The guide's interpretation bands: `ΔQVT ≥ +2%` means GraphRAG meaningfully improved consistency, `−2% to +2%` means robustness held steady, `< −2%` means GraphRAG's pruning made the system *less* stable across phrasings — a warning sign that the schema pruning is cutting too close to the bone.
