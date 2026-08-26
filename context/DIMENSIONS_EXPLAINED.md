# Dimensions — Explained Simply

> Companion to `EVALUATION_ANALYSIS_GUIDE.md` Bagian 3 and to `context/METRICS_EXPLAINED.md` — one level up from that file. `METRICS_EXPLAINED.md` explains what a single metric's *number* means (SLA, Token Consumption, TEP, QVT). This file explains what a whole **dimension** — a specific *combination* of those metrics, assembled to answer one research question — tells you that no single metric alone would. Currently covers: **Dimension 1, Dimension 2, Dimension 3**. The rest (Dimensions 4–6) will be added here as we get to each one.
>
> Every dimension in this project is 100% custom orchestration code (`src/dimensions/`) — it doesn't touch SPIDER's official tooling directly, it just combines numbers that `src/metrics/` already produced (and those modules' own 🔧/✍️ attribution is explained in `METRICS_EXPLAINED.md`, not repeated here).

## Dimension 1 (Efficiency)

> Code: `src/dimensions/dim1_efficiency.py`, `run_dimension_1()`. Alur & urutan output: guide Bagian 3, "Dimensi 1". Metrics used: Token Consumption, EX, TEP — all explained individually in `context/METRICS_EXPLAINED.md`.

### The question this dimension actually answers

This is **RM1** from the proposal: *does GraphRAG cut token cost without cutting performance?* Notice that's really two questions glued together with an "and" — "is it cheaper" and "did it stay accurate" — and neither half alone answers the real question.

- Token Consumption alone can only tell you "GraphRAG used fewer tokens." It says nothing about whether the answers were still any good — a system that always outputs `SELECT 1` would look *extremely* token-efficient and be completely useless.
- EX alone can only tell you "GraphRAG got more/fewer answers right." It says nothing about what that accuracy cost — a system could be more accurate simply by stuffing the *entire* database schema into every prompt, which would be a boring, expensive way to win.

Dimension 1 exists specifically to stop either of those from being read in isolation. It's Token Consumption and EX, always shown *side by side*, plus a third number (TEP) whose entire job is to state how those two moved *together*.

### Why the display order is fixed, not a style choice

The guide requires the output in exactly this order — Token Consumption table, then EX table, then TEP — and that's not arbitrary formatting. TEP is a single number that compresses "cost went down AND/OR accuracy changed" into one verdict. If you show that verdict *first*, a reader has no way to sanity-check it, or notice something surprising underneath it (like the per-difficulty catch below). Showing the raw ingredients first means the TEP number, when it finally appears, is something the reader can already half-predict and verify — not something they have to take on faith.

### Why the per-difficulty breakdown isn't optional decoration

This is the part that's easy to skip and shouldn't be. Here's a scenario that shows why:

Imagine the **overall** numbers look great: Baseline averages 2000 tokens per query and 60% EX; GraphRAG averages 1400 tokens and 68% EX. Plugged into TEP, that's the guide's own worked example (see `METRICS_EXPLAINED.md`'s TEP section) — `TEP = -0.444`, the "ideal" case. Case closed, right?

Now imagine what's actually happening underneath, split by difficulty:

| Difficulty | Baseline EX | GraphRAG EX | Baseline T | GraphRAG T |
|---|---|---|---|---|
| Easy | 90% | 95% | 900 | 600 |
| Hard | 40% | **35%** | 4500 | 2200 |

On **easy** queries, GraphRAG is both cheaper and a bit more accurate — genuinely good. But on **hard** queries — the ones that matter most, arguably — GraphRAG actually got *worse* (40% → 35%). It just also happened to save a huge number of tokens there, because pruning has more to cut on a complex query. Averaged across the whole dataset, the token savings on hard queries are so large that they drag the *overall* TEP number into looking uniformly great, quietly burying the fact that the pruning strategy is hurting exactly the queries where getting it wrong matters most.

Nothing about the overall Token Consumption/EX/TEP numbers alone would ever surface this — you'd have to already suspect it and go looking. That's the whole reason the guide insists on the per-difficulty rows: they're what would actually catch this pattern, if it's really happening.

### Reading the TEP number itself

`run_dimension_1()` doesn't reinvent TEP's meaning — see `METRICS_EXPLAINED.md`'s TEP section for the full elasticity explanation and the sign-reading logic (`TEP < 0` = ideal, `≈ 0` = neutral, `> 0` = trade-off). What's specific to *this dimension* is that the TEP shown here is always computed from the **overall** aggregates only (not per-difficulty) — it's the single combined verdict that sits at the bottom of the report, after the reader has already seen the breakdown that could complicate it.

### Implementation note

`run_dimension_1()` computes nothing new — every actual number it reports comes from a `src/metrics/` module that was already built and tested independently (`compute_token_consumption()`/`aggregate_token_consumption()`, `esm_ex_cm.aggregate_ex()`, `compute_tep()`). Its own job is purely mechanical: group each condition's raw-logs entries by difficulty (`easy`/`medium`/`hard`/`extra`), run those three metric functions over the whole set and over each difficulty bucket, and assemble everything into the guide's mandated order. A difficulty level with zero matching queries in the sample reports as `N/A`/`None`, not `0` — a bucket with no data and a bucket that scored 0% are not the same thing, and collapsing them together would misreport an absence of data as a failure.

### Where it fits in the bigger picture

Dimension 1 is the entry point of the six-dimension analysis — the guide's own execution order (Bagian 3) reads efficiency (this one) → structure (Dimension 2) → components (Dimension 3) → robustness (Dimension 4) → bottleneck (Dimension 5, conditional) → ablation (Dimension 6, prerequisite for the others' final `k`). If Dimension 1 shows GraphRAG trading away meaningful accuracy for token savings, that's the signal to go look at *why* in Dimension 2 (did the SQL structure degrade?) and Dimension 5 (was it retrieval or generation that broke?).

---

## Dimension 2 (SQL Structure)

> Code: `src/dimensions/dim2_structure.py`, `run_dimension_2()`. Alur & tabel klasifikasi: guide Bagian 3, "Dimensi 2". Metrics used: ESM, EX — both explained individually in `context/METRICS_EXPLAINED.md`.

### The question this dimension actually answers

This is **RM2**'s first angle: *when GraphRAG prunes the context down, what actually happens to the correctness of the SQL it writes?* Dimension 1 already told you *whether* accuracy moved — Dimension 2 is about reading ESM and EX **together** to figure out *what kind* of accuracy problem (or win) you're looking at, because ESM and EX can fail independently of each other, in ways that mean very different things.

Quick refresher (full detail in `METRICS_EXPLAINED.md`): ESM checks whether the SQL's *structure* matches the gold answer — same clauses, same shape — and ignores literal values entirely. EX checks whether *running* the SQL produces the right answer, regardless of whether the structure looks the same. A model can nail one and miss the other.

### Why reading them "as a pair" is the whole point

If you only looked at EX, a rise could mean GraphRAG got smarter — or it could hide the fact that the SQL now looks nothing like a textbook answer, which might matter if the thesis also cares about interpretability. If you only looked at ESM, a rise looks like an unambiguous win — but ESM doesn't check literal values, so a query that's *structurally* perfect can still be *functionally* wrong (wrong number in a `WHERE`, wrong string in a filter) and ESM would never catch it. Reading them together is what turns "a number went up" into an actual diagnosis.

### The four combinations, in plain terms

Think of it as a 2×2 grid: did EX go up or down (vs. Baseline), and did ESM go up or down (vs. Baseline)? Four cells, four different stories:

- **EX up, ESM down** — GraphRAG is writing SQL that *looks* less like the textbook answer, but *works* better anyway. Not a red flag — it usually means the model found a different, equally valid way to write the query (a subquery instead of a join, say), and the leaner context didn't hurt the logic at all.
- **ESM up, EX down** — the opposite, and more worrying: the SQL's *shape* is a closer match to gold, but it stopped *running* correctly. This is the signature of a model that copies structure well but is dropping small details — usually literal values (the wrong number, the wrong string) that ESM doesn't even look at.
- **EX up, ESM up** — both got better. This is the best-case outcome: the leaner, pruned context wasn't just "not harmful," it was actually helping the model be more precise.
- **EX down, ESM down** — both got worse. This is a comprehensive failure signal, not a subtle trade-off, and the guide's response to it is specific: **stop analyzing here and go run Dimension 5** (Bottleneck Analysis) to find out whether the failure started at retrieval (wrong schema handed to the model) or generation (right schema, model still got it wrong).

### A wording gap in the guide, and how it was resolved

Worth flagging honestly: the guide's own table writes that last row as "EX rendah, ESM rendah" ("EX low, ESM low") rather than "EX turun, ESM turun" ("EX down, ESM down") like the other three rows use — a wording shift with no number attached to say what counts as "low," and no statement of low *compared to what*. Three of the four rows are clearly about *direction of change* (GraphRAG vs. Baseline); this one reads, on its face, like it might be about an *absolute* value instead.

The researcher and I resolved this by treating all four rows the same way — directional, GraphRAG vs. Baseline — which is also the reading that cleanly completes the four possible combinations of "up" and "down" (the other three rows are up/down, down/up, and up/up; this one fills in the missing down/down). The alternative — inventing a new absolute percentage threshold for "low" — isn't supported by anything else in the guide and would risk overlapping confusingly with the other three rows. Full reasoning is logged in `context/IMPLEMENTATION_DECISIONS.md` poin 12.

### Where the numbers come from, and one honest edge case

The classification always runs on the **overall** aggregate — not per difficulty level — mirroring how Dimension 1's TEP is also computed only from overall numbers. Per-difficulty ESM/EX are still reported in the table (same reason as Dimension 1: an aggregate can hide a difficulty-specific story), just not separately classified into one of the four boxes.

One edge case worth knowing about: if EX and ESM's deltas come out to *exactly* zero for either metric — which practically only happens with small or synthetic samples, not realistic full-dataset runs — none of the four official combinations technically fit. Rather than force it into one of the four boxes anyway (which the guide never says how to do), that case is reported plainly as "not classified," with the raw deltas shown, so it's visibly a genuine edge case rather than a silently wrong label.

### Where it fits in the bigger picture

Dimension 2 is the natural follow-up to Dimension 1 — if the token/accuracy trade-off in Dimension 1 looked concerning, Dimension 2 tells you whether that's a structural problem, a literal-value problem, or nothing to worry about. And its worst-case outcome (both down) is the direct trigger for Dimension 5 — the two dimensions are meant to be read as a pipeline, not independently.

---

## Dimension 3 (SQL Components)

> Code: `src/dimensions/dim3_component.py`, `run_dimension_3()`. Alur: guide Bagian 3, "Dimensi 3". Metric used: CM (Component Match) — explained individually in `context/METRICS_EXPLAINED.md`.

### The question this dimension actually answers

Dimension 2 tells you *whether* GraphRAG's SQL got structurally worse, in one combined verdict. Dimension 3 answers the natural next question: **worse at *what*, specifically?** A SQL query has several moving parts — `SELECT`, `WHERE`, `JOIN`, `GROUP BY`, and so on — and a model can be rock-solid at some of them while quietly falling apart at others. Averaging everything into one ESM number hides exactly which part is the weak link. Dimension 3 breaks it back apart, clause by clause, so the weak link is visible.

### Why this isn't just Dimension 2 again

Think of ESM (Dimension 2) as a pass/fail exam grade, and CM (this dimension) as the per-question breakdown behind that grade. Two systems can both score "70% ESM" for completely different reasons — one might be acing `SELECT`/`FROM` and consistently fumbling `WHERE`; another might be the opposite. Dimension 2 alone can't tell those two failure modes apart. Dimension 3 exists specifically to make that distinction visible, because *where* the SQL breaks down points to a different root cause than *whether* it broke down.

### Reading the table: which clause dropped the most

The core output is a simple side-by-side table — Baseline vs. GraphRAG, one row per clause — with the *delta* (GraphRAG minus Baseline) called out for each. The one clause with the single biggest drop gets a plain-language note attached, but only for two specific clauses the guide names explicitly:

- **Biggest drop in `WHERE`** → read as: the pruned context is hurting the model's ability to filter correctly — it's losing track of the specific conditions a question is asking for.
- **Biggest drop in `FROM`/`JOIN`** → read as: the model is losing track of *how tables relate to each other* — a more structural kind of confusion, since it's about the shape of the query, not just a filter detail.

If the biggest drop lands on any of the other nine clauses (`SELECT`, `GROUP BY`, `HAVING`, `ORDER BY`, etc.), the implementation deliberately does **not** invent a parallel explanation for it — it just reports the number. The guide only hands down these two specific interpretations and explicitly warns against over-generalizing beyond them, so making up a plausible-sounding story for, say, a `HAVING` drop would be exactly the kind of guessing the guide is trying to prevent.

### `union_all`: a column that's honestly blank, not a column that failed

One clause, `union_all`, never gets a real score at all — it always shows `N/A`. This isn't a bug or a gap in the data; it's an honest limitation being reported honestly. SPIDER's official SQL parser has no way to tell `UNION` apart from `UNION ALL` in the text it reads — both parse into the exact same internal shape — so there is no way, ever, for any query, to compute whether `union_all` "matched." Showing `N/A` says "we genuinely can't check this one," which is a very different and much more honest statement than showing `0%`, which would say "we checked, and it consistently failed."

That distinction turned out to matter in practice, not just in theory. While building this dimension, a real bug was found and fixed: when a query failed so badly it couldn't be evaluated *at all* (a total crash, unrelated to `UNION ALL` specifically), the fallback code was accidentally writing `0` for `union_all` instead of leaving it blank. Since `union_all`'s score is only ever averaged from non-blank entries, every one of those crashed queries — regardless of why they crashed — was quietly dragging `union_all`'s reported number down. In effect, `union_all`'s score had accidentally become a measurement of "how many queries failed for unrelated reasons," dressed up as if it meant something about `UNION ALL` correctness. That's now fixed (`context/IMPLEMENTATION_DECISIONS.md` poin 13) — a crashed query now correctly still shows `union_all` as unmeasurable, not as a fabricated failure.

### Why there's no per-difficulty breakdown here

Unlike Dimensions 1 and 2, this one doesn't split its numbers by Easy/Medium/Hard/Extra — that's a deliberate reading of the guide, which asks for the per-clause breakdown "for B and G" without the "(agregat + per difficulty)" qualifier it explicitly attaches to the other dimensions. The per-clause split *is* the granularity this dimension is for; stacking a difficulty split on top wasn't something the guide asked for here.

### Where it fits in the bigger picture

Dimension 3 hands its finding — which specific clause degraded the most, and by how much — forward as diagnostic input to **Dimension 5** (Bottleneck Analysis): a `WHERE`-heavy failure pattern points toward a different root cause (detail loss during generation) than a `FROM`/`JOIN`-heavy one (relational confusion, possibly a retrieval problem). It's the piece that turns "something got worse" (Dimension 2) into "here's specifically what got worse," which is what makes the bottleneck diagnosis in Dimension 5 possible to actually act on.
