# Dimensions — Explained Simply

> Companion to `EVALUATION_ANALYSIS_GUIDE.md` Bagian 3 and to `context/METRICS_EXPLAINED.md` — one level up from that file. `METRICS_EXPLAINED.md` explains what a single metric's *number* means (SLA, Token Consumption, TEP, QVT). This file explains what a whole **dimension** — a specific *combination* of those metrics, assembled to answer one research question — tells you that no single metric alone would. Currently covers: **Dimension 1**. The rest (Dimensions 2–6) will be added here as we get to each one.
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
