# Dimensions — Explained Simply

> Companion to `EVALUATION_ANALYSIS_GUIDE.md` Bagian 3 and to `context/METRICS_EXPLAINED.md` — one level up from that file. `METRICS_EXPLAINED.md` explains what a single metric's *number* means (SLA, Token Consumption, TEP, QVT). This file explains what a whole **dimension** — a specific *combination* of those metrics, assembled to answer one research question — tells you that no single metric alone would. Currently covers: **Dimension 1, Dimension 2, Dimension 3, Dimension 5, Dimension 6**. Dimension 4 will be added once it's implemented.
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

---

## Dimension 4 (Robustness: Consistency Under Paraphrase)

> Code: `src/dimensions/dim4_robustness.py`, `run_dimension_4()`. Alur & tabel interpretasi: guide Bagian 3, "Dimensi 4". Metrics used: QVT (consistency) and ESM (SQL structure) — both explained individually in `context/METRICS_EXPLAINED.md`.

### The question this dimension actually answers

Every other dimension asks some version of "how often is the system right?" This one asks something different: **is the system right for the same reasons every time?** Two people can ask for the exact same thing in different words — "How many singers do we have?" and "What is the total number of singers?" — and a system that genuinely understands the question should produce the same SQL for both. A system that gets one right and the other wrong isn't really *understanding* the question; it's reacting to the specific wording it happened to see. That's brittleness, and a plain accuracy number hides it completely — a system that gets exactly half of every paraphrase pair right can post the same accuracy as one that's perfectly consistent but slightly less capable.

The reason this matters specifically for *this* thesis: GraphRAG's whole premise is cutting the schema down to only what's relevant. A fair worry is that aggressive pruning makes the pipeline fragile — the smaller the context, the more a single differently-worded question could swing retrieval onto the wrong columns. Dimension 4 is the check on that worry.

### Where the paraphrases come from — a genuinely lucky finding

Measuring this normally means writing paraphrases yourself, which is slow and introduces your own bias about what "the same question, reworded" means. It turned out not to be necessary: SPIDER's own dev set already contains **470 SQL queries that each have two different human-written natural-language phrasings** — covering about 91% of the dev set — a side effect of how the benchmark was annotated. Those are exactly the `N_i1, N_i2` the QVT formula needs, written by actual humans rather than generated, and they cost nothing extra to run because the pipeline already answers every dev question anyway. (`IMPLEMENTATION_DECISIONS.md` poin 21.)

The catch worth stating plainly in the methodology: there are always **exactly two** phrasings per query, never three or more. So "consistent" here means "got both of them right" and "inconsistent" means "got exactly one right" — a binary distinction, not a fine-grained spectrum.

### The filter that's easy to get wrong

QVT deliberately **throws away** any query where *all* of the phrasings failed, rather than scoring it zero. This looks like cherry-picking at first glance, but it isn't — it's the difference between two very different failures. A query the system gets wrong every single time, no matter how you word it, is **consistently wrong**: that's an accuracy problem, and Dimensions 2, 3, and 5 already measure it thoroughly. Scoring it as "zero consistency" would be double-counting an accuracy failure as a robustness failure, and would drag the QVT number down for a reason that has nothing to do with robustness. QVT is only meaningful for queries the system *can* get right — the question is whether it does so reliably. The count of dropped queries is still reported next to the score, so it's never invisible.

### Reading the ±2% threshold

The verdict is the gap between the two conditions' QVT scores (GraphRAG minus Baseline), read against a fixed ±2 percentage point band:

- **≥ +2%** → GraphRAG is *meaningfully* more consistent.
- **between −2% and +2%** → consistency is effectively unchanged — the token savings didn't cost robustness. For this thesis, this is a perfectly good result, not a null one: the hypothesis being tested is that pruning doesn't *hurt*, so "no change" is a pass.
- **< −2%** → GraphRAG is *less* consistent, which reads as over-pruning: the context got small enough that wording started to matter more than it should.

The ±2% isn't arbitrary or tunable. It comes from the guide, and the reasoning is that published state-of-the-art methods differ from each other by only 3–4 percentage points, and the model here isn't fine-tuned — so it's noisier than those. Anything under 2 points is inside the noise floor and shouldn't be narrated as a real effect.

### Crossing QVT with ESM, and why that needs a second data source

The last step asks whether the unstable queries are *also* the structurally broken ones — are these one problem or two? If inconsistent queries are the same queries that fail ESM, then paraphrase instability is a symptom of a system that was already struggling on those queries. If they're mostly *different* queries, then instability is its own separate failure mode and deserves its own treatment.

This step needs data the QVT files don't have. Within QVT, "correct" is deliberately defined by **EX** (did the SQL actually return the right answer) and held fixed everywhere, so that one definition is never mixed with another mid-calculation (`IMPLEMENTATION_DECISIONS.md` poin 10). ESM is a different question entirely — did the SQL have the right *structure* — so it has to be read from the raw logs, its original source. That's why `run_dimension_4()` takes the raw logs as extra optional arguments: passing them runs the cross-tab, leaving them out prints an explicit "this step was skipped, and here's why" notice rather than quietly dropping a step the guide asks for (`IMPLEMENTATION_DECISIONS.md` poin 23).

Both axes use the same all-or-nothing shape — "did *every* phrasing succeed?" for QVT, "did *every* phrasing have correct structure?" for ESM — so the resulting 2×2 table compares like with like rather than a strict standard against a lenient one.

### Where it fits in the bigger picture

Dimensions 2 and 3 measure whether GraphRAG's SQL is *correct*. Dimension 4 measures whether that correctness is *dependable* — and those can diverge. A pipeline that improves accuracy while quietly becoming more wording-sensitive has traded a visible number for an invisible weakness, and Dimension 4 is the only place in this analysis where that trade would show up at all.

---

## Dimension 5 (Bottleneck: Retrieval vs. Generation)

> Code: `src/dimensions/dim5_bottleneck.py`, `run_dimension_5()`. Alur & tabel diagnostik: guide Bagian 3, "Dimensi 5". Metrics used: SLA (retrieval quality) and EX (generation quality) — both explained individually in `context/METRICS_EXPLAINED.md`.

### The question this dimension actually answers

Dimensions 2 and 3 tell you *that* something broke and *which part of the SQL* broke. Dimension 5 asks a different, earlier question: **did the pipeline break because it fetched the wrong information, or because it had the right information and still wrote the wrong SQL?** That's the difference between a retrieval problem and a generation problem — and they call for completely different fixes. If retrieval is the problem, you'd tune the schema-linking step. If generation is the problem, retrieval is already doing its job and the fix belongs somewhere in the LLM's prompting or the model itself.

### Why this dimension is shaped differently from the others

Dimensions 1–3 are all **comparisons** — Baseline vs. GraphRAG, side by side. Dimension 5 is a **diagnosis of one pipeline at a time**. You point it at a single condition's results (typically GraphRAG, since that's the system actually being investigated) and it tells you where *that* pipeline's weak point is. Nothing stops you from running it on Baseline too, for comparison, but the diagnosis itself doesn't require a Baseline number to make sense — a pipeline can be unhealthy in absolute terms, independent of how the other one is doing.

### Reading the 2×2 diagnostic grid

Two health checks feed into it: **SLA F1** (did retrieval find the right schema? threshold: 80%) and **EX** (did the final SQL actually work? threshold: 65%). Crossing them gives four possible stories:

- **Both healthy** → the pipeline is working end to end.
- **Retrieval healthy, EX unhealthy** → **generation is the bottleneck.** The model was handed the right schema and still couldn't write correct SQL — the problem is downstream of retrieval.
- **Retrieval unhealthy, EX unhealthy** → **retrieval is the bottleneck.** The model never had a fair shot — if it's working from a wrong or incomplete schema, blaming its SQL-writing ability would be missing the actual cause.
- **Retrieval unhealthy, EX healthy** → an unusual case — the guide's own read is that this is probably a run full of simple questions the model could still answer correctly even from partial schema, or it got lucky. Worth a second look if it comes up, rather than treated as a normal outcome.

### Why SLA gets computed at *both* zoom levels, with two separate diagnoses

This is the one place in the whole analysis where a single metric — SLA — gets *two* independent numbers (table-level and column-level, see `METRICS_EXPLAINED.md`), and both get run through the **full** diagnostic grid separately, rather than picking one to decide the "official" answer.

Why that matters: it's entirely possible for a pipeline to find the right *table* every time, while still missing most of the specific *columns* it actually needs from that table — table-level SLA would call that healthy, column-level SLA would call it a retrieval bottleneck. Those aren't two measurements of the same fact reported at different precisions — they're two genuinely different claims, and collapsing them into one number by picking a side would quietly throw away real information about the shape of the failure.

A concrete version of this actually surfaces in testing: a pipeline that always retrieves the correct table, but only ever grabs one of the three columns actually needed from it, scores a perfect 100% at the table level ("pipeline sehat") while scoring only 50% at the column level ("jarang terjadi, kemungkinan query sederhana") — two different diagnoses, from the same run, both correct at their own level of resolution. When this happens, the result plainly shows both diagnoses side by side rather than forcing a single verdict — the guide doesn't say which level should win when they disagree, so nothing here invents an answer it doesn't have.

### When SLA is unhealthy: recall matters more than precision

If SLA F1 falls under 80%, the diagnosis goes one level deeper, the same way it does in `METRICS_EXPLAINED.md`'s SLA section: **low recall is the more serious problem.** Missing a column you actually needed (low recall) means the model is working with an incomplete picture and structurally can't write the right SQL. Grabbing some extra unnecessary columns (low precision) is untidy but survivable — the model usually still has everything it needs, just with some noise mixed in.

### Where it fits in the bigger picture

The guide frames Dimension 5 as conditional — worth running specifically when Dimension 2 turns up its worst-case outcome (both ESM and EX down). That trigger is a judgment call for whoever is running the analysis, not something the code itself checks — `run_dimension_5()` is a self-contained diagnostic tool that can be pointed at any condition's results whenever the question "where is this actually breaking?" needs an answer, whether or not Dimension 2 flagged it first.

---

## Dimension 6 (Few-Shot Ablation)

> Code: `src/dimensions/dim6_ablation.py`, `run_dimension_6()`. Alur: guide Bagian 3, "Dimensi 6". Metric used: EX, at four different few-shot example counts.

### The question this dimension actually answers

Every dimension so far compares Baseline against GraphRAG. Dimension 6 steps outside that comparison entirely and asks a setup question that has to be answered *before* the other five dimensions can be run fairly: **how many worked examples should the model be shown before it answers a question?** The pipeline can hand the LLM 0, 1, 3, or 5 example question-and-SQL pairs as a reference before the real question — more examples generally cost more tokens, but don't necessarily buy proportionally more accuracy. Dimension 6 exists to pick one number and settle on it, so that Baseline and GraphRAG are later compared using the *same* number of examples rather than each getting whatever happened to look best for it individually.

### Why this has to happen before the others, in principle

If Baseline used 5 examples and GraphRAG used 1, and GraphRAG came out ahead, you'd have no way to know whether that's because of GraphRAG's actual retrieval strategy or just because it happened to get a different amount of help. Dimension 6 removes that confound by finding one `k` (the example count) that gets used everywhere else. In practice, this project built its metrics and dimension code before running this experiment — which is fine for building and testing the code, but the *real* `k_final` this dimension would settle on doesn't exist yet, because the underlying experiment (`ablation.py`) hasn't been run for real yet either (see `RESEARCHER_TODO.md`).

### Reading the pattern: a step-by-step rule, not a single "best" number

The guide is explicit about what *not* to do here: don't just look at the four EX numbers and pick whichever `k` happened to score highest. A `k` could score highest by a hair's width while costing far more tokens than the runner-up — that's not a meaningful win, just noise plus expense.

Instead, this walks through `k` in order — 0, then 1, then 3, then 5 — asking one question at each step: *did moving up to this next `k` buy a meaningful amount of extra accuracy?* As long as the answer keeps being "yes," it keeps climbing. The moment a step's gain gets small — or EX actually drops — it stops right there and settles on the `k` it had *before* that weak step. It never resumes climbing after that, even if, by coincidence, a much larger `k` further out happens to spike back up — a fluke like that isn't worth chasing, and settling early is the whole point of "diminishing returns" in the first place.

That single rule naturally produces all of the guide's named patterns as different outcomes of the same walk, rather than needing three separate special-case checks:

- **Big jump early, then flat** → the walk climbs through the first big step, then stops at the next weak one. This is "diminishing returns" — the model mostly needed just that first example.
- **Small steps the whole way, right from the start** → the very first step already fails to be a meaningful gain, so the walk never leaves `k=0`. This reads as "the system barely depends on example count at all" — and since more examples aren't buying anything, there's no reason to pay for them.
- **A step where EX actually drops** → the walk stops right before the drop. This is "context overload" — past some point, more examples are actively confusing the model rather than helping it.
- **Keeps climbing meaningfully all the way to `k=5`** → the walk never finds a weak step within the range tested. The guide doesn't name this case explicitly (its examples assume a plateau shows up somewhere in {0,1,3,5}), so this implementation labels it honestly as "still climbing" rather than forcing it into one of the three named patterns — the honest reading is that `k=5` is the best available answer *within what was tested*, not that the search is necessarily finished.

### What counts as "a meaningful gain," and where that number comes from

The guide never states a number for how big a step needs to be to count as "significant." Rather than invent an arbitrary one, this reuses **2 percentage points** — the same threshold the guide already fixes elsewhere, for how much `ΔQVT` has to move to count as a meaningful change in Dimension 4. Reusing an existing, guide-sanctioned number keeps the whole analysis internally consistent instead of introducing a second, unrelated notion of "significant" with no shared basis.

### Where it fits in the bigger picture

Dimension 6 is unusual among the six in being a **prerequisite**, not a downstream analysis — its output (`k_final`) is meant to feed back into how Dimensions 1–5 are run, not to be read alongside them as one more comparison. The guide's own execution-order note (Bagian 0.1) says as much: ideally this runs *first*, before any Baseline-vs-GraphRAG numbers are taken as final, even though — for this project specifically — the code for Dimensions 1–5 was written and tested before this experiment had real data to run on.
