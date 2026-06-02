# Development Log — GraphRAG Text-to-SQL (Thesis)

Format: `[YYYY-MM-DD HH:MM]` | file(s) | what changed | **why**

---

## 2026-06-02

### `config.py`
- **`max_new_tokens`: 200 → 256**
  - Why: LLMs perform better at power-of-2 token boundaries; 200 was an arbitrary round number.

---

### `generation.py`
- **Device portability: `.to("cuda")` → `.to(model.device)`**
  - Why: Hard-coded `"cuda"` crashes on CPU-only machines and Kaggle sessions where device assignment is handled by `device_map="auto"`. Using `model.device` follows the model wherever it was placed.

---

### `pipeline.py`
- **Removed `prune_path_nodes` call in `run_single`**
  - Why: Experimental finding — pruning raises precision but reduces recall. For text-to-SQL, a missing table/column guarantees a wrong query, so recall is the more critical metric. Pruning is kept in `retrieval.py` for reference but is no longer invoked.
- **Removed dead imports**: `precompute_column_embeddings`, `prune_path_nodes`

---

### `sweep.py`
- **Removed `prune_path_nodes` call in `_run_combination`**
  - Why: Same reasoning as `pipeline.py` — keeps sweep metrics consistent with the actual pipeline behaviour (no pruning).

---

### `baseline.py`
- **Rewrote `evaluate_table_linking` — schema-aware gold element parsing**
  - Why: Old implementation used naive `sql.split()` token overlap, which mis-classified SQL keywords (`id`, `name`, `type`, etc.) as schema elements and inflated recall/precision. New implementation mirrors `evaluate_schema_linking` in `retrieval.py`: uses `_SQL_STOPWORDS` filter + validates each token against actual table/column names from the graph.
  - Impact: Baseline metrics are now directly comparable to GraphRAG metrics.

- **Fixed FK direction bug in `build_table_schema_context`**
  - Why: The graph is undirected, so iterating `graph.edges(node)` yields both outgoing and incoming FK edges. Without a guard, referenced (parent) tables were also annotated with `FOREIGN KEY` lines pointing back at the child — incorrect DDL. Fixed with a `this_cols` check so only edges originating from the current table's columns are emitted.

---

### `ablation.py`
- **Added `--mode {graphrag, baseline}` CLI flag**
  - Why: CLAUDE.md documented this flag in the usage examples but it was never implemented. Without it, the ablation study only covered GraphRAG; Baseline could not be ablated. The 2×4 design (2 modes × 4 k-values) requires both modes.
- **Refactored `_run_k`**: now accepts `mode` param and branches between full GraphRAG path and full Baseline path (lazy-imports from `baseline.py`)
- **Prediction filenames**: `ablation_predictions_k{k}.txt` → `ablation_{mode}_predictions_k{k}.txt` to avoid overwriting between modes
- **`AblationResult` dataclass**: added `mode: str` field
- **`_print_table`**: shows `mode` column; prints Spider eval commands grouped per mode×k
- **`_save_csv`**: added `mode` field
- **`main()`**: builds column-level graph + `SchemaIndex` cache for graphrag, or table-level graph + `TableSchemaIndex` cache for baseline
- **Removed `SchemaIndex` from imports** (no longer directly used after refactor — IDE flagged as unused)

---

---

## 2026-06-02 (session cont.)

### `sweep.py`
- **Replaced recall-weighted scoring with F6 (β=6) as the primary ranking criterion**
  - Why: arxiv 2501.17174 shows F6 has the peak correlation with Execution Accuracy (EX) among all F-beta variants (F6: 0.911 vs F5: 0.897 vs F7: 0.821 — peak at β=6). Using F6 directly optimises for the metric that best predicts actual SQL correctness, and is defensible in the thesis with a citation.
  - Formula: F6 = 37 × P × R / (36P + R) — weights recall 36× more than precision (β² = 36).
  - `meets_recall_target` (90% threshold) is kept as an informational display column but no longer drives the ranking.

### `ablation.py`
- **Token consumption tracking: count tokens inline + save full prompts to JSONL**
  - Why: Token consumption is a core thesis metric (see CLAUDE.md — Token Consumption, TEP). Counting inline with the already-loaded tokenizer is accurate and free. Saving full prompts to JSONL enables post-hoc TEP analysis without re-running inference.
  - Per-sample log line: `[12/1034] db=concert_singer tokens=847 | R=100.0% P=72.3%` — visible immediately during the run.
  - New output files per mode×k: `ablation_{mode}_prompts_k{k}.jsonl` (fields: i, db_id, question, tokens, prompt).
  - `AblationResult` now carries `avg_prompt_tokens` and `prompts_file`; both appear in the results table and `ablation_results.csv`.

---

---

## 2026-06-02 (session cont.)

### `ablation.py`
- **`_print_table`: now prints both `--etype match` AND `--etype exec` commands with real paths**
  - Why: Old implementation only printed `--etype exec` and used `<gold_path>` / `<db_dir>` / `<tables_json>` placeholders. EM (Exact Match) scores would be silently missed. Fixed to print both eval types and substitute actual paths from `PipelineConfig` so commands are copy-pasteable on Kaggle.
- **Added `_run_ablation_evals(results, cfg)` function**
  - Why: `pipeline.py` auto-runs Spider eval at the end; `ablation.py` did not. With 8 prediction files (2 modes × 4 k-values) × 2 eval types = 16 commands to run manually. New function runs them all automatically.
- **Added `--run-eval` CLI flag**
  - Why: Auto-eval is opt-in (adds wall-clock time and requires `evaluation.py` + `process_sql.py` to be present). Flag lets the user decide: omit for fast prediction-only runs, pass for a single command that produces all EM/EX numbers.
- **Added sweep-not-run warning in `main()`**
  - Why: If `ablation.py` is run before `sweep.py` and `config.py` is not updated, the pipeline uses default `top_k_tables=3`, `top_k_columns=5`. Results would be subtly wrong without any indication. Warning fires when values match the defaults.

---

## 2026-06-02 (session cont.)

### `config.py`
- **Added `token_output_weight: float = 1.0` (α)**
  - Why: Token Consumption formula is T = T_in + α×T_out. α lets you weight output tokens differently from input tokens (e.g. to mirror API pricing where output costs 3–5× more). Default 1.0 for local model (equal compute cost per token). Change this before TEP calculation if comparing to API-based systems.

### `ablation.py`
- **Corrected Token Consumption to T = T_in + α×T_out (was only T_in)**
  - Why: The thesis definition (Zhu et al., 2025) counts both input and output tokens, with α multiplier on output. Previous implementation only measured prompt length (T_in), missing the generated SQL token cost (T_out).
  - `generate_sql` is now called before the JSONL write so T_out is available in the same sample pass.
  - JSONL fields updated: `tokens` → `tokens_in`, `tokens_out`, `token_consumption` (T), `pred_sql` added.
- **`AblationResult` now carries `avg_output_tokens` and `avg_token_consumption`**
  - Why: Thesis needs T, T_in, and T_out separately for TEP and for the breakdown table.
- **`_save_csv` updated** — new columns: `avg_output_tokens`, `avg_token_consumption`
- **`_print_table` updated** — shows T_consume, T_in, T_out columns; prints α value in header

### `CLAUDE.md`
- **Corrected Token Consumption and TEP metric descriptions**
  - TEP formula: TEP_G = (ΔEX_G/EX_B) / (ΔT_G/T_B) — comparing GraphRAG vs Baseline (not k vs k)
  - Token Consumption: T = T_in + α×T_out
  - JSONL fields updated to match new schema

### `generation.py` + `ablation.py` — T_out now counts real emitted tokens
- **Added `generate_sql_with_token_count()` returning `(cleaned_sql, n_generated_tokens)`**
  - Why: T_out was being measured as `len(tokenizer.encode(pred_sql))` — i.e. re-encoding the CLEANED SQL. But cleaning strips ` ```sql ` fences, explanations, aliases, and collapses whitespace, so it undercounts the tokens the model actually emitted (the true generation cost, esp. when the model hits max_new_tokens or rambles). The new function returns the raw count of `generated_tokens` from `model.generate`, which is the correct T_out per the thesis definition ("teks yang dikeluarkan oleh LLM").
  - `generate_sql()` kept as a thin wrapper — `pipeline.py` and `baseline.py` are unaffected.
- **Fixed double-append bug in `ablation._run_k`**
  - Why: `recalls.append(r)` / `precisions.append(p)` ran BEFORE generation. If `generate_sql` threw (CUDA OOM on T4 is common), the `except` block appended `0.0` again — that sample contributed two entries to recall/precision but one to the token lists, desyncing the lists and skewing averages. Now all five metric lists are appended together after generation succeeds; the except branch appends exactly one `0` to each.

<!-- Template for future entries:
## YYYY-MM-DD

### `filename.py`
- **Short title**
  - Why: ...
-->
