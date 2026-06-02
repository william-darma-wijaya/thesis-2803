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

<!-- Template for future entries:
## YYYY-MM-DD

### `filename.py`
- **Short title**
  - Why: ...
-->
