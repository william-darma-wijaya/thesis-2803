# Pipeline Foldering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the flat-at-root text2sql pipeline files (`config.py`, `schema.py`, `retrieval.py`, `baseline.py`, `few_shot.py`, `generation.py`, `pipeline.py`, `sweep.py`, `ablation.py`, `evaluation.py`, `process_sql.py`, notebooks, `plot/`) into a tidy `src/{core,retrieval,generation,experiments}/`, `external/spider_eval/`, `notebooks/`, `outputs/{plots,predictions,logs,tables}/` layout mirroring `evaluation_pipeline/`'s structure, and fix every import/path reference so the code still runs.

**Architecture:** Pure reorganization, no logic changes. Files move via `git mv` to preserve history. Internal imports become absolute (`from src.core.config import ...`). The four CLI entry points (`pipeline.py`, `sweep.py`, `ablation.py`, `baseline.py`) get a 1-line `sys.path` bootstrap so `python src/experiments/pipeline.py` keeps working directly, exactly as documented in `CLAUDE.md`'s CLI Quick Reference and used from Kaggle notebooks.

**Tech Stack:** Python 3 (stdlib `pathlib`/`ast` for verification), git.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-07-pipeline-foldering-design.md` — follow it exactly; do not invent a `text2sql_pipeline/` container folder (explicitly rejected).
- Preserve git history: every file move is `git mv`, never delete+recreate.
- `CLAUDE.md` stays physically at the repo root (not moved into any subfolder) — it must keep auto-loading as the project's instructions file.
- `evaluation_pipeline/` is out of scope — do not touch anything inside it.
- No logic, prompt format, or hyperparameter changes — only paths and imports.
- **This dev environment does not have `torch`, `transformers`, `sentence_transformers`, or `bitsandbytes` installed** (confirmed via `pip list` — only `networkx`, `numpy`, `pandas`, `tqdm` are present). Every file being moved imports at least one of the missing packages transitively through `config.py` (`import torch`). This means `python <entry>.py --help` **cannot** be used to verify the refactor here — it will fail with `ModuleNotFoundError: No module named 'torch'` regardless of whether the refactor is correct. Verification in this plan is therefore **static**: `python -m py_compile` (syntax) + a small stdlib-`ast`-based import-graph checker (resolves every `from src.* import` to a real file on disk) + `grep` sweeps for leftover flat imports and stale `"evaluation.py"` references. True end-to-end runtime verification (`--help`, an actual pipeline run) must happen next time this code runs on Kaggle or in an environment with the ML deps installed — flag this to the user at the end, don't claim more than the static checks prove.

---

### Task 1: Scaffold target directories

**Files:**
- Create: `src/__init__.py`, `src/core/__init__.py`, `src/retrieval/__init__.py`, `src/generation/__init__.py`, `src/experiments/__init__.py`
- Create: `outputs/predictions/.gitkeep`, `outputs/logs/.gitkeep`, `outputs/tables/.gitkeep`
- Create directories (no marker file needed, populated by later tasks): `external/spider_eval/`, `notebooks/compiled/`, `outputs/plots/`

**Interfaces:**
- Produces: the directory skeleton every later task moves files into. `src/core/`, `src/retrieval/`, `src/generation/`, `src/experiments/` are Python packages (have `__init__.py`).

- [ ] **Step 1: Create all directories**

```bash
mkdir -p src/core src/retrieval src/generation src/experiments
mkdir -p external/spider_eval
mkdir -p notebooks/compiled
mkdir -p outputs/plots outputs/predictions outputs/logs outputs/tables
```

- [ ] **Step 2: Add package markers and .gitkeep files**

```bash
touch src/__init__.py src/core/__init__.py src/retrieval/__init__.py src/generation/__init__.py src/experiments/__init__.py
touch outputs/predictions/.gitkeep outputs/logs/.gitkeep outputs/tables/.gitkeep
```

- [ ] **Step 3: Verify the tree**

Run: `find src external notebooks outputs -type d | sort`
Expected output (order may vary slightly):
```
external
external/spider_eval
notebooks
notebooks/compiled
outputs
outputs/logs
outputs/plots
outputs/predictions
outputs/tables
src
src/core
src/experiments
src/generation
src/retrieval
```

- [ ] **Step 4: Commit**

```bash
git add src external outputs notebooks
git commit -m "chore: scaffold src/, external/, notebooks/, outputs/ directory skeleton"
```

---

### Task 2: Move config.py and schema.py into src/core/

Neither file has any internal (`config`/`schema`/`retrieval`/...) imports today, so this is a pure move — no code edits.

**Files:**
- Move: `config.py` → `src/core/config.py`
- Move: `schema.py` → `src/core/schema.py`

**Interfaces:**
- Produces: `src.core.config.PipelineConfig`, `src.core.schema.load_spider_schema(json_path)`, `src.core.schema.build_schema_graph(df)` — used by every later task.

- [ ] **Step 1: Move the files**

```bash
git mv config.py src/core/config.py
git mv schema.py src/core/schema.py
```

- [ ] **Step 2: Verify syntax**

Run: `python -m py_compile src/core/config.py src/core/schema.py`
Expected: no output, exit code 0.

- [ ] **Step 3: Commit**

```bash
git add -A -- config.py schema.py src/core/config.py src/core/schema.py
git commit -m "refactor: move config.py and schema.py into src/core/"
```

---

### Task 3: Move generation.py and few_shot.py into src/generation/

**Files:**
- Move: `generation.py` → `src/generation/generation.py`
- Move: `few_shot.py` → `src/generation/few_shot.py`
- Modify: `src/generation/generation.py:10`
- Modify: `src/generation/few_shot.py:22`

**Interfaces:**
- Consumes: `src.core.config.PipelineConfig` (Task 2)
- Produces: `src.generation.generation.{build_prompt, generate_sql, generate_sql_with_token_count, load_model_and_tokenizer}`, `src.generation.few_shot.{FewShotIndex, FewShotExample, build_few_shot_index, format_few_shot_block, retrieve_few_shot_examples}` — used by `baseline.py`, `sweep.py`, `ablation.py`, `pipeline.py`.

- [ ] **Step 1: Move the files**

```bash
git mv generation.py src/generation/generation.py
git mv few_shot.py src/generation/few_shot.py
```

- [ ] **Step 2: Fix the import in generation.py**

File: `src/generation/generation.py`

```python
old_string:
from config import PipelineConfig

new_string:
from src.core.config import PipelineConfig
```

- [ ] **Step 3: Fix the import in few_shot.py**

File: `src/generation/few_shot.py`

```python
old_string:
from config import PipelineConfig

new_string:
from src.core.config import PipelineConfig
```

- [ ] **Step 4: Verify syntax**

Run: `python -m py_compile src/generation/generation.py src/generation/few_shot.py`
Expected: no output, exit code 0.

- [ ] **Step 5: Commit**

```bash
git add -A -- generation.py few_shot.py src/generation/generation.py src/generation/few_shot.py
git commit -m "refactor: move generation.py and few_shot.py into src/generation/"
```

---

### Task 4: Move retrieval.py into src/retrieval/

**Files:**
- Move: `retrieval.py` → `src/retrieval/retrieval.py`
- Modify: `src/retrieval/retrieval.py:25`

**Interfaces:**
- Consumes: `src.core.config.PipelineConfig` (Task 2)
- Produces: `src.retrieval.retrieval.{SchemaIndex, build_schema_index, build_schema_context, evaluate_schema_linking, semantic_schema_linking, trace_schema_paths, _parse_gold_elements, _SQL_STOPWORDS}` — used by `baseline.py`, `sweep.py`, `ablation.py`, `pipeline.py`.

- [ ] **Step 1: Move the file**

```bash
git mv retrieval.py src/retrieval/retrieval.py
```

- [ ] **Step 2: Fix the import**

File: `src/retrieval/retrieval.py`

```python
old_string:
from config import PipelineConfig

new_string:
from src.core.config import PipelineConfig
```

- [ ] **Step 3: Verify syntax**

Run: `python -m py_compile src/retrieval/retrieval.py`
Expected: no output, exit code 0.

- [ ] **Step 4: Commit**

```bash
git add -A -- retrieval.py src/retrieval/retrieval.py
git commit -m "refactor: move retrieval.py into src/retrieval/"
```

---

### Task 5: Move baseline.py into src/retrieval/

`baseline.py` is a CLI entry point (`python baseline.py --sample 1.0`), so it needs the `sys.path` bootstrap. It has 3 top-level internal imports, 3 lazy (in-function) internal imports, an `evaluation.py` path reference, 2 `subprocess.run` calls, and 4 hardcoded output-path defaults to fix.

**Files:**
- Move: `baseline.py` → `src/retrieval/baseline.py`
- Modify: `src/retrieval/baseline.py` (multiple locations, see steps)

**Interfaces:**
- Consumes: `src.core.config.PipelineConfig`, `src.core.schema.{build_schema_graph, load_spider_schema}`, `src.generation.generation.{build_prompt, generate_sql, load_model_and_tokenizer}`, `src.retrieval.retrieval.{_SQL_STOPWORDS, build_schema_index}`. Also lazily imports `src.experiments.pipeline.{main as run_graphrag_main, PipelineResult, run_single}` — this specific import only resolves once Task 8 is done; that's fine, it's inside a function body, never evaluated at import time.
- Produces: `src.retrieval.baseline.{BaselineResult, build_table_graph, build_table_index, run_single_baseline, run_baseline, semantic_linking_table_level, trace_table_paths, build_table_schema_context, evaluate_table_linking, _save_predictions, _save_log, _save_csv, _print_summary, _run_spider_eval}` — used by `ablation.py` and `pipeline.py`.

- [ ] **Step 1: Move the file**

```bash
git mv baseline.py src/retrieval/baseline.py
```

- [ ] **Step 2: Add the sys.path bootstrap and fix the top-level imports**

File: `src/retrieval/baseline.py`

```python
old_string:
from config import PipelineConfig
from generation import build_prompt, generate_sql, load_model_and_tokenizer
from schema import build_schema_graph, load_spider_schema

new_string:
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.config import PipelineConfig
from src.generation.generation import build_prompt, generate_sql, load_model_and_tokenizer
from src.core.schema import build_schema_graph, load_spider_schema
```

(`sys` and `Path` are already imported earlier in this file's import block — no new imports needed.)

- [ ] **Step 3: Fix the lazy import at the top of the recall/precision helper**

File: `src/retrieval/baseline.py`

```python
old_string:
    from retrieval import _SQL_STOPWORDS

new_string:
    from src.retrieval.retrieval import _SQL_STOPWORDS
```

- [ ] **Step 4: Fix the two lazy imports inside run_single_baseline (or equivalent) and build_schema_index usage**

File: `src/retrieval/baseline.py`

```python
old_string:
    # Import lazily to avoid circular imports when baseline is used standalone
    from pipeline import main as run_graphrag_main, PipelineResult, run_single

    schema_df = load_spider_schema(cfg.tables_json)
    from schema import build_schema_graph
    graph = build_schema_graph(schema_df)

new_string:
    # Import lazily to avoid circular imports when baseline is used standalone
    from src.experiments.pipeline import main as run_graphrag_main, PipelineResult, run_single

    schema_df = load_spider_schema(cfg.tables_json)
    from src.core.schema import build_schema_graph
    graph = build_schema_graph(schema_df)
```

```python
old_string:
    from retrieval import build_schema_index

new_string:
    from src.retrieval.retrieval import build_schema_index
```

- [ ] **Step 5: Fix the evaluation.py path and warning message**

File: `src/retrieval/baseline.py`

```python
old_string:
def _run_spider_eval(pred_file: Path, cfg: PipelineConfig) -> None:
    evaluator = Path("evaluation.py")
    if not evaluator.exists():
        logger.warning(
            "evaluation.py not found — skipping Spider eval.\n"
            "  wget https://raw.githubusercontent.com/taoyds/spider/master/evaluation.py\n"
            "  wget https://raw.githubusercontent.com/taoyds/spider/master/process_sql.py"
        )
        return

new_string:
def _run_spider_eval(pred_file: Path, cfg: PipelineConfig) -> None:
    evaluator = Path("external/spider_eval/evaluation.py")
    if not evaluator.exists():
        logger.warning(
            "external/spider_eval/evaluation.py not found — skipping Spider eval.\n"
            "  mkdir -p external/spider_eval && cd external/spider_eval\n"
            "  wget https://raw.githubusercontent.com/taoyds/spider/master/evaluation.py\n"
            "  wget https://raw.githubusercontent.com/taoyds/spider/master/process_sql.py"
        )
        return
```

- [ ] **Step 6: Fix the two subprocess.run calls**

File: `src/retrieval/baseline.py`

```python
old_string:
    subprocess.run(["python", "evaluation.py"] + base_args + ["--etype", "match"], check=False)

    print("\n" + "=" * 60)
    print("  SPIDER EVALUATION (Execution Accuracy)")
    print("=" * 60)
    subprocess.run(["python", "evaluation.py"] + base_args + ["--etype", "exec"], check=False)

new_string:
    subprocess.run(["python", "external/spider_eval/evaluation.py"] + base_args + ["--etype", "match"], check=False)

    print("\n" + "=" * 60)
    print("  SPIDER EVALUATION (Execution Accuracy)")
    print("=" * 60)
    subprocess.run(["python", "external/spider_eval/evaluation.py"] + base_args + ["--etype", "exec"], check=False)
```

- [ ] **Step 7: Fix the module docstring's output file list**

File: `src/retrieval/baseline.py`

```python
old_string:
Output files:
    baseline_predictions.txt    — SQL predictions (Spider format)
    baseline_log.txt            — per-sample detail log
    baseline_results.csv        — recall / precision per sample
    comparison_report.txt       — side-by-side summary (only with --compare)

new_string:
Output files:
    outputs/predictions/baseline_predictions.txt    — SQL predictions (Spider format)
    outputs/logs/baseline_log.txt                   — per-sample detail log
    outputs/tables/baseline_results.csv             — recall / precision per sample
    outputs/tables/comparison_report.txt            — side-by-side summary (only with --compare)
```

- [ ] **Step 8: Fix the four hardcoded output-path defaults**

File: `src/retrieval/baseline.py`

```python
old_string:
    pred_path: Path = Path("baseline_predictions.txt"),
    log_path:  Path = Path("baseline_log.txt"),
    csv_path:  Path = Path("baseline_results.csv"),

new_string:
    pred_path: Path = Path("outputs/predictions/baseline_predictions.txt"),
    log_path:  Path = Path("outputs/logs/baseline_log.txt"),
    csv_path:  Path = Path("outputs/tables/baseline_results.csv"),
```

```python
old_string:
        pred_path=Path("baseline_predictions.txt"),
        log_path=Path("baseline_log.txt"),
        csv_path=Path("baseline_results.csv"),

new_string:
        pred_path=Path("outputs/predictions/baseline_predictions.txt"),
        log_path=Path("outputs/logs/baseline_log.txt"),
        csv_path=Path("outputs/tables/baseline_results.csv"),
```

```python
old_string:
    report_path = Path("comparison_report.txt")

new_string:
    report_path = Path("outputs/tables/comparison_report.txt")
```

- [ ] **Step 9: Verify syntax**

Run: `python -m py_compile src/retrieval/baseline.py`
Expected: no output, exit code 0.

- [ ] **Step 10: Verify no flat internal imports remain**

Run: `grep -nE "^from (config|schema|retrieval|generation) import|^\s+from (config|schema|retrieval|generation|pipeline) import" src/retrieval/baseline.py`
Expected: no output (all such lines now start with `from src.`).

- [ ] **Step 11: Commit**

```bash
git add -A -- baseline.py src/retrieval/baseline.py
git commit -m "refactor: move baseline.py into src/retrieval/, fix imports and output paths"
```

---

### Task 6: Move sweep.py into src/experiments/

**Files:**
- Move: `sweep.py` → `src/experiments/sweep.py`
- Modify: `src/experiments/sweep.py` (multiple locations, see steps)

**Interfaces:**
- Consumes: `src.core.config.PipelineConfig`, `src.core.schema.{build_schema_graph, load_spider_schema}`, `src.retrieval.retrieval.{SchemaIndex, build_schema_index, evaluate_schema_linking, semantic_schema_linking, trace_schema_paths}`
- Produces: `src.experiments.sweep.run_sweep_and_get_best` — used by `pipeline.py`.

- [ ] **Step 1: Move the file**

```bash
git mv sweep.py src/experiments/sweep.py
```

- [ ] **Step 2: Add `import sys`, the bootstrap, and fix the top-level imports**

File: `src/experiments/sweep.py`

```python
old_string:
import argparse
import csv
import json
import logging
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import PipelineConfig
from retrieval import (
    SchemaIndex,
    build_schema_index,
    evaluate_schema_linking,
    semantic_schema_linking,
    trace_schema_paths,
)
from schema import build_schema_graph, load_spider_schema

new_string:
import argparse
import csv
import json
import logging
import sys
from dataclasses import dataclass
from itertools import product
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.config import PipelineConfig
from src.retrieval.retrieval import (
    SchemaIndex,
    build_schema_index,
    evaluate_schema_linking,
    semantic_schema_linking,
    trace_schema_paths,
)
from src.core.schema import build_schema_graph, load_spider_schema
```

- [ ] **Step 3: Fix the lazy import inside the sweep runner**

File: `src/experiments/sweep.py`

```python
old_string:
    import json
    from sentence_transformers import SentenceTransformer
    from schema import build_schema_graph, load_spider_schema

new_string:
    import json
    from sentence_transformers import SentenceTransformer
    from src.core.schema import build_schema_graph, load_spider_schema
```

- [ ] **Step 4: Fix both hardcoded output-path defaults**

File: `src/experiments/sweep.py`

```python
old_string:
    _save_csv(results, Path("sweep_results.csv"))

new_string:
    _save_csv(results, Path("outputs/tables/sweep_results.csv"))
```

Use `replace_all: true` for this edit — the identical line appears twice (once in each of the two sweep-running functions) and both need the same fix.

- [ ] **Step 5: Verify syntax**

Run: `python -m py_compile src/experiments/sweep.py`
Expected: no output, exit code 0.

- [ ] **Step 6: Verify no flat internal imports remain**

Run: `grep -nE "^from (config|schema|retrieval) import|^\s+from (config|schema|retrieval) import" src/experiments/sweep.py`
Expected: no output.

- [ ] **Step 7: Commit**

```bash
git add -A -- sweep.py src/experiments/sweep.py
git commit -m "refactor: move sweep.py into src/experiments/, fix imports and output paths"
```

---

### Task 7: Move ablation.py into src/experiments/

**Files:**
- Move: `ablation.py` → `src/experiments/ablation.py`
- Modify: `src/experiments/ablation.py` (multiple locations, see steps)

**Interfaces:**
- Consumes: `src.core.config.PipelineConfig`, `src.core.schema.{build_schema_graph, load_spider_schema}`, `src.generation.few_shot.{FewShotIndex, build_few_shot_index, format_few_shot_block, retrieve_few_shot_examples}`, `src.generation.generation.{build_prompt, generate_sql_with_token_count, load_model_and_tokenizer}`, `src.retrieval.retrieval.{build_schema_context, build_schema_index, evaluate_schema_linking, semantic_schema_linking, trace_schema_paths}`, `src.retrieval.baseline.{semantic_linking_table_level, trace_table_paths, build_table_schema_context, evaluate_table_linking, build_table_graph, build_table_index}` (Task 5)
- Produces: nothing consumed by other tasks — `ablation.py` is a leaf CLI entry point.

- [ ] **Step 1: Move the file**

```bash
git mv ablation.py src/experiments/ablation.py
```

- [ ] **Step 2: Add `import sys`, the bootstrap, and fix the top-level imports**

File: `src/experiments/ablation.py`

```python
old_string:
import argparse
import csv
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import PipelineConfig
from few_shot import FewShotIndex, build_few_shot_index, format_few_shot_block, retrieve_few_shot_examples
from generation import build_prompt, generate_sql_with_token_count, load_model_and_tokenizer
from retrieval import (
    build_schema_context,
    build_schema_index,
    evaluate_schema_linking,
    semantic_schema_linking,
    trace_schema_paths,
)
from schema import build_schema_graph, load_spider_schema

new_string:
import argparse
import csv
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.config import PipelineConfig
from src.generation.few_shot import FewShotIndex, build_few_shot_index, format_few_shot_block, retrieve_few_shot_examples
from src.generation.generation import build_prompt, generate_sql_with_token_count, load_model_and_tokenizer
from src.retrieval.retrieval import (
    build_schema_context,
    build_schema_index,
    evaluate_schema_linking,
    semantic_schema_linking,
    trace_schema_paths,
)
from src.core.schema import build_schema_graph, load_spider_schema
```

- [ ] **Step 3: Fix the two lazy imports from baseline**

File: `src/experiments/ablation.py`

```python
old_string:
    from baseline import (
        semantic_linking_table_level,
        trace_table_paths,
        build_table_schema_context,
        evaluate_table_linking,
    )

new_string:
    from src.retrieval.baseline import (
        semantic_linking_table_level,
        trace_table_paths,
        build_table_schema_context,
        evaluate_table_linking,
    )
```

```python
old_string:
        from baseline import build_table_graph, build_table_index

new_string:
        from src.retrieval.baseline import build_table_graph, build_table_index
```

- [ ] **Step 4: Fix the eval-command print string**

File: `src/experiments/ablation.py`

```python
old_string:
                print(
                    f"  python evaluation.py --gold {gold} "
                    f"--pred {r.predictions_file} "
                    f"--db {db_dir} --table {tables} --etype {etype}"
                )

new_string:
                print(
                    f"  python external/spider_eval/evaluation.py --gold {gold} "
                    f"--pred {r.predictions_file} "
                    f"--db {db_dir} --table {tables} --etype {etype}"
                )
```

- [ ] **Step 5: Fix the evaluation.py path, warning message, and subprocess call**

File: `src/experiments/ablation.py`

```python
old_string:
    import subprocess
    evaluator = Path("evaluation.py")
    if not evaluator.exists():
        logger.warning(
            "evaluation.py not found — cannot auto-run Spider eval.\n"
            "Download it first:\n"
            "  wget https://raw.githubusercontent.com/taoyds/spider/master/evaluation.py\n"
            "  wget https://raw.githubusercontent.com/taoyds/spider/master/process_sql.py"
        )
        return

new_string:
    import subprocess
    evaluator = Path("external/spider_eval/evaluation.py")
    if not evaluator.exists():
        logger.warning(
            "external/spider_eval/evaluation.py not found — cannot auto-run Spider eval.\n"
            "Download it first:\n"
            "  mkdir -p external/spider_eval && cd external/spider_eval\n"
            "  wget https://raw.githubusercontent.com/taoyds/spider/master/evaluation.py\n"
            "  wget https://raw.githubusercontent.com/taoyds/spider/master/process_sql.py"
        )
        return
```

```python
old_string:
            subprocess.run(
                ["python", "evaluation.py"] + base_args +
                ["--pred", str(r.predictions_file), "--etype", etype],
                check=False,
            )

new_string:
            subprocess.run(
                ["python", "external/spider_eval/evaluation.py"] + base_args +
                ["--pred", str(r.predictions_file), "--etype", etype],
                check=False,
            )
```

- [ ] **Step 6: Fix the output_dir default and the results CSV path**

File: `src/experiments/ablation.py`

```python
old_string:
    cfg = PipelineConfig()
    output_dir = Path(".")

new_string:
    cfg = PipelineConfig()
    output_dir = Path("outputs/predictions")
```

```python
old_string:
    _print_table(results, cfg)
    _save_csv(results, Path("ablation_results.csv"))

new_string:
    _print_table(results, cfg)
    _save_csv(results, Path("outputs/tables/ablation_results.csv"))
```

- [ ] **Step 7: Verify syntax**

Run: `python -m py_compile src/experiments/ablation.py`
Expected: no output, exit code 0.

- [ ] **Step 8: Verify no flat internal imports remain**

Run: `grep -nE "^from (config|schema|retrieval|generation|few_shot|baseline) import|^\s+from (config|schema|retrieval|generation|few_shot|baseline) import" src/experiments/ablation.py`
Expected: no output.

- [ ] **Step 9: Commit**

```bash
git add -A -- ablation.py src/experiments/ablation.py
git commit -m "refactor: move ablation.py into src/experiments/, fix imports and output paths"
```

---

### Task 8: Move pipeline.py into src/experiments/

**Files:**
- Move: `pipeline.py` → `src/experiments/pipeline.py`
- Modify: `src/experiments/pipeline.py` (multiple locations, see steps)

**Interfaces:**
- Consumes: `src.core.config.PipelineConfig`, `src.core.schema.{build_schema_graph, load_spider_schema}`, `src.experiments.sweep.run_sweep_and_get_best` (Task 6), `src.generation.few_shot.{FewShotIndex, build_few_shot_index, format_few_shot_block, retrieve_few_shot_examples}`, `src.generation.generation.{build_prompt, generate_sql, load_model_and_tokenizer}`, `src.retrieval.retrieval.{SchemaIndex, build_schema_context, build_schema_index, evaluate_schema_linking, semantic_schema_linking, trace_schema_paths, _parse_gold_elements}`, `src.retrieval.baseline.{BaselineResult, build_table_graph, build_table_index, run_single_baseline, _save_predictions, _save_log, _save_csv, _print_summary, _run_spider_eval}` (Task 5)
- Produces: `src.experiments.pipeline.{main, PipelineResult, run_single}` — consumed lazily by `src.retrieval.baseline` (Task 5's forward reference, now resolvable).

- [ ] **Step 1: Move the file**

```bash
git mv pipeline.py src/experiments/pipeline.py
```

- [ ] **Step 2: Add the bootstrap and fix the top-level imports**

File: `src/experiments/pipeline.py`

```python
old_string:
from config import PipelineConfig
from sweep import run_sweep_and_get_best
from few_shot import FewShotIndex, build_few_shot_index, format_few_shot_block, retrieve_few_shot_examples
from generation import build_prompt, generate_sql, load_model_and_tokenizer
from retrieval import (
    SchemaIndex,
    build_schema_context,
    build_schema_index,
    evaluate_schema_linking,
    semantic_schema_linking,
    trace_schema_paths,
)
from schema import build_schema_graph, load_spider_schema

new_string:
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.core.config import PipelineConfig
from src.experiments.sweep import run_sweep_and_get_best
from src.generation.few_shot import FewShotIndex, build_few_shot_index, format_few_shot_block, retrieve_few_shot_examples
from src.generation.generation import build_prompt, generate_sql, load_model_and_tokenizer
from src.retrieval.retrieval import (
    SchemaIndex,
    build_schema_context,
    build_schema_index,
    evaluate_schema_linking,
    semantic_schema_linking,
    trace_schema_paths,
)
from src.core.schema import build_schema_graph, load_spider_schema
```

(`sys` and `Path` are already imported earlier in this file's import block — no new imports needed.)

- [ ] **Step 3: Fix the lazy retrieval import**

File: `src/experiments/pipeline.py`

```python
old_string:
    from retrieval import _parse_gold_elements, build_schema_context

new_string:
    from src.retrieval.retrieval import _parse_gold_elements, build_schema_context
```

- [ ] **Step 4: Fix the evaluation.py path, warning message, and both subprocess calls**

File: `src/experiments/pipeline.py`

```python
old_string:
def run_official_evaluation(cfg: PipelineConfig) -> None:
    """Run the Spider official evaluation script for EM and EX metrics."""
    evaluator = Path("evaluation.py")
    if not evaluator.exists():
        logger.warning(
            "evaluation.py not found — skipping official Spider evaluation.\n"
            "Download it with:\n"
            "  wget https://raw.githubusercontent.com/taoyds/spider/master/evaluation.py\n"
            "  wget https://raw.githubusercontent.com/taoyds/spider/master/process_sql.py"
        )
        return

new_string:
def run_official_evaluation(cfg: PipelineConfig) -> None:
    """Run the Spider official evaluation script for EM and EX metrics."""
    evaluator = Path("external/spider_eval/evaluation.py")
    if not evaluator.exists():
        logger.warning(
            "external/spider_eval/evaluation.py not found — skipping official Spider evaluation.\n"
            "Download it with:\n"
            "  mkdir -p external/spider_eval && cd external/spider_eval\n"
            "  wget https://raw.githubusercontent.com/taoyds/spider/master/evaluation.py\n"
            "  wget https://raw.githubusercontent.com/taoyds/spider/master/process_sql.py"
        )
        return
```

```python
old_string:
    subprocess.run(["python", "evaluation.py"] + common_args + ["--etype", "match"], check=False)

    print("\n" + "=" * 60)
    print("🎯 OFFICIAL SPIDER EVALUATION (Execution Accuracy)")
    print("=" * 60)
    subprocess.run(["python", "evaluation.py"] + common_args + ["--etype", "exec"], check=False)

new_string:
    subprocess.run(["python", "external/spider_eval/evaluation.py"] + common_args + ["--etype", "match"], check=False)

    print("\n" + "=" * 60)
    print("🎯 OFFICIAL SPIDER EVALUATION (Execution Accuracy)")
    print("=" * 60)
    subprocess.run(["python", "external/spider_eval/evaluation.py"] + common_args + ["--etype", "exec"], check=False)
```

- [ ] **Step 5: Fix the two lazy imports from baseline**

File: `src/experiments/pipeline.py`

```python
old_string:
    from baseline import (
        BaselineResult,
        build_table_graph,
        build_table_index,
        run_single_baseline,
        _save_predictions as bl_save_pred,
        _save_log        as bl_save_log,
        _save_csv        as bl_save_csv,
        _print_summary   as bl_print_summary,
        _run_spider_eval as bl_spider_eval,
    )

new_string:
    from src.retrieval.baseline import (
        BaselineResult,
        build_table_graph,
        build_table_index,
        run_single_baseline,
        _save_predictions as bl_save_pred,
        _save_log        as bl_save_log,
        _save_csv        as bl_save_csv,
        _print_summary   as bl_print_summary,
        _run_spider_eval as bl_spider_eval,
    )
```

```python
old_string:
        from baseline import (
            BaselineResult,
            build_table_graph,
            build_table_index,
            run_single_baseline,
            _save_predictions,
            _save_log,
            _save_csv,
            _print_summary,
            _run_spider_eval,
        )

new_string:
        from src.retrieval.baseline import (
            BaselineResult,
            build_table_graph,
            build_table_index,
            run_single_baseline,
            _save_predictions,
            _save_log,
            _save_csv,
            _print_summary,
            _run_spider_eval,
        )
```

- [ ] **Step 6: Fix the lazy few_shot import**

File: `src/experiments/pipeline.py`

```python
old_string:
        from few_shot import build_few_shot_index

new_string:
        from src.generation.few_shot import build_few_shot_index
```

- [ ] **Step 7: Fix the predictions/log/csv output-path defaults**

File: `src/experiments/pipeline.py`

```python
old_string:
    g_pred_path = Path("predictions.txt")
    b_pred_path = Path("baseline_predictions.txt")

new_string:
    g_pred_path = Path("outputs/predictions/predictions.txt")
    b_pred_path = Path("outputs/predictions/baseline_predictions.txt")
```

```python
old_string:
    bl_save_log(baseline_results, Path("baseline_log.txt"))
    bl_save_csv(baseline_results, Path("baseline_results.csv"))

new_string:
    bl_save_log(baseline_results, Path("outputs/logs/baseline_log.txt"))
    bl_save_csv(baseline_results, Path("outputs/tables/baseline_results.csv"))
```

- [ ] **Step 8: Fix the comparison report path and its log message**

File: `src/experiments/pipeline.py`

```python
old_string:
    report = "\n".join(report_lines)
    print(report)
    Path("comparison_report.txt").write_text(report, encoding="utf-8")
    logger.info("Comparison report → comparison_report.txt")

new_string:
    report = "\n".join(report_lines)
    print(report)
    Path("outputs/tables/comparison_report.txt").write_text(report, encoding="utf-8")
    logger.info("Comparison report → outputs/tables/comparison_report.txt")
```

- [ ] **Step 9: Verify syntax**

Run: `python -m py_compile src/experiments/pipeline.py`
Expected: no output, exit code 0.

- [ ] **Step 10: Verify no flat internal imports remain**

Run: `grep -nE "^from (config|schema|retrieval|generation|few_shot|sweep|baseline) import|^\s+from (config|schema|retrieval|generation|few_shot|sweep|baseline) import" src/experiments/pipeline.py`
Expected: no output.

- [ ] **Step 11: Commit**

```bash
git add -A -- pipeline.py src/experiments/pipeline.py
git commit -m "refactor: move pipeline.py into src/experiments/, fix imports and output paths"
```

---

### Task 9: Move evaluation.py and process_sql.py into external/spider_eval/

Both scripts take all paths as CLI args (`--gold`, `--pred`, `--db`, `--table`) — no hardcoded relative paths exist in either file (verified: only `open(gold)`, `open(predict)`, `os.path.join(db_dir, ...)`, `open(table)` in `evaluation.py`, and `open(fpath)` in `process_sql.py`, all using variables, not literals). `evaluation.py`'s `from process_sql import ...` stays valid since both files move into the same folder together. No code edits needed.

**Files:**
- Move: `evaluation.py` → `external/spider_eval/evaluation.py`
- Move: `process_sql.py` → `external/spider_eval/process_sql.py`

**Interfaces:**
- Produces: `external/spider_eval/evaluation.py` invoked via subprocess by `src/retrieval/baseline.py`, `src/experiments/pipeline.py`, `src/experiments/ablation.py` (all already updated in Tasks 5, 7, 8 to reference this new path).

- [ ] **Step 1: Move the files**

```bash
git mv evaluation.py external/spider_eval/evaluation.py
git mv process_sql.py external/spider_eval/process_sql.py
```

- [ ] **Step 2: Verify syntax**

Run: `python -m py_compile external/spider_eval/evaluation.py external/spider_eval/process_sql.py`
Expected: no output, exit code 0.

- [ ] **Step 3: Verify the sibling import still reads correctly**

Run: `grep -n "^from process_sql import" external/spider_eval/evaluation.py`
Expected: `from process_sql import tokenize, get_schema, get_tables_with_alias, Schema, get_sql` — unchanged, still valid since both files are now siblings in `external/spider_eval/`.

- [ ] **Step 4: Commit**

```bash
git add -A -- evaluation.py process_sql.py external/spider_eval/evaluation.py external/spider_eval/process_sql.py
git commit -m "refactor: move evaluation.py and process_sql.py into external/spider_eval/"
```

---

### Task 10: Move notebooks and plot images

**Files:**
- Move: `graphrag-text-to-sql-eda.ipynb` → `notebooks/graphrag-text-to-sql-eda.ipynb`
- Move: `Text-To-SQL_Graph_Representation.ipynb` → `notebooks/Text-To-SQL_Graph_Representation.ipynb`
- Move: `notebook_compiled/graphrag_text2sql.ipynb` → `notebooks/compiled/graphrag_text2sql.ipynb`
- Move: all 16 files in `plot/` → `outputs/plots/`
- Remove: now-empty `notebook_compiled/` and `plot/` directories

**Interfaces:**
- None — these are terminal artifacts, nothing imports them.

- [ ] **Step 1: Move the notebooks**

```bash
git mv graphrag-text-to-sql-eda.ipynb notebooks/graphrag-text-to-sql-eda.ipynb
git mv "Text-To-SQL_Graph_Representation.ipynb" "notebooks/Text-To-SQL_Graph_Representation.ipynb"
git mv notebook_compiled/graphrag_text2sql.ipynb notebooks/compiled/graphrag_text2sql.ipynb
```

- [ ] **Step 2: Move the plot images**

```bash
git mv plot/plot_01_queries_per_db.png outputs/plots/plot_01_queries_per_db.png
git mv plot/plot_02_schema_complexity.png outputs/plots/plot_02_schema_complexity.png
git mv plot/plot_03_schema_corr.png outputs/plots/plot_03_schema_corr.png
git mv plot/plot_04_sql_clause_freq.png outputs/plots/plot_04_sql_clause_freq.png
git mv plot/plot_05_hardness.png outputs/plots/plot_05_hardness.png
git mv plot/plot_06_sql_length.png outputs/plots/plot_06_sql_length.png
git mv plot/plot_06b_sql_tokens_by_diff.png outputs/plots/plot_06b_sql_tokens_by_diff.png
git mv plot/plot_07_question_length.png outputs/plots/plot_07_question_length.png
git mv plot/plot_08_q_vs_sql.png outputs/plots/plot_08_q_vs_sql.png
git mv plot/plot_09_token_budget.png outputs/plots/plot_09_token_budget.png
git mv plot/plot_10_token_composition.png outputs/plots/plot_10_token_composition.png
git mv plot/plot_11_tep_foundation.png outputs/plots/plot_11_tep_foundation.png
git mv plot/plot_12_schema_linking.png outputs/plots/plot_12_schema_linking.png
git mv plot/plot_13_fewshot.png outputs/plots/plot_13_fewshot.png
git mv plot/plot_ego1_concert_singer.png outputs/plots/plot_ego1_concert_singer.png
git mv plot/plot_ego2_concert_singer.png outputs/plots/plot_ego2_concert_singer.png
```

- [ ] **Step 3: Remove the now-empty source directories**

```bash
rmdir notebook_compiled plot
```

- [ ] **Step 4: Verify nothing was left behind**

Run: `ls notebook_compiled plot 2>&1`
Expected: `No such file or directory` for both (they no longer exist).

Run: `ls notebooks notebooks/compiled outputs/plots | wc -l`
Expected: a count reflecting 2 root notebooks + 1 compiled notebook + 16 plots (exact total depends on `ls` formatting, just confirm no file is missing by eye).

- [ ] **Step 5: Commit**

```bash
git add -A -- graphrag-text-to-sql-eda.ipynb "Text-To-SQL_Graph_Representation.ipynb" notebook_compiled plot notebooks outputs
git commit -m "refactor: move notebooks into notebooks/, plot images into outputs/plots/"
```

---

### Task 11: Update CLAUDE.md

**Files:**
- Modify: `CLAUDE.md` (stays at repo root — do not move this file)

**Interfaces:** None — documentation only.

- [ ] **Step 1: Rewrite the "Struktur File" tree**

File: `CLAUDE.md`

```markdown
old_string:
## Struktur File

```
text2sql_pipeline/
├── config.py       — semua hyperparameter dan path (sumber kebenaran tunggal)
├── schema.py       — parsing tables.json Spider → DataFrame → NetworkX graph (column-level)
├── retrieval.py    — GraphRAG: two-stage semantic linking, graph traversal, path pruning, context builder
├── baseline.py     — Baseline: table-level graph, table-level semantic linking, context builder
├── few_shot.py     — few-shot index: pre-compute training set embeddings, dynamic retrieval
├── generation.py   — prompt builder, SQL cleaner, model loading (4-bit quantized), greedy decode
├── pipeline.py     — orchestration utama, CLI entry point, Spider evaluation
├── sweep.py        — hyperparameter sweep: top_k_tables × top_k_columns (tanpa LLM)
├── ablation.py     — ablation study: few-shot k={0,1,3,5} × {baseline, graphrag}
└── CLAUDE.md       — file ini
```

new_string:
## Struktur File

```
thesis-2803/
├── src/
│   ├── core/
│   │   ├── config.py       — semua hyperparameter dan path (sumber kebenaran tunggal)
│   │   └── schema.py       — parsing tables.json Spider → DataFrame → NetworkX graph (column-level)
│   ├── retrieval/
│   │   ├── retrieval.py    — GraphRAG: two-stage semantic linking, graph traversal, path pruning, context builder
│   │   └── baseline.py     — Baseline: table-level graph, table-level semantic linking, context builder
│   ├── generation/
│   │   ├── generation.py   — prompt builder, SQL cleaner, model loading (4-bit quantized), greedy decode
│   │   └── few_shot.py     — few-shot index: pre-compute training set embeddings, dynamic retrieval
│   └── experiments/
│       ├── pipeline.py     — orchestration utama, CLI entry point, Spider evaluation
│       ├── sweep.py        — hyperparameter sweep: top_k_tables × top_k_columns (tanpa LLM)
│       └── ablation.py     — ablation study: few-shot k={0,1,3,5} × {baseline, graphrag}
├── external/
│   └── spider_eval/        — official SPIDER evaluation.py + process_sql.py (di-wget manual)
├── notebooks/               — EDA notebooks + notebooks/compiled/ (Kaggle-ready compiled notebook)
├── outputs/
│   ├── plots/               — EDA plot PNGs
│   ├── predictions/         — predictions.txt, baseline_predictions.txt, ablation_*_predictions_k*.txt
│   ├── logs/                — baseline_log.txt
│   └── tables/              — sweep_results.csv, ablation_results.csv, baseline_results.csv, comparison_report.txt
├── evaluation_pipeline/     — proyek terpisah: metrics + dimensi analisis skripsi (lihat evaluation_pipeline/CLAUDE.md-nya sendiri kalau ada)
└── CLAUDE.md                — file ini (tetap di root)
```

Semua modul di `src/` dipanggil sebagai `src.<paket>.<modul>` (mis. `from src.core.config import PipelineConfig`). Entry point CLI (`pipeline.py`, `sweep.py`, `ablation.py`, `baseline.py`) masih bisa dijalankan langsung dengan `python src/experiments/pipeline.py` dkk — setiap entry point punya bootstrap `sys.path.insert(...)` di baris import supaya import `src.*`-nya tetap resolve meski dijalankan sebagai script, bukan module.
```

- [ ] **Step 2: Update the ablation run commands**

File: `CLAUDE.md`

```markdown
old_string:
**Run 1 — Baseline RAG:**
```bash
python ablation.py --mode baseline --k-values 0 1 3 5 --sample 1.0
```

**Run 2 — GraphRAG:**
```bash
python ablation.py --mode graphrag --k-values 0 1 3 5 --sample 1.0
```

new_string:
**Run 1 — Baseline RAG:**
```bash
python src/experiments/ablation.py --mode baseline --k-values 0 1 3 5 --sample 1.0
```

**Run 2 — GraphRAG:**
```bash
python src/experiments/ablation.py --mode graphrag --k-values 0 1 3 5 --sample 1.0
```
```

- [ ] **Step 3: Update the ablation output file descriptions**

File: `CLAUDE.md`

```markdown
old_string:
Output per run:
- `ablation_{mode}_predictions_k{k}.txt` — SQL predictions → masuk ke Spider `evaluation.py` untuk EM/EX
- `ablation_{mode}_prompts_k{k}.jsonl` — per sample: i, db_id, question, tokens_in, tokens_out, token_consumption, prompt, pred_sql
- `ablation_results.csv` — avg_recall, avg_precision, avg_prompt_tokens (T_in), avg_output_tokens (T_out), avg_token_consumption (T) per mode×k → dasar perhitungan TEP

new_string:
Output per run:
- `outputs/predictions/ablation_{mode}_predictions_k{k}.txt` — SQL predictions → masuk ke Spider `external/spider_eval/evaluation.py` untuk EM/EX
- `outputs/predictions/ablation_{mode}_prompts_k{k}.jsonl` — per sample: i, db_id, question, tokens_in, tokens_out, token_consumption, prompt, pred_sql
- `outputs/tables/ablation_results.csv` — avg_recall, avg_precision, avg_prompt_tokens (T_in), avg_output_tokens (T_out), avg_token_consumption (T) per mode×k → dasar perhitungan TEP
```

- [ ] **Step 4: Update the "Spider evaluation scripts" download note**

File: `CLAUDE.md`

```markdown
old_string:
Spider evaluation scripts harus didownload manual:
```bash
wget https://raw.githubusercontent.com/taoyds/spider/master/evaluation.py
wget https://raw.githubusercontent.com/taoyds/spider/master/process_sql.py
```

new_string:
Spider evaluation scripts harus didownload manual ke `external/spider_eval/`:
```bash
mkdir -p external/spider_eval && cd external/spider_eval
wget https://raw.githubusercontent.com/taoyds/spider/master/evaluation.py
wget https://raw.githubusercontent.com/taoyds/spider/master/process_sql.py
```
```

- [ ] **Step 5: Update the CLI Quick Reference block**

File: `CLAUDE.md`

```markdown
old_string:
## CLI Quick Reference

```bash
# GraphRAG — full dev set
python pipeline.py --skip-sweep

# GraphRAG — dengan sweep otomatis dulu
python pipeline.py

# Baseline — table-level retrieval
python baseline.py --sample 1.0

# Full schema bypass (eksperimen, belum jadi mode resmi)
python pipeline.py --full-schema

# Ablation few-shot
python ablation.py --k-values 0 1 3 5

# Hyperparameter sweep saja (tanpa LLM, cepat)
python sweep.py --sample 0.2

# Perbandingan GraphRAG vs Baseline
python pipeline.py --baseline
```

new_string:
## CLI Quick Reference

```bash
# GraphRAG — full dev set
python src/experiments/pipeline.py --skip-sweep

# GraphRAG — dengan sweep otomatis dulu
python src/experiments/pipeline.py

# Baseline — table-level retrieval
python src/retrieval/baseline.py --sample 1.0

# Full schema bypass (eksperimen, belum jadi mode resmi)
python src/experiments/pipeline.py --full-schema

# Ablation few-shot
python src/experiments/ablation.py --k-values 0 1 3 5

# Hyperparameter sweep saja (tanpa LLM, cepat)
python src/experiments/sweep.py --sample 0.2

# Perbandingan GraphRAG vs Baseline
python src/experiments/pipeline.py --baseline
```
```

- [ ] **Step 6: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: update CLAUDE.md paths and commands for new src/ layout"
```

---

### Task 12: Update README.md

**Files:**
- Modify: `README.md`

**Interfaces:** None — documentation only.

- [ ] **Step 1: Rewrite the Project Structure tree**

File: `README.md`

```markdown
old_string:
## Project Structure

```
text2sql_pipeline/
├── config.py       — All hyperparameters and paths in one place
├── schema.py       — Spider schema loading + graph construction (NetworkX)
├── retrieval.py    — GraphRAG: semantic linking, path tracing, context builder, evaluation
├── generation.py   — Prompt template, SQL cleaning, model loading & inference
├── pipeline.py     — Orchestration, CLI entry point, official Spider evaluation
└── README.md
```

new_string:
## Project Structure

```
src/
├── core/
│   ├── config.py       — All hyperparameters and paths in one place
│   └── schema.py       — Spider schema loading + graph construction (NetworkX)
├── retrieval/
│   ├── retrieval.py    — GraphRAG: semantic linking, path tracing, context builder, evaluation
│   └── baseline.py     — Baseline: table-level retrieval
├── generation/
│   ├── generation.py   — Prompt template, SQL cleaning, model loading & inference
│   └── few_shot.py     — Few-shot example retrieval
└── experiments/
    ├── pipeline.py      — Orchestration, CLI entry point, official Spider evaluation
    ├── sweep.py         — top_k_tables × top_k_columns sweep
    └── ablation.py      — few-shot k ablation
external/spider_eval/    — Official SPIDER evaluation.py + process_sql.py
README.md
```

new_string_note: keep everything else in the README (Setup, Usage, Architecture, Configuration, Results sections) — this step only replaces the tree.
```

- [ ] **Step 2: Update the Setup wget destination**

File: `README.md`

```markdown
old_string:
```bash
pip install -U bitsandbytes>=0.46.1 sentence-transformers transformers networkx pandas torch tqdm

# Download Spider evaluation scripts
wget https://raw.githubusercontent.com/taoyds/spider/master/evaluation.py
wget https://raw.githubusercontent.com/taoyds/spider/master/process_sql.py
```

new_string:
```bash
pip install -U bitsandbytes>=0.46.1 sentence-transformers transformers networkx pandas torch tqdm

# Download Spider evaluation scripts into external/spider_eval/
mkdir -p external/spider_eval && cd external/spider_eval
wget https://raw.githubusercontent.com/taoyds/spider/master/evaluation.py
wget https://raw.githubusercontent.com/taoyds/spider/master/process_sql.py
```
```

- [ ] **Step 3: Update the Usage commands**

File: `README.md`

```markdown
old_string:
### Normal run (GraphRAG)
```bash
python pipeline.py
```

### Ablation: full schema bypass (no retrieval)
```bash
python pipeline.py --full-schema
```

new_string:
### Normal run (GraphRAG)
```bash
python src/experiments/pipeline.py
```

### Ablation: full schema bypass (no retrieval)
```bash
python src/experiments/pipeline.py --full-schema
```
```

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: update README.md paths and commands for new src/ layout"
```

---

### Task 13: Final verification sweep

**Files:** none created or modified — this task only runs checks across everything done in Tasks 1–12.

**Interfaces:** none.

- [ ] **Step 1: py_compile every touched Python file**

Run:
```bash
python -m py_compile src/core/config.py src/core/schema.py src/generation/generation.py src/generation/few_shot.py src/retrieval/retrieval.py src/retrieval/baseline.py src/experiments/sweep.py src/experiments/ablation.py src/experiments/pipeline.py external/spider_eval/evaluation.py external/spider_eval/process_sql.py
```
Expected: no output, exit code 0.

- [ ] **Step 2: Write and run the static import-graph checker**

This dev environment lacks `torch`/`transformers`/`sentence_transformers`/`bitsandbytes`, so the `src.*` imports can't be verified by actually running the files (see Global Constraints). Instead, statically resolve every `from src.* import` in the tree to a real file using only the stdlib (`ast`, `pathlib`).

Write this to the scratchpad directory (NOT into the repo — it's a one-off check, not a deliverable):

File: `C:\Users\TDEV\AppData\Local\Temp\claude\D--Binus-Thesis-Codes-thesis-2803\322a100e-8fad-471f-a5f7-8b8ef6edc5f3\scratchpad\check_imports.py`

```python
import ast
import pathlib
import sys

REPO_ROOT = pathlib.Path(r"D:\Binus\Thesis\Codes\thesis-2803")
SRC_ROOT = REPO_ROOT / "src"

missing = []
checked = 0

for pyfile in sorted(SRC_ROOT.rglob("*.py")):
    tree = ast.parse(pyfile.read_text(encoding="utf-8"), filename=str(pyfile))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("src."):
            checked += 1
            parts = node.module.split(".")
            as_module = REPO_ROOT.joinpath(*parts).with_suffix(".py")
            as_package = REPO_ROOT.joinpath(*parts, "__init__.py")
            if not as_module.exists() and not as_package.exists():
                missing.append((str(pyfile.relative_to(REPO_ROOT)), node.lineno, node.module))

print(f"Checked {checked} 'src.*' import statements across {len(list(SRC_ROOT.rglob('*.py')))} files.")
if missing:
    print("UNRESOLVED IMPORTS:")
    for f, ln, mod in missing:
        print(f"  {f}:{ln}  from {mod} import ...")
    sys.exit(1)
print("All src.* imports resolve to real files.")
```

Run: `python "C:\Users\TDEV\AppData\Local\Temp\claude\D--Binus-Thesis-Codes-thesis-2803\322a100e-8fad-471f-a5f7-8b8ef6edc5f3\scratchpad\check_imports.py"`

Expected: `Checked N 'src.*' import statements across 9 files.` followed by `All src.* imports resolve to real files.`, exit code 0. If it reports unresolved imports, go back and fix the specific `file:line` it names before continuing.

- [ ] **Step 3: Grep for leftover flat internal imports anywhere in the repo (outside evaluation_pipeline/)**

Run:
```bash
grep -rnE "^from (config|schema|retrieval|generation|few_shot|sweep|baseline) import|^[[:space:]]+from (config|schema|retrieval|generation|few_shot|sweep|baseline) import" --include=*.py . | grep -v evaluation_pipeline
```
Expected: no output. Any hit means an import was missed in Tasks 3–8 — fix it and re-run.

- [ ] **Step 4: Grep for stale evaluation.py references**

Run:
```bash
grep -rn '"evaluation.py"' --include=*.py . | grep -v evaluation_pipeline
```
Expected: no output (every reference should now read `"external/spider_eval/evaluation.py"`, which this pattern deliberately excludes by requiring the exact quoted bare filename).

- [ ] **Step 5: Confirm the repo root no longer has the old flat .py files**

Run: `ls *.py 2>&1`
Expected: `ls: cannot access '*.py': No such file or directory` (or equivalent — zero .py files at repo root; `evaluation_pipeline/` and `src/` etc. are subdirectories, not root-level .py files).

- [ ] **Step 6: Clean up the stale root __pycache__**

The root `__pycache__/` contains bytecode compiled from the old flat locations (`schema.cpython-310.pyc`, `process_sql.cpython-310.pyc`) and is stale now that those source files have moved. It isn't git-tracked (confirmed via `git check-ignore`), so this is just local hygiene.

```bash
rm -rf __pycache__
```

- [ ] **Step 7: Report the runtime-verification gap to the user**

This is a reporting step, not a code step. State plainly: static checks (syntax, import-graph resolution, grep sweeps) all pass, but nothing in this refactor has actually been *executed* — this dev environment doesn't have `torch`/`transformers`/`sentence_transformers`/`bitsandbytes` installed, so `python src/experiments/pipeline.py --help` etc. cannot be run here. Recommend the researcher run at least one entry point (e.g. `python src/experiments/sweep.py --sample 0.05`, the cheapest one — no LLM involved) on Kaggle or in an environment with the ML deps installed, to confirm the refactor works end-to-end before relying on it for real experiments.

- [ ] **Step 8: Final commit (if Step 6 produced a change)**

`__pycache__/` was already untracked and gitignored, so `rm -rf __pycache__` produces no git-visible change — there is nothing to commit for this task. Confirm with:

```bash
git status
```

Expected: `nothing to commit, working tree clean` (all prior tasks' commits already captured every change).

---

## Post-plan state

After Task 13, the repo root should contain: `evaluation_pipeline/`, `src/`, `external/`, `notebooks/`, `outputs/`, `docs/`, `CLAUDE.md`, `README.md`, `log.md`, `.git/` — and no loose `.py` files, notebooks, or `plot/` directory at the root.
