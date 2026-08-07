# Design: Restructure text2sql pipeline into src/, external/, notebooks/, outputs/

Date: 2026-08-07

## Context

The GraphRAG text-to-SQL pipeline code (`config.py`, `schema.py`, `retrieval.py`,
`baseline.py`, `few_shot.py`, `generation.py`, `pipeline.py`, `sweep.py`,
`ablation.py`) plus the Spider official eval scripts (`evaluation.py`,
`process_sql.py`) currently sit flat at the repo root, as siblings to
`evaluation_pipeline/` — a separately scaffolded evaluation/metrics project that
already follows a tidy `src/{utils,metrics,dimensions}/`, `external/spider_eval/`,
`data/`, `outputs/tables/` layout.

Goal: reorganize the root-level pipeline code to match that same tidy, modular
style — without introducing a `text2sql_pipeline/` container folder (rejected;
`src/`, `external/`, `notebooks/`, `outputs/` go directly at repo root, as
siblings to `evaluation_pipeline/`).

## Target folder structure

```
thesis-2803/
├── evaluation_pipeline/        (untouched)
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py           (was ./config.py)
│   │   └── schema.py           (was ./schema.py)
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── retrieval.py        (was ./retrieval.py — GraphRAG)
│   │   └── baseline.py         (was ./baseline.py — table-level baseline)
│   ├── generation/
│   │   ├── __init__.py
│   │   ├── generation.py       (was ./generation.py)
│   │   └── few_shot.py         (was ./few_shot.py)
│   └── experiments/
│       ├── __init__.py
│       ├── pipeline.py         (was ./pipeline.py — CLI entry point)
│       ├── sweep.py            (was ./sweep.py — CLI entry point)
│       └── ablation.py         (was ./ablation.py — CLI entry point)
├── external/
│   └── spider_eval/
│       ├── evaluation.py       (was ./evaluation.py)
│       └── process_sql.py      (was ./process_sql.py)
├── notebooks/
│   ├── graphrag-text-to-sql-eda.ipynb
│   ├── Text-To-SQL_Graph_Representation.ipynb
│   └── compiled/
│       └── graphrag_text2sql.ipynb
├── outputs/
│   ├── plots/                  (16 pngs, was ./plot/)
│   ├── predictions/            (predictions.txt, baseline_predictions.txt,
│   │                             ablation_*_predictions_k*.txt,
│   │                             ablation_*_prompts_k*.jsonl)
│   ├── logs/                   (baseline_log.txt)
│   └── tables/                 (baseline_results.csv, sweep_results.csv,
│                                 ablation_results.csv, comparison_report.txt)
├── CLAUDE.md                   (stays at repo root — content updated)
├── README.md                   (content updated)
└── log.md                      (untouched)
```

`evaluation_pipeline/` itself is not touched by this work — it's a separate,
already-scaffolded sub-project.

## Import strategy

All internal imports become absolute, rooted at `src`:

- `from config import PipelineConfig` → `from src.core.config import PipelineConfig`
- `from schema import build_schema_graph, load_spider_schema` → `from src.core.schema import ...`
- `from retrieval import (...)` → `from src.retrieval.retrieval import (...)`
- `from generation import ...` → `from src.generation.generation import ...`
- `from few_shot import ...` → `from src.generation.few_shot import ...`
- `from sweep import run_sweep_and_get_best` → `from src.experiments.sweep import run_sweep_and_get_best`

Empty `__init__.py` files are added to `src/`, `src/core/`, `src/retrieval/`,
`src/generation/`, `src/experiments/` to make each a proper package (matches
`evaluation_pipeline`'s convention).

### Entry-point bootstrap

`pipeline.py`, `sweep.py`, `ablation.py`, `baseline.py` are documented CLI entry
points (`CLAUDE.md`'s CLI Quick Reference: `python pipeline.py`, `python sweep.py
--sample 0.2`, etc.), and are also invoked this way from Kaggle notebooks. Once
they live two levels deep (`src/experiments/pipeline.py`,
`src/retrieval/baseline.py`), running them directly via `python
src/experiments/pipeline.py` won't have the repo root on `sys.path`, so `from
src.core...`-style imports fail.

Fix: add a small `sys.path` bootstrap at the top of just these 4 entry-point
files, before their `src.*` imports:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
```

This keeps `python src/experiments/pipeline.py` (and `python
src/retrieval/baseline.py`) working exactly as today, with no switch to `python
-m ...` module invocation and no new packaging (`pyproject.toml` / `pip install
-e`) required. Non-entry modules (`config.py`, `schema.py`, `retrieval.py`,
`generation.py`, `few_shot.py`) don't need the bootstrap — they're only ever
imported, never run directly.

## Spider evaluation subprocess calls

`pipeline.py`, `baseline.py`, `ablation.py` currently do:

```python
evaluator = Path("evaluation.py")
...
subprocess.run(["python", "evaluation.py"] + args, check=False)
```

assuming cwd == repo root. These become:

```python
evaluator = Path("external/spider_eval/evaluation.py")
...
subprocess.run(["python", "external/spider_eval/evaluation.py"] + args, check=False)
```

(Path resolved relative to the same repo-root constant used by the bootstrap.)
The "not found" warning messages that tell the researcher to `wget` the Spider
scripts get updated to say they belong in `external/spider_eval/`.

## Output path defaults

Hardcoded default output paths move under `outputs/`:

| File | Current default | New default |
|---|---|---|
| `config.py` | `predictions_file = Path("predictions.txt")` | `Path("outputs/predictions/predictions.txt")` |
| `baseline.py` | `pred_path = Path("baseline_predictions.txt")` | `Path("outputs/predictions/baseline_predictions.txt")` |
| `baseline.py` | `log_path = Path("baseline_log.txt")` | `Path("outputs/logs/baseline_log.txt")` |
| `baseline.py` | `csv_path = Path("baseline_results.csv")` | `Path("outputs/tables/baseline_results.csv")` |
| `baseline.py` | `report_path = Path("comparison_report.txt")` | `Path("outputs/tables/comparison_report.txt")` |
| `pipeline.py` | `g_pred_path = Path("predictions.txt")` | `Path("outputs/predictions/predictions.txt")` |
| `pipeline.py` | `b_pred_path = Path("baseline_predictions.txt")` | `Path("outputs/predictions/baseline_predictions.txt")` |
| `pipeline.py` | `Path("baseline_log.txt")` / `Path("baseline_results.csv")` | `outputs/logs/` / `outputs/tables/` |
| `pipeline.py` | `Path("comparison_report.txt")` | `Path("outputs/tables/comparison_report.txt")` |
| `sweep.py` | `Path("sweep_results.csv")` (x2) | `Path("outputs/tables/sweep_results.csv")` |
| `ablation.py` | `output_dir = Path(".")` | `Path("outputs/predictions")` |
| `ablation.py` | `Path("ablation_results.csv")` | `Path("outputs/tables/ablation_results.csv")` |

`outputs/plots/`, `outputs/predictions/`, `outputs/logs/`, `outputs/tables/` each
get a `.gitkeep` so git tracks the empty directories, matching
`evaluation_pipeline`'s convention (`outputs/tables/.gitkeep`).

Scope note: this only repoints the literal default path strings. It does not
add new `PipelineConfig` fields for these paths or otherwise restructure how
they're threaded through function signatures/argparse — that's a separate,
larger refactor not requested here.

## Docs updates

- `CLAUDE.md`: "Struktur File" section rewritten to the new tree; "Paths
  (Kaggle)" section's mention of `predictions.txt` etc. updated; "CLI Quick
  Reference" commands updated (e.g. `python src/experiments/pipeline.py
  --skip-sweep`, `python src/retrieval/baseline.py --sample 1.0`, `python
  src/experiments/ablation.py --k-values 0 1 3 5`, `python
  src/experiments/sweep.py --sample 0.2`).
- `README.md`: "Project Structure" tree, "Setup" (wget destination now
  `external/spider_eval/`), "Usage" commands updated to match.

`CLAUDE.md` stays physically at the repo root (not moved into any subfolder) so
it continues to be auto-loaded as the project's instructions file.

## Migration mechanics

1. `git mv` each file to its new location (preserves history).
2. Add `__init__.py` files and `.gitkeep` files.
3. Update imports, subprocess paths, and output-path defaults per above.
4. Update `CLAUDE.md` and `README.md`.

## Verification plan

No test suite exists for this pipeline. Verification consists of:

1. `python -m py_compile` on every moved/edited `.py` file — catches syntax and
   import-path typos.
2. `python src/experiments/pipeline.py --help`, `python
   src/experiments/sweep.py --help`, `python src/experiments/ablation.py
   --help`, `python src/retrieval/baseline.py --help` — confirms each entry
   point still boots, resolves its imports, and reaches argparse without
   hitting the Kaggle-only `data_path` (any failure must occur logically after
   arg parsing, not at import time).
3. Repo-wide grep for leftover flat imports (`from config import`, `from schema
   import`, `from retrieval import`, `from generation import`, `from few_shot
   import`, `from sweep import`, `from baseline import`) to confirm nothing was
   missed.
4. Repo-wide grep for the string `"evaluation.py"` to confirm every reference
   points at `external/spider_eval/evaluation.py`.

## Out of scope

- Any change to `evaluation_pipeline/`.
- Implementing the `NotImplementedError` stubs in `evaluation_pipeline/src/`.
- Adding `pyproject.toml`/packaging or switching entry points to `python -m`
  invocation.
- Threading output paths through `PipelineConfig` as new fields.
- Changing any pipeline logic, prompt format, or hyperparameters.
