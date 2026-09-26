# GraphRAG Text-to-SQL Pipeline

A modular pipeline for the Spider Text-to-SQL benchmark, combining **Graph-based schema retrieval (GraphRAG)** with a **quantized LLM** (Qwen2.5-Coder-7B-Instruct) for SQL generation.

---

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
    ├── pipeline.py      — Orchestration, CLI entry point, official Spider evaluation, model comparison (ablation stage 3)
    ├── sweep.py         — ablation stage 1: top-k fraction sweep (tables × columns), ranked by SLA
    └── ablation.py      — ablation stage 2: few-shot k, ranked by EX
external/spider_eval/    — Official SPIDER evaluation.py + process_sql.py
README.md
```

---

## Setup

```bash
pip install -U bitsandbytes>=0.46.1 sentence-transformers transformers networkx pandas torch tqdm

# Download Spider evaluation scripts into external/spider_eval/
mkdir -p external/spider_eval && cd external/spider_eval
wget https://raw.githubusercontent.com/taoyds/spider/master/evaluation.py
wget https://raw.githubusercontent.com/taoyds/spider/master/process_sql.py
```

---

## Usage

### Normal run (GraphRAG)
```bash
python src/experiments/pipeline.py
```

### Ablation: full schema bypass (no retrieval)
```bash
python src/experiments/pipeline.py --full-schema
```

---

## Architecture

```
Question
   │
   ▼
Semantic Schema Linking  ← BGE-M3 embeddings, n-gram matching
   │
   ▼
Graph Path Tracing       ← NetworkX shortest-path on schema graph
   │
   ▼
Schema Context Builder   ← CREATE TABLE DDL with FK annotations
   │
   ▼
Prompt Builder           ← Structured prompt with value hints
   │
   ▼
LLM (Qwen2.5-Coder 7B)  ← 4-bit quantized, greedy decode
   │
   ▼
SQL Cleaner              ← Strip dialect quirks, fix aliases
   │
   ▼
predictions.txt
   │
   ▼
Official Spider Eval     ← EM + EX via external/spider_eval/evaluation.py
```

---

## Configuration

All settings live in `config.py` (`PipelineConfig` dataclass):

| Parameter | Default | Description |
|---|---|---|
| `embedding_model` | `BAAI/bge-m3` | Sentence encoder for schema linking |
| `llm_model` | `Qwen/Qwen2.5-Coder-7B-Instruct` | SQL generation model |
| `llm_model_frame` | Qwen2.5-Coder 1.5B/3B/7B/14B | Models compared in ablation stage 3 |
| `top_k_tables_pct` / `top_k_columns_pct` | `0.6` / `0.6` | GraphRAG top-k as a fraction of tables in the DB / columns in the candidate tables |
| `baseline_top_k_tables_pct` | `0.6` | Baseline top-k as a fraction of tables in the DB |
| `top_k_frame` | `[0.4, 0.5, 0.6, 0.8]` | Ablation stage 1 grid (`sweep.py`, ranked by SLA) |
| `few_shot_k` | `3` | Few-shot examples, same k for GraphRAG and Baseline |
| `few_shot_frame` | `[0, 1, 3, 5]` | Ablation stage 2 grid (`ablation.py`, ranked by EX) |
| `semantic_similarity_threshold` | `0.35` | Min cosine sim for column detection |
| `max_ngram` | `3` | Max phrase length for query segmentation |
| `max_new_tokens` | `256` | LLM generation budget |
| `use_full_schema_bypass` | `False` | Skip GraphRAG (ablation) |

---

## Results

| Metric | Score |
|---|---|
| Exact Match (EM) | 0.593 |
| Execution Accuracy (EX) | 0.622 |
