# GraphRAG Text-to-SQL Pipeline

A modular pipeline for the Spider Text-to-SQL benchmark, combining **Graph-based schema retrieval (GraphRAG)** with a **quantized LLM** (Qwen2.5-Coder-7B-Instruct) for SQL generation.

---

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

---

## Setup

```bash
pip install -U bitsandbytes>=0.46.1 sentence-transformers transformers networkx pandas torch tqdm

# Download Spider evaluation scripts
wget https://raw.githubusercontent.com/taoyds/spider/master/evaluation.py
wget https://raw.githubusercontent.com/taoyds/spider/master/process_sql.py
```

---

## Usage

### Normal run (GraphRAG)
```bash
python pipeline.py
```

### Ablation: full schema bypass (no retrieval)
```bash
python pipeline.py --full-schema
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
Official Spider Eval     ← EM + EX via evaluation.py
```

---

## Configuration

All settings live in `config.py` (`PipelineConfig` dataclass):

| Parameter | Default | Description |
|---|---|---|
| `embedding_model` | `BAAI/bge-m3` | Sentence encoder for schema linking |
| `llm_model` | `Qwen/Qwen2.5-Coder-7B-Instruct` | SQL generation model |
| `semantic_similarity_threshold` | `0.35` | Min cosine sim for column detection |
| `max_ngram` | `3` | Max phrase length for query segmentation |
| `max_new_tokens` | `200` | LLM generation budget |
| `use_full_schema_bypass` | `False` | Skip GraphRAG (ablation) |

---

## Results

| Metric | Score |
|---|---|
| Exact Match (EM) | 0.593 |
| Execution Accuracy (EX) | 0.622 |
