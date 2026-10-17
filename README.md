# GraphRAG Text-to-SQL Pipeline

A modular pipeline for the Spider Text-to-SQL benchmark, combining **Graph-based schema retrieval (GraphRAG)** with a **quantized LLM** (Qwen2.5-Coder-7B-Instruct) for SQL generation.

---

## Project Structure

```
In Progress of finalization of the structure 🩷
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
```
Building it now hehehehehhee
---
```
## Architecture (Still improving, but this one below is what the general architecture is now)

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


---

## Results
We are still experimenting!
