# CLAUDE.md — GraphRAG Text-to-SQL Pipeline

Skripsi: **Graph-Based Retrieval-Augmented Generation untuk Text-to-SQL**
Dataset: Spider benchmark
Environment: Kaggle Notebook (GPU T4, memory terbatas)

---


## Konteks Penelitian

Penelitian ini menganalisis trade-off antara performa SQL generation dan efisiensi token consumption. Ada dua pendekatan utama yang dibandingkan:

- **Baseline RAG** — table-level retrieval: retrieve top-k tabel, kirim semua kolom dari tabel terpilih ke LLM
- **GraphRAG** — two-stage column-level retrieval + graph traversal + path pruning: kirim hanya kolom yang relevan

Token efficiency GraphRAG berasal dari selektivitas retrieval — bukan dari seluruh database, hanya k tabel → k kolom → pruned columns yang masuk ke prompt.

---

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

---

## Arsitektur Pipeline (GraphRAG)

```
Pertanyaan user (bahasa alami)
        │
        ▼
[1] Two-Stage Semantic Retrieval  ← BGE-M3 embeddings, n-gram matching
        │  Stage 1: top-k table selection (cosine similarity)
        │  Stage 2: top-k column selection dari tabel kandidat
        ▼
[2] Graph Traversal               ← NetworkX shortest-path pada schema graph
        │  Input: kolom hasil retrieval sebagai anchor nodes
        │  Output: jalur konektivitas antar tabel (untuk JOIN)
        ▼
[3] Path Pruning                  ← buang intermediate nodes yang bukan PK/FK/direct hit
        ▼
[4] Schema Context Builder        ← CREATE TABLE DDL dengan FK annotations
        ▼
[5] Few-Shot Retrieval            ← cosine similarity terhadap training set embeddings
        ▼
[6] Prompt Builder                ← Hybrid: ASP + CRP + TRP + ODP
        │  ### Task
        │  ### Database Schema   (DDL hasil retrieval)
        │  ### Examples          (few-shot, jika k > 0)
        │  ### Question
        │  ### Instructions
        │  ### Answer\n```sql
        ▼
[7] LLM Inference                 ← Qwen2.5-Coder-7B-Instruct, 4-bit NF4 quantized
        ▼
[8] SQL Cleaner                   ← strip markdown, fix dialect quirks, normalize aliases
        ▼
predictions.txt → Spider Official Eval (EM + EX)
```

---

## Arsitektur Pipeline (Baseline)

```
Pertanyaan user
        │
        ▼
[1] Table-Level Semantic Linking  ← single query embedding vs table embeddings
        │  (bukan n-gram, bukan two-stage)
        │  Embed format: "table <name> columns <col1> <col2> ..."
        ▼
[2] Table Path Tracing            ← shortest path antar tabel via FK edges
        ▼
[3] Schema Context Builder        ← CREATE TABLE DDL, semua kolom dari tabel terpilih
        │  (TIDAK ada pruning — seluruh kolom tabel ikut masuk)
        ▼
[4] Prompt Builder + LLM + SQL Cleaner  (sama dengan GraphRAG)
        ▼
baseline_predictions.txt → Spider Official Eval
```

---

## Schema Graph

Dibangun dari `tables.json` Spider via `schema.py`:

- **Node** = satu kolom, identifier unik: `{database}.{table}.{column}`
- **Node attributes**: database, table, column, column_type, is_pk
- **Edge type 1** `belongs_to_pk` — kolom non-PK → PK tabel yang sama
- **Edge type 2** `foreign_key` — kolom FK → kolom PK di tabel lain
- Total: 8.747 nodes, 2.854 edges

Graph baseline (`baseline.py`) berbeda — node-nya per tabel, bukan per kolom.

---

## Model

| Komponen | Model | Keterangan |
|---|---|---|
| Embedding | `BAAI/bge-m3` | Dipakai untuk schema linking DAN few-shot retrieval |
| LLM | `Qwen/Qwen2.5-Coder-7B-Instruct` | 4-bit NF4 quantized via BitsAndBytes |

Embedding model **sama** untuk dua keperluan berbeda — schema linking dan few-shot retrieval. Jangan ganti salah satu tanpa ganti keduanya.

---

## Konfigurasi (`config.py`)

Semua parameter ada di `PipelineConfig` dataclass. **Jangan hardcode nilai di file lain** — selalu referensikan dari `cfg`.

| Parameter | Nilai saat ini | Status | Keterangan |
|---|---|---|---|
| `embedding_model` | `BAAI/bge-m3` | Final | |
| `llm_model` | `Qwen/Qwen2.5-Coder-7B-Instruct` | Final | |
| `top_k_tables` | `3` | TBD | Ditentukan oleh sweep.py |
| `top_k_columns` | `5` | TBD | Ditentukan oleh sweep.py |
| `semantic_similarity_threshold` | `0.35` | TBD | Fallback jika top_k_columns=0 |
| `few_shot_k` | `3` | TBD | Hasil ablation study |
| `few_shot_same_db_first` | `True` | Final | Prioritaskan contoh dari DB yang sama |
| `max_ngram` | `3` | Final | Segmentasi query sampai trigram |
| `max_new_tokens` | `200` | Final | Budget generasi LLM |
| `temperature` | `0.0` | Final | Greedy decoding, deterministik |
| `load_in_4bit` | `True` | Final | Kuantisasi NF4 |

Parameter bertanda **TBD** akan diupdate setelah sweep dan ablation selesai.

---

## Ablation Study

Cross design 2×4 — jalankan terpisah 2x untuk menghindari OOM:

**Run 1 — Baseline RAG:**
```bash
python ablation.py --mode baseline --k-values 0 1 3 5 --sample 1.0
```

**Run 2 — GraphRAG:**
```bash
python ablation.py --mode graphrag --k-values 0 1 3 5 --sample 1.0
```

Output per run:
- `ablation_{mode}_predictions_k{k}.txt` — SQL predictions → masuk ke Spider `evaluation.py` untuk EM/EX
- `ablation_{mode}_prompts_k{k}.jsonl` — per sample: i, db_id, question, tokens_in, tokens_out, token_consumption, prompt, pred_sql
- `ablation_results.csv` — avg_recall, avg_precision, avg_prompt_tokens (T_in), avg_output_tokens (T_out), avg_token_consumption (T) per mode×k → dasar perhitungan TEP

| | k=0 | k=1 | k=3 | k=5 |
|---|---|---|---|---|
| Baseline RAG | ✓ | ✓ | ✓ | ✓ |
| GraphRAG | ✓ | ✓ | ✓ | ✓ |

Tujuan ablation: cari **elbow point** — nilai k di mana penambahan few-shot example sudah tidak signifikan meningkatkan performa relatif terhadap token cost. Hipotesis: lonjakan terbesar ada di k=0→1, setelah itu diminishing returns.

---

## Evaluasi Metrik

| Metrik | Tool | Keterangan |
|---|---|---|
| Exact Set Match (ESM) | Spider `evaluation.py --etype match` | |
| Execution Accuracy (EX) | Spider `evaluation.py --etype exec` | |
| Component Match (CM) | Spider `evaluation.py` | |
| Token Consumption | `avg_token_consumption` di `ablation_results.csv` | T = T_in + α×T_out. T_in = prompt tokens, T_out = generated SQL tokens, α = `token_output_weight` di config.py (default 1.0 untuk local model). Diukur inline per sample setelah generate_sql. |
| TEP (Token Elasticity of Performance) | Custom metric, hitung post-hoc | TEP_G = (ΔEX_G/EX_B) / (ΔT_G/T_B) — elastisitas performa GraphRAG relatif terhadap konsumsi token vs Baseline. ΔEX_G = EX_G − EX_B, ΔT_G = T_G − T_B. Hitung dari `ablation_results.csv` + EX dari Spider eval. |
| QVT (Query Variance Testing) | Custom metric | Stabilitas output terhadap variasi pertanyaan |

Spider evaluation scripts harus didownload manual:
```bash
wget https://raw.githubusercontent.com/taoyds/spider/master/evaluation.py
wget https://raw.githubusercontent.com/taoyds/spider/master/process_sql.py
```

---

## Paths (Kaggle)

```python
data_path = Path("/kaggle/input/datasets/alrette/spiderdataset/spider_data")
# tables.json  → data_path / "tables.json"
# dev.json     → data_path / "dev.json"
# train.json   → data_path / "train_spider.json"
# database/    → data_path / "database"
# gold SQL     → data_path / "dev_gold.sql"
```

---

## Data Storage

**Tidak ada vector database eksternal.** Semua embedding disimpan sebagai `torch.Tensor` di memori Python selama runtime:

- `SchemaIndex` — embedding tabel dan kolom per database, di-cache di `schema_cache` dict
- `FewShotIndex` — embedding seluruh training set (~7000 pertanyaan), built once at startup
- Semua hilang saat session selesai — ini by design untuk research pipeline

---

## Hal yang Jangan Diubah Tanpa Diskusi

- Format prompt di `generation.py` — sudah di-tune, perubahan kecil berdampak besar ke output LLM
- Struktur graph di `schema.py` — node identifier `{db}.{table}.{col}` dipakai di banyak tempat
- `few_shot_same_db_first=True` — sudah jadi keputusan desain final
- Greedy decoding (`temperature=0.0`, `do_sample=False`) — perlu deterministik untuk reprodusibilitas

## Hal yang Masih TBD (update setelah eksperimen)

- Nilai final `top_k_tables`, `top_k_columns`, `semantic_similarity_threshold` → tunggu hasil `sweep.py`
- Nilai final `few_shot_k` → tunggu hasil ablation study
- Definisi final baseline: saat ini table-level retrieval, mungkin diubah ke full-schema bypass — diskusikan dulu dengan kelompok
- `max_new_tokens`: di `config.py` = 200, di beberapa tempat tertulis 256 — perlu diseragamkan

---

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
