"""
Tes lokal src/experiments/trial_steps.py (dua notebook trial, IMPLEMENTATION_DECISIONS.md poin 42), tanpa Spider
dan tanpa model: skema sintetis 1 DB (3 tabel, satu file SQLite untuk value linking), embedder palsu
(bag-of-words), tokenizer palsu (1 token per kata).

    python tests/test_trial_steps.py
"""

import sqlite3
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import networkx as nx
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.core.config import PipelineConfig  # noqa: E402
from src.experiments import result_tables as rt  # noqa: E402
from src.experiments import trial_steps as ts  # noqa: E402

DB = "shop"
TABLES = {
    "customer": [("id", "number", True), ("name", "text", False), ("city", "text", False)],
    "orders": [("id", "number", True), ("customer_id", "number", False), ("amount", "number", False)],
    "product": [("id", "number", True), ("title", "text", False)],
}
QUESTIONS = [
    ("What is the name of each customer?", "SELECT name FROM customer"),
    ("Show the amount of orders for customers in Paris", "SELECT T2.amount FROM customer AS T1 JOIN orders AS T2 ON T1.id = T2.customer_id WHERE T1.city = 'Paris'"),
    ("How many products are there?", "SELECT count(*) FROM product"),
]


class WordTokenizer:
    def encode(self, text):
        return text.split()


class BagOfWords:
    """encode(texts) -> tensor: satu dimensi per kata (hash), dinormalisasi; cukup untuk menjalankan retrieval."""

    def encode(self, texts, convert_to_tensor=True, **kw):
        out = torch.zeros(len(texts), 256)
        for i, t in enumerate(texts):
            for w in str(t).lower().replace("_", " ").split():
                out[i, hash(w.rstrip("s")) % 256] += 1
        return torch.nn.functional.normalize(out + 1e-6, dim=1)


def graphs():
    g = nx.Graph()
    for t, cols in TABLES.items():
        for c, typ, pk in cols:
            g.add_node(f"{DB}.{t}.{c}", type="column", database=DB, table=t, column=c, column_type=typ, is_pk=pk)
        for c, _, pk in cols:
            if not pk:
                g.add_edge(f"{DB}.{t}.{c}", f"{DB}.{t}.id", relation="belongs_to_pk")
    g.add_edge(f"{DB}.orders.customer_id", f"{DB}.customer.id", relation="foreign_key")
    tg = nx.Graph()
    for t, cols in TABLES.items():
        tg.add_node(f"{DB}.{t}", type="table", database=DB, table=t,
                    columns=[{"Column": c, "Type": typ, "is_pk": "Yes" if pk else "No"} for c, typ, pk in cols])
    tg.add_edge(f"{DB}.orders", f"{DB}.customer", relation="foreign_key",
                fks=[{"from_table": "orders", "from_col": "customer_id", "to_col": "id"}])
    return g, tg


def make_data(tmp: Path) -> ts.TrialData:
    db_dir = tmp / "database" / DB
    db_dir.mkdir(parents=True)
    con = sqlite3.connect(db_dir / f"{DB}.sqlite")
    con.execute("CREATE TABLE customer (id INTEGER PRIMARY KEY, name TEXT, city TEXT)")
    con.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY, customer_id INTEGER, amount REAL)")
    con.execute("CREATE TABLE product (id INTEGER PRIMARY KEY, title TEXT)")
    con.executemany("INSERT INTO customer VALUES (?, ?, ?)", [(1, "Ann", "Paris"), (2, "Bob", "Rome")])
    con.commit()
    con.close()
    cfg = PipelineConfig(data_path=tmp, top_k_tables_pct=0, top_k_columns_pct=0, table_similarity_threshold=0.1,
                         semantic_similarity_threshold=0.1)
    db_schema = {DB: [f"{t}.{c}" for t, cols in TABLES.items() for c, _, _ in cols]}
    dev = [{"db_id": DB, "question": q, "query": sql} for q, sql in QUESTIONS]
    # Parser Spider butuh data NLTK yang tidak ada di mesin lokal: gold schema ditulis tangan untuk tes ini
    gold = {QUESTIONS[0][1]: {"customer.name"}, QUESTIONS[1][1]: {"orders.amount", "customer.id", "orders.customer_id", "customer.city"},
            QUESTIONS[2][1]: set()}
    original, ts.extract_ground_truth_schema = ts.extract_ground_truth_schema, lambda sql, db, schema: gold[sql]
    try:
        samples, n_failed, n_empty = ts.prepare_samples(dev, [0, 1, 2], db_schema)
    finally:
        ts.extract_ground_truth_schema = original
    g, tg = graphs()
    data = ts.TrialData(cfg=cfg, schema_df=None, graph=g, tbl_graph=tg, db_schema=db_schema, tokenizer=WordTokenizer(),
                        dev_data=dev, indices=[0, 1, 2], samples=samples, n_failed=n_failed, n_empty=n_empty)
    data = ts.attach_databases(data)
    data.embed_model = BagOfWords()
    data.embedder = ts.CachedEmbedder(data.embed_model)
    return data


def test_settings():
    options = {"stopword": ["none", "edge"], "value_link": ["off", "text", "text + numbers"], "traversal": ["full", "fk_only"],
               "no_pk": ["not connected", "connected"], "nama_embed": ["original", "natural"]}
    combos = ts.all_combinations(options)
    assert len(combos) == 48 and len({ts.describe(c) for c in combos}) == 48
    assert combos[0] == {"stopword": "none", "value_link": "off", "traversal": "full", "no_pk": "not connected",
                         "nama_embed": "original"}
    try:
        ts.all_combinations({"traversal": ["fk-only"]})
    except ValueError as exc:
        assert "fk_only" in str(exc)
    else:
        raise AssertionError("nilai salah harus ditolak sebelum run")

    cfg = PipelineConfig()
    final = ts.settings_of(cfg)
    assert ts.switches_to_settings(True, True, True, True, True, True) == final          # default saklar = config.py
    assert ts.settings_of(replace(cfg, **ts.knobs_of(final))) == final
    alt = ts.switches_to_settings(False, True, True, True, True, True)
    assert alt["stopword"] == "none" and alt["value_link"] == "text + numbers"
    assert ts.switches_to_settings(True, False, True, False, False, False)["value_link"] == "off"   # angka butuh value linking

    name, pipeline, c = ts.one_stage_trial(cfg, 0.55)
    assert (name, pipeline, c.baseline_embed_columns, c.table_similarity_threshold) == ("1-stage | T0.55", "baseline", False, 0.55)


def test_run_trials_and_tables():
    with tempfile.TemporaryDirectory() as d:
        data = make_data(Path(d))
        assert data.n_empty == 1 and len(data.samples) == 3                       # count(*) = gold kosong, tetap dipakai
        cfg = data.cfg
        combos = ts.all_combinations({"value_link": ["off", "text"], "traversal": ["full", "fk_only"]})
        trials = [ts.two_stage_trial(cfg, s, 0.1, 0.1) for s in combos]
        cache = {}
        recap, results = ts.run_trials(data, trials, cache)
        assert len(recap) == 4 and set(recap["pipeline"]) == {"graphrag"} and len(cache) == 4
        assert recap["n"].eq(3).all() and recap["col_R_ne"].between(0, 1).all()
        # value linking "Paris" -> customer.city ikut sebagai anchor
        linked = results[ts.two_stage_trial(cfg, {"value_link": "text", "traversal": "fk_only"}, 0.1, 0.1)[0]]
        assert "customer.city" in linked[1]["pred"]
        # sel dijalankan ulang: semua dari cache, hasil sama
        again, _ = ts.run_trials(data, trials, cache)
        assert len(cache) == 4 and again.equals(recap)

        with_ex = recap.assign(ex=[0.2, 0.4, 0.6, 0.8])
        # Grid: 2-Stage di dua K dan 1-Stage, 1-Stage di atas pada T yang sama
        grid_trials = [ts.two_stage_trial(cfg, ts.settings_of(cfg), 0.1, k) for k in (0.1, 0.3)] + [ts.one_stage_trial(cfg, 0.1)]
        grid, grid_results = ts.run_trials(data, grid_trials, cache)
        vs = ts.compare_with_one_stage(grid.assign(ex=[0.5, 0.4, 0.3]))
        assert len(vs) == 2 and abs(vs["ex_delta"].iloc[0] - 0.2) < 1e-9
        assert (vs["cols_B"] == grid.loc["1-stage | T0.10", "cols_sent"]).all()
        assert len(ts.compare_trials(grid_results, grid_trials[0][0], grid_trials[1][0])) >= 0

        tables = rt.trial_tables(with_ex.reset_index(), grid.assign(ex=[0.5, 0.4, 0.3]).reset_index())
        assert list(tables["grid"]["Pipeline"]) == ["1-stage", "2-stage", "2-stage"]
        assert "Trial" not in tables["steps"].columns and len(tables["steps"]) == 4 and set(tables) == {"steps", "grid"}
        for t in (tables["steps"], tables["grid"]):      # SLA non-empty gold tepat di samping EX
            cols = list(t.columns)
            i = cols.index("EX (zero-shot)")
            assert cols[i + 1:i + 3] == ["Column P (non-empty gold)", "Column R (non-empty gold)"], cols[i:i + 3]
        for t in tables.values():
            rt.formatted(t)

        # Excel dari dua CSV notebook: tiga sheet trial, tabel di A1, tanpa Info
        from openpyxl import load_workbook

        from src.experiments.export_excel import build_workbook

        out = Path(d) / "tables"
        ts.save_recap(with_ex, out, ts.PREPROCESSING_RECAP)
        ts.save_recap(grid, out, ts.THRESHOLDS_RECAP)
        wb = load_workbook(build_workbook(Path(d) / "none", out, out / "x.xlsx", skip_missing=True))
        assert wb.sheetnames == ["Trial Preprocessing", "Trial Thresholds"], wb.sheetnames
        assert wb["Trial Preprocessing"].freeze_panes == "F2" and wb["Trial Thresholds"].freeze_panes == "D2"
        for ws in wb:
            assert next(iter(ws.tables.values())).ref.startswith("A1:")
        assert [c.value for c in wb["Trial Thresholds"]["A"][1:4]] == ["1-stage", "2-stage", "2-stage"]
        wb.close()


def test_score_ex_without_gpu():
    if torch.cuda.is_available():
        return
    recap = pd.DataFrame({"pipeline": ["graphrag"]}, index=["x"])
    out, llm = ts.score_ex(recap, {"x": []}, PipelineConfig(), WordTokenizer(), Path("unused.jsonl"))
    assert "ex" not in out.columns and llm is None


def main():
    test_settings()
    test_run_trials_and_tables()
    test_score_ex_without_gpu()
    print("OK: trial steps tests passed")


if __name__ == "__main__":
    main()
