"""
Tes lokal tracing trial threshold (IMPLEMENTATION_DECISIONS.md poin 35), tanpa Spider dan tanpa model:
persentase tabel/kolom per DB, reduksi terhadap DDL penuh, cache EX, dan tabel Excel-nya.

    python tests/test_trial_tracing.py
"""

import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiments import result_tables as rt
from src.experiments.trial_ex import ExScorer
from src.metrics.token_reduction import MEDIAN_COLUMNS, full_db_summary, median_summary


def test_full_db_summary():
    # DB kecil 4 tabel / 20 kolom, DB besar 10 tabel / 100 kolom
    rows = [
        {"tables_sent": 2, "tables_db": 4, "cols_sent": 5, "cols_db": 20, "schema_tokens": 50, "schema_tokens_db": 200,
         "prompt_tokens_k0": 150, "prompt_tokens_k0_db": 300},
        {"tables_sent": 2, "tables_db": 10, "cols_sent": 10, "cols_db": 100, "schema_tokens": 100, "schema_tokens_db": 1000,
         "prompt_tokens_k0": 200, "prompt_tokens_k0_db": 1100},
    ]
    s = full_db_summary(rows)
    assert abs(s["tables_pct"] - (0.50 + 0.20) / 2) < 1e-9
    assert abs(s["cols_pct"] - (0.25 + 0.10) / 2) < 1e-9
    assert abs(s["schema_red_db"] - (1 - 150 / 1200)) < 1e-9
    assert abs(s["prompt_red_db"] - (1 - 350 / 1400)) < 1e-9
    assert s["tables_sent"] == 2 and s["tables_db"] == 7 and s["schema_tokens_db"] == 600
    # bobot per query vs rasio total: sengaja beda (1 - 0.175 = 0.825, 1 - 15/120 = 0.875)
    assert abs((1 - s["cols_pct"]) - (1 - 15 / 120)) > 0.04


def test_median_summary():
    # Sembilan query 10 kolom, satu query fallback 100 kolom: rata-rata 19, median tetap 10
    rows = [{"tables_sent": 2, "tables_db": 4, "cols_sent": 10, "cols_db": 20, "schema_tokens": 100, "prompt_tokens_k0": 200}
            for _ in range(9)]
    rows.append({"tables_sent": 5, "tables_db": 5, "cols_sent": 100, "cols_db": 100, "schema_tokens": 1000, "prompt_tokens_k0": 1100})
    s = median_summary(rows)
    assert set(s) == set(MEDIAN_COLUMNS)
    assert s["cols_sent_med"] == 10 and s["tables_pct_med"] == 0.5 and s["cols_pct_med"] == 0.5 and s["prompt_tokens_k0_med"] == 200

    two = median_summary([{**rows[0], "cols_sent": 5}, {**rows[0], "cols_sent": 10}])
    assert two["cols_sent_med"] == 7.5 and two["cols_pct_med"] == 0.375   # 5/20 dan 10/20
    assert median_summary(rows[:1])["cols_sent_med"] == 10                # satu query tidak boleh crash


def test_ex_scorer():
    calls = []

    def generate(prompt):
        calls.append(prompt)
        if "boom" in prompt:
            raise RuntimeError("CUDA out of memory")
        return "SELECT 1", 7

    def evaluate(sql, gold, db):
        return SimpleNamespace(ex=int(gold == "right"), esm=0)

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "sub" / "cache.jsonl"
        s = ExScorer(generate, evaluate, path, "model-a")
        assert s.score("p1", "right", "db") == 1 and s.score("p1", "right", "db") == 1
        assert s.score("p2", "wrong", "db") == 0
        assert len(calls) == 2 and (s.generated, s.reused) == (2, 1)

        # gagal dihitung salah, tidak disimpan
        assert s.score("boom", "right", "db") == 0 and s.failed == 1
        assert s.score("boom", "right", "db") == 0 and len(calls) == 4

        # melanjutkan dari file: tanpa generasi ulang; model lain tidak memakai hasilnya
        calls.clear()
        again = ExScorer(generate, evaluate, path, "model-a")
        assert again.score("p1", "right", "db") == 1 and calls == []
        other = ExScorer(generate, evaluate, path, "model-b")
        assert other.score("p1", "right", "db") == 1 and calls == ["p1"]


def recap_row(trial, t, k, ex=None):
    row = {"trial": trial, "table_threshold": t, "column_threshold": k, "col_R_ne": 0.9, "cols_sent": 5.0, "col_reduction": 0.5,
           "schema_tokens": 50.0, "prompt_tokens_k0": 150.0, "cols_db": 10.0, "tables_sent": 2.0, "tables_db": 4.0,
           "tables_pct": 0.5, "cols_pct": 0.5, "schema_tokens_db": 100.0, "schema_red_db": 0.5,
           "prompt_tokens_k0_db": 200.0, "prompt_red_db": 0.25, "n": 2}
    if ex is not None:
        row["ex"] = ex
    return row


def test_tables():
    recap = pd.DataFrame([
        recap_row("G1 threshold saja | T0.50/K0.70", 0.5, 0.7, 0.45),
        recap_row("G6 + sambung no-PK | T0.50/K0.70", 0.5, 0.7, 0.50),
        recap_row("G7 + nama natural | T0.50/K0.75", 0.5, 0.75, 0.52),
        recap_row("G7 + nama natural | T0.50/K0.70", 0.5, 0.7, 0.55),
        recap_row("G8 tanpa stopword + angka | T0.50/K0.70", 0.5, 0.7, 0.58),
        recap_row("G8 tanpa stopword + angka | T0.50/K0.75", 0.5, 0.75, 0.53),
        recap_row("GraphRAG 1-Stage | T0.50", 0.5, None, 0.50),
    ])
    tables = rt.sla_trial_tables(recap)
    assert list(tables) == ["steps", "grid"]
    for t in tables.values():
        rt.formatted(t)

    # Tanpa kolom Trial di kedua sheet, baris dikenali dari pengaturan dan pipeline
    assert "Trial" not in tables["steps"].columns and "Trial" not in tables["grid"].columns
    # Langkah nama natural ikut di sheet preprocessing, hanya pada pasangan (T, K) yang sama, dan tetap ada di grid
    assert list(tables["steps"]["EX (zero-shot)"]) == [0.45, 0.50, 0.55, 0.58]   # G8 di sheet preprocessing, tidak di grid
    assert set(tables["steps"]["Column threshold (K)"]) == {0.7}
    # Grid: G7 dan G8 di tiap (T, K), G7 lebih dulu pada (T, K) yang sama
    assert list(tables["grid"]["EX (zero-shot)"]) == [0.50, 0.55, 0.58, 0.52, 0.53]
    assert rt.sla_trial_tables(recap[recap["trial"].str.startswith("G7")])["steps"] is None   # tanpa langkah lain, bukan riwayat
    assert list(tables["grid"]["Pipeline"]) == ["1-stage"] + ["2-stage"] * 4
    assert list(tables["grid"]["Column threshold (K)"].iloc[1:]) == [0.7, 0.7, 0.75, 0.75]
    assert list(tables["grid"].columns[:3]) == ["Pipeline", "Table threshold (T)", "Column threshold (K)"]
    for t in tables.values():
        for header in ("EX (zero-shot)", "Avg columns in prompt", "Avg tables retrieved, % of DB", "Avg columns retrieved, % of DB",
                       "Column reduction vs full DDL", "Schema-token reduction vs full DDL", "Prompt-token reduction vs full DDL"):
            assert header in t.columns, header

    with_median = recap.assign(cols_sent_med=4.0, cols_pct_med=0.4, cols_sent_p75=6.0)     # CSV lama boleh masih punya kuartil
    grid = rt.sla_trial_tables(with_median)["grid"]
    assert grid["Median columns in prompt"].iloc[0] == 4
    assert grid.attrs["formats"]["Median columns retrieved, % of DB"] == "pct"
    cols = list(grid.columns)
    assert cols.index("Avg columns in prompt") + 1 == cols.index("Median columns in prompt")
    assert not any("percentile" in c for c in cols)

    # Nilai pengaturan dari notebook diterjemahkan, dan kolom pengaturan ada di depan sheet preprocessing
    settings = rt.sla_trial_tables(recap.assign(stopword="none", value_link="teks+angka", no_pk="sambung", nama_embed="asli", traversal="full"))
    row = settings["steps"].iloc[0]
    assert (row["Stopwords"], row["Value linking"], row["No-PK tables"], row["Embedded names"]) == ("none", "text + numbers", "connected", "original")
    assert list(settings["steps"].columns[:5]) == ["Stopwords", "Value linking", "Traversal", "No-PK tables", "Embedded names"]
    assert rt.sla_trial_tables(recap.assign(value_link="-"))["grid"]["Value linking"].iloc[0] == "-"

    # Rekap lama (tanpa field baru), run tanpa EX, dan belum ada rekap tidak boleh crash
    old = rt.sla_trial_tables(recap.drop(columns=["tables_pct", "ex"]))
    assert "EX (zero-shot)" not in old["grid"].columns and "Avg tables retrieved, % of DB" not in old["grid"].columns
    assert rt.sla_trial_tables(None) == {"steps": None, "grid": None}
    assert rt.sla_trial_tables(recap[recap["trial"].str.startswith("GraphRAG 1-Stage")])["steps"] is None


def test_one_stage_trial_names():
    # Rekap baru (poin 39) menamai baris 1-Stage "GraphRAG 1-Stage | T.." dan "GraphRAG 1-Stage+ | T..";
    # rekap lama "Baseline | T.." dulu selalu meng-embed nama + kolom, jadi dibaca sebagai 1-stage+
    recap = pd.DataFrame([
        recap_row("G7 + nama natural | T0.50/K0.70", 0.5, 0.7, 0.55),
        recap_row("GraphRAG 1-Stage | T0.50", 0.5, None, 0.50),
        recap_row("GraphRAG 1-Stage+ | T0.50", 0.5, None, 0.48),
        recap_row("Baseline | T0.55", 0.55, None, 0.47),
    ])
    grid = rt.sla_trial_tables(recap)["grid"]
    assert list(grid["Pipeline"]) == ["1-stage", "1-stage+", "2-stage", "1-stage+"]
    assert list(grid["EX (zero-shot)"]) == [0.50, 0.48, 0.55, 0.47]
    assert rt.pipeline_label("GraphRAG 2-Stage") == "2-stage" and rt.pipeline_label("Baseline") == "1-stage"


def test_preprocessing_steps():
    def step(name, ex, r, tokens):
        return {**recap_row(f"{name} | T0.60/K0.70", 0.6, 0.7, ex), "col_R_ne": r, "prompt_tokens_k0": tokens}

    recap = pd.DataFrame([
        step("G1 threshold saja", 0.40, 0.70, 100.0), step("G2 + stopword edge", 0.42, 0.72, 110.0),
        step("G3 + value linking", 0.50, 0.80, 120.0), step("G4 + angka", 0.45, 0.82, 200.0),
        step("G5 + fk_only", 0.55, 0.85, 90.0), step("G6 + sambung no-PK", 0.56, 0.86, 95.0),
        step("G7 + nama natural", 0.60, 0.90, 95.0), step("G8 tanpa stopword + angka", 0.62, 0.93, 99.0),
        # pasangan lain dan Baseline bukan bagian dari riwayat preprocessing
        recap_row("G7 + nama natural | T0.60/K0.75", 0.6, 0.75, 0.99), recap_row("Baseline | T0.60", 0.6, None, 0.99),
    ])
    t = rt.preprocessing_steps(recap)
    rt.formatted(t)
    assert len(t) == 8 and t.attrs["title"].endswith("T0.60/K0.70")
    assert pd.isna(t["Δ EX"].iloc[0]) and t["Compared with"].iloc[0] != t["Compared with"].iloc[0]
    assert abs(t["Δ EX"].iloc[3] - (-5.0)) < 1e-9                    # G4 vs G3, ditolak
    assert t["Compared with"].iloc[4] == "G3"                        # G5 vs G3, bukan G4
    assert abs(t["Δ EX"].iloc[4] - 5.0) < 1e-9 and abs(t["Δ Column R"].iloc[4] - 5.0) < 1e-9
    assert t["Δ Prompt tokens"].iloc[4] == -30
    assert abs(t["Δ EX"].iloc[6] - 4.0) < 1e-9
    assert t["Compared with"].iloc[7] == "G7" and abs(t["Δ EX"].iloc[7] - 2.0) < 1e-9 and t["Δ Prompt tokens"].iloc[7] == 4

    no_ex = rt.preprocessing_steps(recap.drop(columns="ex"))
    assert "Δ EX" not in no_ex.columns and "Δ Column R" in no_ex.columns
    assert rt.preprocessing_steps(recap[recap["trial"].str.startswith("Baseline")]) is None


def test_workbook_sheets():
    from openpyxl import load_workbook

    from src.experiments.export_excel import build_workbook

    recap = pd.DataFrame([
        recap_row("G1 threshold saja | T0.50/K0.70", 0.5, 0.7, 0.50),
        recap_row("G7 + nama natural | T0.50/K0.70", 0.5, 0.7, 0.55),
        recap_row("GraphRAG 1-Stage | T0.50", 0.5, None, 0.50),
    ])
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        recap.to_csv(d / "sla_trial_recap.csv", index=False)
        out = d / "x.xlsx"
        build_workbook(d / "none", d, out, skip_missing=True)
        wb = load_workbook(out)
        assert wb.sheetnames == ["Trial Preprocessing", "Trial Thresholds"]      # tanpa sheet Info
        for ws in wb:
            assert len(ws.tables) == 1 and next(iter(ws.tables.values())).ref.startswith("A1:")   # tabel langsung di baris 1
        assert wb["Trial Preprocessing"].freeze_panes == "F2" and wb["Trial Thresholds"].freeze_panes == "F2"
        text = [c.value for ws in wb for row in ws.iter_rows() for c in row if isinstance(c.value, str)]
        assert not [t for t in text if re.search(r"\bG[1-7]\b|Baseline|GraphRAG|percentile", t)], text
        wb.close()

        # Sumber tidak ada sama sekali: tetap menulis workbook berisi catatan, tidak crash
        empty = d / "empty"
        empty.mkdir()
        wb = load_workbook(build_workbook(empty, empty, d / "y.xlsx", skip_missing=True))
        assert "Info" not in wb.sheetnames and wb["Summary"]["A1"].value.startswith("Source not found")
        wb.close()


def main():
    test_full_db_summary()
    test_median_summary()
    test_ex_scorer()
    test_tables()
    test_one_stage_trial_names()
    test_preprocessing_steps()
    test_workbook_sheets()
    print("OK: threshold trial tracing tests passed")


if __name__ == "__main__":
    main()
