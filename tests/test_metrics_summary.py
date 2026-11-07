"""
Tes lokal alur metrik -> Excel -> gabungan dengan raw_logs sintetis (tanpa GPU/torch/data Spider).

    python tests/test_metrics_summary.py      (atau: python -m pytest tests/test_metrics_summary.py)
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from openpyxl import load_workbook  # noqa: E402

from src.experiments import combine_models as cm  # noqa: E402
from src.experiments import result_tables as rt  # noqa: E402
from src.experiments.export_excel import build_workbook  # noqa: E402
from src.experiments.metrics_summary import build_metrics_results, save_metrics_results  # noqa: E402

CLAUSE_VALUES = {c: None for c in rt.CLAUSES}


def _entry(i, ex, esm, t_in, t_out, predicted, difficulty="easy", where=1):
    return {
        "query_id": f"dev_{i:04d}", "db_id": "db", "difficulty": difficulty,
        "gold_sql": "SELECT a FROM t", "predicted_sql": "SELECT a FROM t",
        "gold_schema": ["t.a"], "predicted_schema": predicted,
        "token_input": t_in, "token_output": t_out, "esm_result": esm, "ex_result": ex,
        "cm_per_clause": {**CLAUSE_VALUES, "select": 1, "where": where},
    }


BASELINE = [_entry(1, 1, 1, 100, 10, ["t.a", "t.b"]), _entry(2, 0, 0, 100, 10, ["t.a", "t.b"], "hard", where=0)]
GRAPHRAG = [_entry(1, 1, 1, 50, 10, ["t.a"]), _entry(2, 1, 0, 50, 10, ["t.a"], "hard")]


def _qvt_dir(tmp: Path) -> Path:
    qvt = tmp / "qvt"
    qvt.mkdir()
    for condition in ("baseline", "graphrag"):
        (qvt / f"{condition}_qvt.json").write_text(json.dumps([
            {"query_id": "dev_0001", "variations": [{"is_correct": 1}, {"is_correct": 0}]},
            {"query_id": "dev_0002", "variations": [{"is_correct": 0}, {"is_correct": 0}]},
        ]), encoding="utf-8")
    return qvt


def test_six_metrics_values():
    with tempfile.TemporaryDirectory() as d:
        r = build_metrics_results(BASELINE, GRAPHRAG, 3.0, qvt_dir=_qvt_dir(Path(d)))
    m = r["metrics"]
    assert set(m) == {"sla", "esm", "ex", "cm", "token_tep", "qvt"}
    assert m["ex"]["baseline"]["overall"] == 0.5 and m["ex"]["graphrag"]["overall"] == 1.0
    assert m["esm"]["graphrag"]["by_difficulty"]["hard"] == 0
    assert m["cm"]["baseline"]["where"] == 0.5 and m["cm"]["baseline"]["having"] is None
    assert m["sla"]["graphrag"]["column"]["f1"] == 1.0
    assert abs(m["sla"]["baseline"]["column"]["precision"] - 0.5) < 1e-9
    assert m["token_tep"]["token"]["baseline"]["t_mean"] == 130
    assert m["token_tep"]["token"]["graphrag"]["t_mean"] == 80
    # TEP = ((1.0 - 0.5) / 0.5) / ((80 - 130) / 130) = -2.6
    assert abs(m["token_tep"]["tep"]["tep"] - (-2.6)) < 1e-9
    # Grup yang semua variasinya salah dibuang (filter wajib guide 1.6): hanya dev_0001 terhitung.
    assert m["qvt"]["graphrag"] == {"status": "ok", "qvt": 0.5, "queries_included": 1, "queries_total": 2}


def test_summary_table_has_all_six_metrics_in_one_table():
    with tempfile.TemporaryDirectory() as d:
        table = rt.metrics_summary_table(build_metrics_results(BASELINE, GRAPHRAG, 3.0, qvt_dir=_qvt_dir(Path(d))))
    rows = table.set_index("Metric")
    for metric in ("SLA column F1", "ESM", "EX", "CM SELECT", "CM WHERE", "Mean token consumption (T)", "TEP (2-Stage vs 1-Stage)", "QVT"):
        assert metric in rows.index, metric
    assert "CM HAVING" not in rows.index  # klausa tanpa nilai tidak ditampilkan
    assert abs(rows.loc["EX", "Difference (2-Stage − 1-Stage)"] - 50.0) < 1e-9
    assert rows.loc["Mean token consumption (T)", "Difference (2-Stage − 1-Stage)"] == -50
    assert abs(rows.loc["TEP (2-Stage vs 1-Stage)", "2-Stage"] - (-2.6)) < 1e-9
    trial = rt.metrics_summary_table(build_metrics_results(BASELINE, GRAPHRAG, 3.0)).set_index("Metric")
    assert "run penuh" in trial.loc["QVT", "Note"]


def test_qvt_and_tep_not_available():
    trial = build_metrics_results(BASELINE, GRAPHRAG, 3.0)
    assert trial["metrics"]["qvt"]["baseline"]["status"] == "not_available"
    same_t = build_metrics_results(BASELINE, BASELINE, 3.0)
    assert same_t["metrics"]["token_tep"]["tep"]["status"] == "not_available"
    with tempfile.TemporaryDirectory() as d:
        stale = build_metrics_results(BASELINE, GRAPHRAG, 3.0, qvt_dir=_qvt_dir(Path(d)), qvt_conditions=[])
    assert stale["metrics"]["qvt"]["graphrag"]["status"] == "not_available"


def _write_run(root: Path, model: str, sample_ratio=1.0, k=3, with_tep=True) -> None:
    slug = model.split("/")[-1]
    tables = root / slug / "tables"
    logs = root / slug / "raw_logs"
    logs.mkdir(parents=True)
    g = GRAPHRAG if with_tep else BASELINE
    meta = {"llm_model": model, "sample_ratio": sample_ratio, "is_full_run": sample_ratio >= 1.0, "commit": "abc",
            "seed": 42, "few_shot_k": k, "few_shot_selection": "skeleton", "table_threshold": 0.6, "column_threshold": 0.7}
    save_metrics_results(build_metrics_results(BASELINE, g, 3.0, meta=meta), tables)
    (logs / "baseline_log.json").write_text(json.dumps(BASELINE), encoding="utf-8")
    (logs / "graphrag_log.json").write_text(json.dumps(g), encoding="utf-8")


def test_excel_has_metric_sheets_in_order():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _write_run(root, "Qwen/Qwen2.5-Coder-7B-Instruct")
        run = root / "Qwen2.5-Coder-7B-Instruct"
        out = build_workbook(run / "raw_logs", run / "tables", run / "tables" / "hasil.xlsx")
        wb = load_workbook(out)
        names = wb.sheetnames
        assert names[:8] == ["Info", "Summary", "M1 SLA", "M2 ESM", "M3 EX", "M4 CM", "M5 Token TEP", "M6 QVT"]
        assert names.index("M6 QVT") < names.index("Dimensions") < names.index("Overall")
        assert any("TEP" in str(c.value) for row in wb["M5 Token TEP"].iter_rows() for c in row)
        wb.close()


def test_combine_three_models():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d) / "runs"
        for size in ("14B", "3B", "7B"):
            _write_run(root, f"Qwen/Qwen2.5-Coder-{size}-Instruct")
        runs = cm.load_runs(root)
        assert list(runs) == [f"Qwen2.5-Coder-{s}-Instruct" for s in ("3B", "7B", "14B")]
        warnings = cm.find_warnings(runs)
        assert all("dimensions_results.json" in w for w in warnings), warnings
        table = cm.summary_table(runs)
        assert len(table) == 6
        assert table["TEP (2-Stage vs 1-Stage)"].notna().sum() == 3
        json_path, xlsx_path = cm.write_outputs(runs, warnings, Path(d) / "out", Path(d) / "none")
        assert set(json.loads(json_path.read_text(encoding="utf-8"))["models"]) == set(runs)
        wb = load_workbook(xlsx_path)
        assert wb.sheetnames[:4] == ["Info", "Summary", "All Metrics", "M1 SLA"]
        assert "Token Reduction" in wb.sheetnames
        wb.close()


def test_combine_flags_incomparable_and_duplicate_runs():
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        _write_run(root, "Qwen/Qwen2.5-Coder-3B-Instruct", sample_ratio=0.05)
        _write_run(root, "Qwen/Qwen2.5-Coder-14B-Instruct", k=5)
        warnings = cm.find_warnings(cm.load_runs(root), ["Qwen/Qwen2.5-Coder-7B-Instruct"])
        text = "\n".join(warnings)
        assert "7B" in text and "run percobaan" in text and "few_shot_k" in text and "sample_ratio" in text

        _write_run(root / "copy", "Qwen/Qwen2.5-Coder-3B-Instruct")
        try:
            cm.load_runs(root)
        except ValueError as exc:
            assert "dua kali" in str(exc)
        else:
            raise AssertionError("model duplikat harus ditolak")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"OK {name}")
