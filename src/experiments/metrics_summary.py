"""
metrics_summary.py — rangkum 6 metrik satu model jadi metrics_results.json.

Enam metrik mengikuti context/EVALUATION_ANALYSIS_GUIDE.md Bagian 1:
    SLA, ESM, EX, CM, Token Consumption + TEP, QVT.

Dua kondisi yang dibandingkan:
    "baseline" = GraphRAG 1-Stage (nama kode lama, retrieval level tabel)
    "graphrag" = GraphRAG 2-Stage (retrieval level tabel lalu kolom)

Tidak ada formula baru. Modul ini hanya merangkum raw_logs dengan fungsi yang sudah
ada di src/metrics/ (kode custom skripsi). Nilai ESM/EX/CM per query di raw_logs berasal
dari esm_ex_cm.py, yang membungkus evaluator resmi Spider. TEP sama dengan Dimensi 1.

Semua persentase disimpan sebagai pecahan 0-1. Hanya angka: teks interpretasi tidak
ditulis di sini (sama dengan result_tables.py).
"""

import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.experiments.result_tables import CLAUSES, DIFFICULTIES, _t, condition_stats  # noqa: E402
from src.metrics.qvt import aggregate_qvt, compute_qvt_per_query  # noqa: E402
from src.metrics.sla import aggregate_sla, compute_sla  # noqa: E402
from src.metrics.tep import compute_tep  # noqa: E402
from src.utils import timing  # noqa: E402

CONDITIONS = ("baseline", "graphrag")
CONDITION_LABELS = {"baseline": "GraphRAG 1-Stage", "graphrag": "GraphRAG 2-Stage"}
RESULTS_FILENAME = "metrics_results.json"


def _mean(values: list) -> Optional[float]:
    return sum(values) / len(values) if values else None


def _by_difficulty(logs: list, value) -> dict:
    """Rata-rata value(entry) per tingkat kesulitan; tingkat tanpa query = None."""
    return {d: _mean([value(e) for e in logs if e["difficulty"] == d]) for d in DIFFICULTIES}


def _sla_block(logs: list, level: str) -> Optional[dict]:
    if not logs:
        return None
    return asdict(aggregate_sla([compute_sla(set(e["gold_schema"]), set(e["predicted_schema"]), level=level)
                                 for e in logs]))


def _sla(logs: list) -> dict:
    return {
        "table": _sla_block(logs, "table"),
        "column": _sla_block(logs, "column"),
        "column_non_empty_gold": _sla_block([e for e in logs if e["gold_schema"]], "column"),
        "column_by_difficulty": {d: _sla_block([e for e in logs if e["difficulty"] == d], "column")
                                 for d in DIFFICULTIES},
    }


def _accuracy(logs: list, key: str) -> dict:
    return {"overall": _mean([e[key] for e in logs]), "by_difficulty": _by_difficulty(logs, lambda e: e[key])}


def _cm(logs: list) -> dict:
    """Rata-rata per klausa atas nilai non-None. Sengaja tidak digabung jadi satu angka (guide 1.4)."""
    return {c: _mean([e["cm_per_clause"][c] for e in logs if e["cm_per_clause"].get(c) is not None])
            for c in CLAUSES}


def _token(logs: list, alpha: float) -> dict:
    stats = condition_stats(logs, alpha)
    return {
        "t_in": stats.get("t_in"), "t_out": stats.get("t_out"),
        "t_mean": stats.get("t"), "t_median": stats.get("t_median"),
        "t_mean_by_difficulty": _by_difficulty(logs, lambda e: _t(e, alpha)),
    }


def _tep(baseline_logs: list, graphrag_logs: list, alpha: float) -> dict:
    """TEP tidak terdefinisi kalau EX 1-Stage = 0 atau T kedua kondisi sama: dilaporkan, bukan crash."""
    if not baseline_logs or not graphrag_logs:
        return {"status": "not_available", "reason": "raw_logs kosong"}
    try:
        result = compute_tep(
            ex_baseline=_mean([e["ex_result"] for e in baseline_logs]),
            ex_graphrag=_mean([e["ex_result"] for e in graphrag_logs]),
            t_baseline=_mean([_t(e, alpha) for e in baseline_logs]),
            t_graphrag=_mean([_t(e, alpha) for e in graphrag_logs]),
        )
    except ValueError as exc:
        return {"status": "not_available", "reason": str(exc)}
    return {"status": "ok", "delta_ex": result.delta_ex, "delta_t": result.delta_t, "tep": result.tep}


def _qvt(condition: str, qvt_dir: Optional[Path], qvt_conditions: Optional[Iterable[str]]) -> dict:
    """qvt_conditions = kondisi yang data QVT-nya dibangun dari raw_logs run ini; file lama tidak dibaca."""
    if qvt_dir is None:
        return {"status": "not_available", "reason": "butuh run penuh (SAMPLE_RATIO = 1.0)"}
    path = Path(qvt_dir) / f"{condition}_qvt.json"
    if (qvt_conditions is not None and condition not in qvt_conditions) or not path.exists():
        return {"status": "not_available", "reason": f"data QVT run ini tidak ada: {path}"}
    data = json.loads(path.read_text(encoding="utf-8"))
    per_query = [compute_qvt_per_query(e["query_id"], e["variations"]) for e in data]
    try:
        value = aggregate_qvt(per_query)
    except ValueError as exc:
        return {"status": "not_available", "reason": str(exc)}
    return {"status": "ok", "qvt": value, "queries_included": sum(r.included for r in per_query),
            "queries_total": len(per_query)}


def build_metrics_results(
    baseline_logs: list,
    graphrag_logs: list,
    alpha: float,
    qvt_dir: Optional[Path] = None,
    qvt_conditions: Optional[Iterable[str]] = None,
    meta: Optional[dict] = None,
) -> dict:
    """Keenam metrik dari raw_logs. qvt_dir=None -> QVT ditandai butuh run penuh."""
    logs = {"baseline": baseline_logs, "graphrag": graphrag_logs}
    return {
        "meta": {**(meta or {}), "alpha": alpha, "condition_labels": CONDITION_LABELS,
                 "queries": {c: len(logs[c]) for c in CONDITIONS}},
        "metrics": {
            "sla": {c: _sla(logs[c]) for c in CONDITIONS},
            "esm": {c: _accuracy(logs[c], "esm_result") for c in CONDITIONS},
            "ex": {c: _accuracy(logs[c], "ex_result") for c in CONDITIONS},
            "cm": {c: _cm(logs[c]) for c in CONDITIONS},
            "token_tep": {
                "token": {c: _token(logs[c], alpha) for c in CONDITIONS},
                "tep": _tep(baseline_logs, graphrag_logs, alpha),
            },
            "qvt": {c: _qvt(c, qvt_dir, qvt_conditions) for c in CONDITIONS},
        },
    }


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True)
        return out.stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def run_metadata(cfg, sample_ratio: float) -> dict:
    """Identitas run. combine_models.py memakainya untuk memastikan ketiga model sebanding."""
    return {
        "llm_model": cfg.llm_model,
        "sample_ratio": sample_ratio,
        "is_full_run": sample_ratio >= 1.0,
        "commit": _git_commit(),
        "seed": cfg.seed,
        "few_shot_k": cfg.few_shot_k,
        "few_shot_selection": cfg.few_shot_selection,
        "table_threshold": cfg.table_similarity_threshold,
        "column_threshold": cfg.semantic_similarity_threshold,
        "gpu": timing.gpu_name(),
        "generated_at": f"{datetime.now():%Y-%m-%d %H:%M}",
    }


def save_metrics_results(results: dict, tables_dir: Path) -> Path:
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / RESULTS_FILENAME
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return path
