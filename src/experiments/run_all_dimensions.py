"""
run_all_dimensions.py — Entry point evaluasi: baca raw_logs, jalankan 6 dimensi.

Referensi: context/EVALUATION_ANALYSIS_GUIDE.md Bagian 3 (alur per dimensi),
context/IMPLEMENTATION_DECISIONS.md poin 24 (keputusan orkestrasi di file ini).

Modul ini MURNI orkestrasi — tidak menghitung metrik apapun sendiri, tidak
menginterpretasi apapun sendiri. Semua angka & semua kalimat interpretasi
datang dari src/dimensions/dim*.py. Yang ditambahkan di sini cuma: load data,
validasi skema raw_logs, urutan, gating Dimensi 5, dan penulisan output.

Usage
-----
    # Semua dimensi yang datanya tersedia
    python src/experiments/run_all_dimensions.py

    # Subset dimensi saja
    python src/experiments/run_all_dimensions.py --only 1 2 3

    # Dengan data ablation untuk Dimensi 6 (EX per k, skala 0-100)
    python src/experiments/run_all_dimensions.py --ex-per-k 0=45.2,1=52.1,3=54.0,5=53.8

Urutan pengerjaan project:
    src/utils/ -> src/metrics/ -> [generate raw_logs] -> src/dimensions/ -> file ini

PRASYARAT DATA (tiap dimensi dilewati dengan pesan jelas kalau datanya belum ada,
BUKAN crash — jadi file ini aman dijalankan kapan saja untuk lihat status):

| Dimensi | Butuh                                                             |
|---------|-------------------------------------------------------------------|
| 1,2,3   | data/raw_logs/{baseline,graphrag}_log.json                        |
| 4       | idem + data/qvt_variations/{baseline,graphrag}_qvt.json            |
| 5       | idem raw_logs (+ gating dari Dimensi 2, lihat di bawah)           |
| 6       | ex_per_k manual dari run ablation.py (--ex-per-k / --ex-per-k-file)|

raw_logs diproduksi oleh `python src/experiments/pipeline.py --baseline`
(butuh GPU/Kaggle). qvt_variations dibangun dari raw_logs itu oleh
`python src/experiments/build_qvt_variations.py`.
"""

import argparse
import importlib
import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.dimensions.dim2_structure import run_dimension_2, print_dimension_2  # noqa: E402
from src.dimensions.dim3_component import run_dimension_3, print_dimension_3  # noqa: E402
from src.dimensions.dim4_robustness import run_dimension_4, print_dimension_4  # noqa: E402
from src.dimensions.dim5_bottleneck import run_dimension_5, print_dimension_5  # noqa: E402
from src.dimensions.dim6_ablation import run_dimension_6, print_dimension_6  # noqa: E402

# Dimensi 1 di-import LAZY (bukan di sini) -- dim1_efficiency.py butuh
# PipelineConfig dari src/core/config.py, yang `import torch` untuk SATU default
# dtype (`bnb_compute_dtype`). torch tidak terpasang di environment evaluasi
# lokal (lihat requirements.txt + open item B/C di IMPLEMENTATION_DECISIONS.md),
# jadi import di top-level akan membunuh SELURUH file ini cuma karena Dimensi 1.
# Lihat _load_dimension_1().

RAW_LOGS_DIR = Path("data/raw_logs")
QVT_DIR = Path("data/qvt_variations")
OUTPUT_DIR = Path("outputs/tables")

ALL_DIMENSIONS = (1, 2, 3, 4, 5, 6)

# Skema wajib raw_logs, persis dari guide Bagian 2. Divalidasi saat load supaya
# file raw_logs yang malformed ketahuan SEKALI di awal dengan pesan yang
# menyebut field + index barisnya -- bukan meledak jadi KeyError di tengah
# dimensi ke-3 tanpa konteks.
_REQUIRED_LOG_FIELDS = (
    "query_id", "db_id", "difficulty", "gold_sql", "predicted_sql",
    "gold_schema", "predicted_schema", "token_input", "token_output",
    "esm_result", "ex_result", "cm_per_clause",
)

_DIMENSION_TITLES = {
    1: "Dimensi 1 -- Efisiensi Token & Performa (RM1)",
    2: "Dimensi 2 -- Akurasi Struktur SQL (RM2)",
    3: "Dimensi 3 -- Akurasi Komponen SQL",
    4: "Dimensi 4 -- Robustness & Konsistensi Query (RM2)",
    5: "Dimensi 5 -- Bottleneck Retrieval vs Generation",
    6: "Dimensi 6 -- Ablation Few-Shot",
}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_raw_logs(path: Path) -> List[dict]:
    """
    Load raw log JSON (list of per-query dict, skema di guide Bagian 2) dan
    validasi field wajibnya. Melempar ValueError dengan pesan spesifik kalau
    bentuknya salah -- jangan diam-diam diteruskan ke dimensi.
    """
    with open(path, "r", encoding="utf-8") as f:
        logs = json.load(f)

    if not isinstance(logs, list):
        raise ValueError(f"{path}: isi harus JSON list, dapat {type(logs).__name__}.")
    if not logs:
        raise ValueError(f"{path}: list kosong -- tidak ada query untuk dianalisis.")

    for i, entry in enumerate(logs):
        if not isinstance(entry, dict):
            raise ValueError(f"{path} baris ke-{i}: harus object, dapat {type(entry).__name__}.")
        missing = [k for k in _REQUIRED_LOG_FIELDS if k not in entry]
        if missing:
            raise ValueError(
                f"{path} baris ke-{i} (query_id={entry.get('query_id', '?')!r}): "
                f"field wajib hilang: {', '.join(missing)}. Skema lengkap ada di "
                f"context/EVALUATION_ANALYSIS_GUIDE.md Bagian 2 dan contohnya di "
                f"data/raw_logs/_example_format.json."
            )
    return logs


def _load_json(path: Path) -> Optional[list]:
    """Load JSON list kalau file-nya ada, None kalau tidak (bukan error)."""
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class _DimensionSkipped(Exception):
    """Dimensi tidak bisa dijalankan karena prasyarat datanya belum ada.

    Dibedakan dari exception lain: ini kondisi NORMAL (data memang belum
    diproduksi), bukan bug. Exception lain tetap ditandai ERROR.
    """


def _load_dimension_1():
    """
    Import run_dimension_1/print_dimension_1 + PipelineConfig secara lazy.
    Melempar _DimensionSkipped kalau dependency-nya tidak ada di environment ini.
    """
    try:
        mod = importlib.import_module("src.dimensions.dim1_efficiency")
        cfg_mod = importlib.import_module("src.core.config")
        return mod.run_dimension_1, mod.print_dimension_1, cfg_mod.PipelineConfig
    except ModuleNotFoundError as exc:
        raise _DimensionSkipped(
            f"dependency tidak tersedia di environment ini: {exc.name!r}. "
            f"Dimensi 1 butuh PipelineConfig (src/core/config.py) untuk mu = "
            f"token_output_weight, dan config.py `import torch` untuk satu default "
            f"dtype (`bnb_compute_dtype`). Remedy: `pip install torch` (versi CPU "
            f"cukup -- tidak ada inference di sini), atau jalankan Dimensi 1 di "
            f"environment Kaggle. Lihat IMPLEMENTATION_DECISIONS.md open item C."
        ) from exc


def parse_ex_per_k(spec: Optional[str], path: Optional[Path]) -> Optional[Dict[int, float]]:
    """
    Ambil ex_per_k untuk Dimensi 6 dari CLI (`0=45.2,1=52.1,...`) atau dari file
    JSON (`{"0": 45.2, "1": 52.1, ...}`). CLI menang kalau dua-duanya diberikan.

    Kenapa manual: `outputs/tables/ablation_results.csv` yang ditulis
    `ablation.py` TIDAK punya kolom EX (cuma recall/precision/token) -- EX untuk
    tiap k dihitung terpisah oleh Spider `evaluation.py` yang outputnya ke
    stdout subprocess, tidak pernah tersimpan machine-readable. Jadi angkanya
    harus dioper peneliti. Lihat IMPLEMENTATION_DECISIONS.md poin 24.
    """
    if spec:
        out: Dict[int, float] = {}
        for chunk in spec.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "=" not in chunk:
                raise ValueError(
                    f"--ex-per-k: potongan {chunk!r} tidak berbentuk k=EX "
                    f"(contoh yang benar: --ex-per-k 0=45.2,1=52.1,3=54.0,5=53.8)."
                )
            k_str, ex_str = chunk.split("=", 1)
            out[int(k_str.strip())] = float(ex_str.strip())
        return out or None

    if path is not None:
        raw = _load_json(path)
        if raw is None:
            return None
        return {int(k): float(v) for k, v in raw.items()}

    return None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def _run_one(number: int, fn) -> Tuple[dict, str]:
    """
    Jalankan satu dimensi, tangkap output print-nya sebagai teks.

    `fn()` mengembalikan `(result, print_fn)` -- print_fn ikut dikembalikan
    (bukan dioper terpisah) karena Dimensi 1 baru tahu fungsi print-nya setelah
    import lazy-nya berhasil.

    Return (record, printed_text). `record` selalu punya "status":
      "ok"      -> jalan, "result" berisi return value dimensi
      "skipped" -> prasyarat data belum ada, "reason" menjelaskan
      "error"   -> gagal tak terduga, "reason" berisi pesan exception
    """
    header = "\n" + "#" * 78 + f"\n# {_DIMENSION_TITLES[number]}\n" + "#" * 78 + "\n"
    try:
        result, print_fn = fn()
    except _DimensionSkipped as exc:
        return {"status": "skipped", "reason": str(exc)}, header + f"\n[DILEWATI] {exc}\n"
    except Exception as exc:  # noqa: BLE001 -- satu dimensi gagal tidak boleh
        # membatalkan lima lainnya; statusnya dilaporkan apa adanya di ringkasan.
        return (
            {"status": "error", "reason": f"{type(exc).__name__}: {exc}"},
            header + f"\n[ERROR] {type(exc).__name__}: {exc}\n",
        )

    buf = io.StringIO()
    with redirect_stdout(buf):
        print_fn(result)
    return {"status": "ok", "result": result}, header + buf.getvalue()


def run_all(
    raw_logs_dir: Path = RAW_LOGS_DIR,
    qvt_dir: Path = QVT_DIR,
    only: Optional[List[int]] = None,
    ex_per_k: Optional[Dict[int, float]] = None,
    always_dim5: bool = False,
) -> Tuple[dict, str]:
    """
    Jalankan dimensi yang diminta, kembalikan (results_dict, report_text).

    Dimensi 5 di-gate oleh Dimensi 2 (guide: "hanya dijalankan jika pipeline
    belum optimal", trigger kondisi "EX rendah, ESM rendah") -- gating itu
    memang tugas pemanggil, bukan run_dimension_5() sendiri (lihat docstring
    dim5_bottleneck.py). `always_dim5=True` menimpa gating itu.
    """
    selected = list(only) if only else list(ALL_DIMENSIONS)
    unknown = [d for d in selected if d not in ALL_DIMENSIONS]
    if unknown:
        raise ValueError(f"Dimensi tidak dikenal: {unknown} (yang valid: {list(ALL_DIMENSIONS)}).")

    results: Dict[str, dict] = {}
    chunks: List[str] = []

    # --- load raw_logs sekali, dipakai bersama oleh dim 1/2/3/4/5 ---
    baseline_logs: Optional[List[dict]] = None
    graphrag_logs: Optional[List[dict]] = None
    raw_logs_error: Optional[str] = None
    b_path, g_path = raw_logs_dir / "baseline_log.json", raw_logs_dir / "graphrag_log.json"
    missing = [str(p) for p in (b_path, g_path) if not p.exists()]
    if missing:
        raw_logs_error = (
            f"raw_logs belum ada: {', '.join(missing)}. Produksi dulu dengan "
            f"`python src/experiments/pipeline.py --baseline --sample 1.0` "
            f"(butuh GPU/Kaggle, lihat context/RESEARCHER_TODO.md)."
        )
    else:
        try:
            baseline_logs = load_raw_logs(b_path)
            graphrag_logs = load_raw_logs(g_path)
        except ValueError as exc:
            raw_logs_error = str(exc)

    def _need_logs() -> Tuple[List[dict], List[dict]]:
        if raw_logs_error is not None:
            raise _DimensionSkipped(raw_logs_error)
        return baseline_logs, graphrag_logs  # type: ignore[return-value]

    # --- Dimensi 1 ---
    if 1 in selected:
        def _dim1():
            b, g = _need_logs()
            run_fn, print_fn, PipelineConfig = _load_dimension_1()
            return run_fn(b, g, PipelineConfig()), print_fn
        rec, text = _run_one(1, _dim1)
        results["dimension_1"] = rec
        chunks.append(text)

    # --- Dimensi 2 (juga menentukan gating Dimensi 5) ---
    if 2 in selected:
        rec, text = _run_one(2, lambda: (run_dimension_2(*_need_logs()), print_dimension_2))
        results["dimension_2"] = rec
        chunks.append(text)

    # --- Dimensi 3 ---
    if 3 in selected:
        rec, text = _run_one(3, lambda: (run_dimension_3(*_need_logs()), print_dimension_3))
        results["dimension_3"] = rec
        chunks.append(text)

    # --- Dimensi 4 ---
    if 4 in selected:
        def _dim4():
            b, g = _need_logs()
            b_qvt = _load_json(qvt_dir / "baseline_qvt.json")
            g_qvt = _load_json(qvt_dir / "graphrag_qvt.json")
            absent = [
                str(qvt_dir / f"{c}_qvt.json")
                for c, d in (("baseline", b_qvt), ("graphrag", g_qvt)) if d is None
            ]
            if absent:
                raise _DimensionSkipped(
                    f"data QVT belum ada: {', '.join(absent)}. Bangun dengan "
                    f"`python src/experiments/build_qvt_variations.py` setelah "
                    f"raw_logs (sample 1.0) tersedia."
                )
            # raw_logs dioper juga -> silang QVT x ESM (guide langkah 6) ikut
            # jalan. Data QVT sendiri cuma bawa EX, bukan ESM (poin 10/23).
            return run_dimension_4(b_qvt, g_qvt, b, g), print_dimension_4
        rec, text = _run_one(4, _dim4)
        results["dimension_4"] = rec
        chunks.append(text)

    # --- Dimensi 5 (kondisional, di-gate Dimensi 2) ---
    if 5 in selected:
        def _dim5():
            b, g = _need_logs()
            dim2 = results.get("dimension_2")
            if not always_dim5:
                if dim2 is None or dim2["status"] != "ok":
                    raise _DimensionSkipped(
                        "gating Dimensi 2 tidak bisa dievaluasi (Dimensi 2 tidak "
                        "dijalankan atau gagal). Guide menjadikan Dimensi 5 "
                        "kondisional terhadap Dimensi 2 -- pakai --always-dim5 "
                        "untuk menjalankannya tanpa gating."
                    )
                if not dim2["result"]["classification"]["escalate_to_dimension_5"]:
                    combo = dim2["result"]["classification"]["combination"]
                    raise _DimensionSkipped(
                        f"Dimensi 2 tidak escalate (kombinasi: {combo or 'tidak terklasifikasi'}). "
                        f"Guide: Dimensi 5 dijalankan HANYA saat kondisi 'EX rendah, ESM rendah'. "
                        f"Pakai --always-dim5 untuk menjalankannya tanpa gating."
                    )
            # Dimensi 5 itu diagnostik SATU kondisi (lihat dim5_bottleneck.py).
            # Dijalankan untuk dua-duanya di sini supaya bisa dibandingkan --
            # bukan digabung jadi satu diagnosis.
            return {
                "baseline": run_dimension_5(b, condition_label="Baseline"),
                "graphrag": run_dimension_5(g, condition_label="GraphRAG"),
            }, _print_dim5

        def _print_dim5(result: dict) -> None:
            for key in ("baseline", "graphrag"):
                print_dimension_5(result[key])
                print()

        rec, text = _run_one(5, _dim5)
        results["dimension_5"] = rec
        chunks.append(text)

    # --- Dimensi 6 ---
    if 6 in selected:
        def _dim6():
            if not ex_per_k:
                raise _DimensionSkipped(
                    "ex_per_k tidak disediakan. `outputs/tables/ablation_results.csv` "
                    "tidak punya kolom EX (cuma recall/precision/token) -- EX per k "
                    "datang dari Spider evaluation.py atas "
                    "`outputs/predictions/ablation_*_predictions_k*.txt`, dan harus "
                    "dioper manual: --ex-per-k 0=45.2,1=52.1,3=54.0,5=53.8 "
                    "(atau --ex-per-k-file berisi {\"0\": 45.2, ...})."
                )
            return run_dimension_6(ex_per_k), print_dimension_6
        rec, text = _run_one(6, _dim6)
        results["dimension_6"] = rec
        chunks.append(text)

    chunks.append(_summary_text(results, selected))
    return results, "".join(chunks)


def _summary_text(results: dict, selected: List[int]) -> str:
    lines = ["\n" + "=" * 78, "RINGKASAN STATUS", "=" * 78]
    labels = {"ok": "OK", "skipped": "DILEWATI", "error": "ERROR"}
    for n in selected:
        rec = results.get(f"dimension_{n}")
        if rec is None:
            continue
        status = labels[rec["status"]]
        lines.append(f"Dimensi {n}: {status}")
        if rec["status"] != "ok":
            lines.append(f"           {rec['reason']}")
    n_ok = sum(1 for r in results.values() if r["status"] == "ok")
    n_err = sum(1 for r in results.values() if r["status"] == "error")
    lines.append("")
    lines.append(f"{n_ok}/{len(results)} dimensi berhasil dijalankan"
                 + (f", {n_err} ERROR (bukan sekadar data belum ada)" if n_err else "."))
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Jalankan 6 dimensi analisis skripsi atas data/raw_logs/.")
    ap.add_argument("--raw-logs-dir", default=str(RAW_LOGS_DIR))
    ap.add_argument("--qvt-dir", default=str(QVT_DIR))
    ap.add_argument("--out-dir", default=str(OUTPUT_DIR))
    ap.add_argument("--only", nargs="+", type=int, metavar="N",
                    help="Jalankan hanya dimensi ini (mis. --only 1 2 3).")
    ap.add_argument("--ex-per-k", default=None, metavar="SPEC",
                    help="EX per k untuk Dimensi 6, skala 0-100 "
                         "(mis. 0=45.2,1=52.1,3=54.0,5=53.8).")
    ap.add_argument("--ex-per-k-file", default=None, metavar="PATH",
                    help='File JSON {"0": 45.2, "1": 52.1, ...} sebagai alternatif --ex-per-k.')
    ap.add_argument("--always-dim5", action="store_true",
                    help="Jalankan Dimensi 5 tanpa menunggu gating dari Dimensi 2.")
    ap.add_argument("--no-save", action="store_true",
                    help="Cuma print ke stdout, jangan tulis file apapun ke --out-dir.")
    args = ap.parse_args()

    # Kesalahan ARGUMEN (bukan kesalahan data) dilaporkan bersih lewat argparse,
    # bukan sebagai traceback -- ini jalur CLI yang dipakai peneliti langsung.
    try:
        ex_per_k = parse_ex_per_k(
            args.ex_per_k, Path(args.ex_per_k_file) if args.ex_per_k_file else None
        )
        results, report = run_all(
            raw_logs_dir=Path(args.raw_logs_dir),
            qvt_dir=Path(args.qvt_dir),
            only=args.only,
            ex_per_k=ex_per_k,
            always_dim5=args.always_dim5,
        )
    except ValueError as exc:
        ap.error(str(exc))
        return 2  # tidak tercapai (ap.error() exit 2) -- untuk kejelasan saja

    print(report)

    if not args.no_save:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        json_path = out_dir / "dimensions_results.json"
        report_path = out_dir / "dimensions_report.txt"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Tersimpan: {json_path}")
        print(f"Tersimpan: {report_path}")

    # Exit code 1 HANYA untuk error tak terduga -- "data belum ada" itu kondisi
    # normal di tahap ini dan tidak boleh bikin CI/script pemanggil gagal.
    return 1 if any(r["status"] == "error" for r in results.values()) else 0


if __name__ == "__main__":
    sys.exit(main())
