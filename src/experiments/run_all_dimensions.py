"""
run_all_dimensions.py — Entry point

Usage:
    python src/experiments/run_all_dimensions.py

Jalankan ini SETELAH semua metric di src/metrics/ sudah terisi dan raw_logs
(data/raw_logs/baseline_log.json, data/raw_logs/graphrag_log.json) sudah ada.

Urutan pengerjaan project (lihat CLAUDE.md root + context/EVALUATION_ANALYSIS_GUIDE.md):
    src/utils/ -> src/metrics/ -> [generate raw_logs] -> src/dimensions/ -> file ini

Saat ini masih skeleton — isi tiap load_* function dan panggilan run_dimension_*
setelah modul terkait selesai diimplementasikan. Jangan jalankan dimensi yang
dependency metric-nya belum selesai (misalnya jangan panggil run_dimension_4
kalau data/qvt_variations/ masih kosong).

⚠️ GAP DATA BELUM DISELESAIKAN: raw_logs/{baseline,graphrag}_log.json dengan
skema lengkap di guide Bagian 2 (query_id, difficulty, gold_schema,
predicted_schema, esm_result, ex_result, cm_per_clause, dst.) BELUM diproduksi
oleh pipeline manapun saat ini. Yang sudah ada:
    - outputs/predictions/predictions.txt, baseline_predictions.txt,
      ablation_*_predictions_k{k}.txt — SQL per baris saja, tanpa metadata.
    - outputs/predictions/ablation_*_prompts_k{k}.jsonl — ADA tokens_in/
      tokens_out/token_consumption/prompt/pred_sql per baris, tapi TIDAK ada
      gold_schema/predicted_schema/esm_result/ex_result/cm_per_clause/difficulty.
    - outputs/logs/baseline_log.txt, outputs/tables/baseline_results.csv —
      cuma index/db_id/recall/precision.
Perlu logging baru di src/experiments/pipeline.py & ablation.py untuk
menghasilkan raw_logs sesuai skema guide — belum dikerjakan, di luar scope
"combine & adjust" saat ini.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# from src.dimensions.dim1_efficiency import run_dimension_1
# from src.dimensions.dim2_structure import run_dimension_2
# from src.dimensions.dim3_component import run_dimension_3
# from src.dimensions.dim4_robustness import run_dimension_4
# from src.dimensions.dim5_bottleneck import run_dimension_5
# from src.dimensions.dim6_ablation import run_dimension_6

RAW_LOGS_DIR = Path("data/raw_logs")
OUTPUT_DIR = Path("outputs/tables")


def load_raw_logs(filename: str) -> list:
    """Load raw log JSON (list of per-query dict, format ada di guide Bagian 2)."""
    path = RAW_LOGS_DIR / filename
    with open(path, "r") as f:
        return json.load(f)


def main():
    """
    TODO (isi bertahap sesuai progress implementasi metric):
    baseline_logs = load_raw_logs("baseline_log.json")
    graphrag_logs = load_raw_logs("graphrag_log.json")

    result_dim1 = run_dimension_1(baseline_logs, graphrag_logs)
    result_dim2 = run_dimension_2(baseline_logs, graphrag_logs)
    result_dim3 = run_dimension_3(baseline_logs, graphrag_logs)
    # dim4 & dim5 & dim6 menyusul sesuai prasyarat masing-masing

    # simpan/print hasil ke OUTPUT_DIR
    """
    raise NotImplementedError


if __name__ == "__main__":
    main()
