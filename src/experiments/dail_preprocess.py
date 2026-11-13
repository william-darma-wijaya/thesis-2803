"""
dail_preprocess.py — jalankan preprocessing DAIL-SQL sekali, hasilnya ke data/dail/.

Isinya hanya memanggil kode DAIL asli (external/dail_sql/data_preprocess.py), dengan path
diarahkan ke dataset Spider skripsi dan folder keluaran yang bisa ditulis (Kaggle input read-only):

  1. Gabungkan train_spider.json + train_others.json -> data/dail/train_spider_and_others.json
     (sama dengan data_preprocess.py baris 114-124: pool contoh DAIL = kedua split train).
  2. schema_linking_producer() DAIL -> data/dail/enc/{test,train}_schema-linking.jsonl
     (schema linking ala RAT-SQL, dipakai DAIL untuk me-mask pertanyaan).
  3. Cek jumlah baris linking = jumlah pertanyaan (DAIL mencocokkannya per posisi).

Butuh dependensi DAIL untuk preprocessing (BUKAN untuk run utama): Java + Stanford CoreNLP
2018-10-05 (CORENLP_HOME), paket `stanford-corenlp`, `torchtext` (GloVe 42B, ~5 GB, diunduh ke
--cache-dir), `bpemb`, `attrs`, `nltk` (stopwords, punkt). torchtext butuh versi torch yang cocok,
jadi sebaiknya dijalankan di environment terpisah (lihat notebooks/eval_dail.ipynb bagian 0).

    python src/experiments/dail_preprocess.py --spider-dir data/spider_data --out-dir data/dail

Kode custom skripsi; lihat context/IMPLEMENTATION_DECISIONS.md poin 43.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
TRAIN_SPLITS = ("train_spider.json", "train_others.json")
TRAIN_FILE = "train_spider_and_others.json"


def merge_train(spider_dir: Path, out_dir: Path) -> Path:
    """Langkah 1: sama dengan data_preprocess.py DAIL, tapi ditulis ke out_dir."""
    total = []
    for split in TRAIN_SPLITS:
        total += json.loads((spider_dir / split).read_text(encoding="utf-8"))
    out = out_dir / TRAIN_FILE
    out.write_text(json.dumps(total), encoding="utf-8")
    print(f"Train gabungan: {len(total)} pertanyaan -> {out}")
    return out


def count_lines(path: Path) -> int:
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def main() -> int:
    ap = argparse.ArgumentParser(description="Preprocessing DAIL-SQL (schema linking) untuk Spider dev + train.")
    ap.add_argument("--spider-dir", type=Path, required=True, help="Folder Spider: dev.json, tables.json, database/, train_*.json")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data" / "dail")
    ap.add_argument("--dail-dir", type=Path, default=REPO_ROOT / "external" / "dail_sql")
    ap.add_argument("--cache-dir", type=Path, default=REPO_ROOT / "data" / "dail_cache",
                    help="Tempat unduhan GloVe 42B (CACHE_DIR milik DAIL), ~5 GB; jangan di-commit")
    args = ap.parse_args()

    spider_dir, out_dir = args.spider_dir.resolve(), args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CACHE_DIR", str(args.cache_dir.resolve()))
    os.environ.setdefault("CORENLP_HOME", str((args.dail_dir / "third_party" / "stanford-corenlp-full-2018-10-05").resolve()))
    print(f"CACHE_DIR    = {os.environ['CACHE_DIR']}")
    print(f"CORENLP_HOME = {os.environ['CORENLP_HOME']}")

    start = time.perf_counter()
    train_path = merge_train(spider_dir, out_dir)

    # Langkah 2: fungsi DAIL asli. Path absolut: os.path.join / Path "/" milik DAIL mengembalikan
    # path absolut apa adanya, jadi dataset_dir hanya menentukan lokasi keluaran (out_dir/enc/).
    sys.path.insert(0, str(args.dail_dir.resolve()))
    from data_preprocess import schema_linking_producer
    schema_linking_producer(str(spider_dir / "dev.json"), str(train_path), str(spider_dir / "tables.json"),
                            str(spider_dir / "database"), str(out_dir))

    # Langkah 3: data_builder.py DAIL memasangkan baris linking ke pertanyaan per posisi
    n_dev = len(json.loads((spider_dir / "dev.json").read_text(encoding="utf-8")))
    n_train = len(json.loads(train_path.read_text(encoding="utf-8")))
    n_test_link = count_lines(out_dir / "enc" / "test_schema-linking.jsonl")
    n_train_link = count_lines(out_dir / "enc" / "train_schema-linking.jsonl")
    print(f"dev {n_dev} / linking {n_test_link} | train {n_train} / linking {n_train_link}")
    print(f"Selesai dalam {time.perf_counter() - start:,.0f} detik -> {out_dir}")
    if (n_dev, n_train) != (n_test_link, n_train_link):
        print("[ERROR] jumlah baris linking tidak sama dengan jumlah pertanyaan")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
