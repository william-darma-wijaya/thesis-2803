"""
trial_ex.py — EX per konfigurasi untuk trial threshold (kode custom skripsi, poin 35).

Banyak konfigurasi grid menghasilkan DDL yang sama untuk satu pertanyaan, dan greedy
decoding deterministik, jadi prompt yang sama tidak perlu digenerate dua kali. Hasil
ditulis per baris ke JSONL: kalau kernel Kaggle mati, run berikutnya melanjutkan dari
file itu. Skor EX tetap dari evaluator yang sama dengan ablation.py (eval_exec_match()
resmi SPIDER lewat esm_ex_cm.py).
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


class ExScorer:
    """
    generate(prompt) -> (sql, n_output_tokens); evaluate(sql, gold_sql, db_id) -> QueryEvalResult.
    Kunci cache memuat nama model, jadi hasil model lain tidak terpakai.
    """

    def __init__(self, generate: Callable, evaluate: Callable, cache_path: Path, model_name: str):
        self.generate, self.evaluate = generate, evaluate
        self.cache_path, self.model_name = Path(cache_path), model_name
        self.generated, self.reused, self.failed = 0, 0, 0
        self._memo: dict = {}
        if self.cache_path.exists():
            with open(self.cache_path, encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    self._memo[rec["key"]] = rec

    def _key(self, prompt: str, gold_sql: str, db_id: str) -> str:
        text = "\0".join((self.model_name, db_id, gold_sql, prompt))
        return hashlib.sha1(text.encode("utf-8")).hexdigest()

    def score(self, prompt: str, gold_sql: str, db_id: str) -> int:
        """EX 0/1. Generasi yang gagal dihitung salah tetapi TIDAK disimpan, supaya dicoba lagi saat run ulang."""
        key = self._key(prompt, gold_sql, db_id)
        if key in self._memo:
            self.reused += 1
            return self._memo[key]["ex"]
        try:
            sql, n_out = self.generate(prompt)
        except Exception:
            logger.exception("Generasi gagal (db=%s)", db_id)
            self.failed += 1
            return 0
        ev = self.evaluate(sql, gold_sql, db_id)
        rec = {"key": key, "db_id": db_id, "pred_sql": sql, "tokens_out": n_out, "ex": ev.ex, "esm": ev.esm}
        self._memo[key] = rec
        self.generated += 1
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return ev.ex
