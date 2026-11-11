"""
dail.py — DAIL-SQL (Gao et al., 2023) sebagai baseline skripsi, dijalankan per pertanyaan.

Kode DAIL asli ada di external/dail_sql/ (salinan apa adanya dari github.com/beachwang/DAIL-SQL,
commit 2061f68, tanpa .git, img/, dan results/). File di sana TIDAK diubah. Modul ini kode custom
skripsi yang membungkusnya, sama polanya dengan external/spider_eval/ + src/metrics/:

- Arsitektur DAIL dipakai apa adanya: representasi schema `SQL` (CREATE TABLE seluruh DB dari
  sqlite_master), organisasi contoh `QA`, seleksi `EUCDISQUESTIONMASK` untuk SQL awal lalu
  `EUCDISMASKPRESKLSIMTHR` (DAIL Selection), k = 9, embedding all-mpnet-base-v2, threshold 0.85,
  pool contoh train_spider + train_others, cleaner SQL milik ask_llm.py.
- Yang diganti hanya LLM: GPT-4 -> Qwen2.5-Coder (4-bit, greedy), dimuat dan dipanggil persis
  seperti pipeline 1-Stage/2-Stage (prompt mentah tanpa chat template, cfg.max_new_tokens).
  Kedua pass memakai Qwen yang sama. Self-consistency tidak dipakai.
- Script CLI DAIL (generate_question.py, ask_llm.py) tidak dipakai saat run: kelas prompt dan
  selector-nya dipanggil langsung per pertanyaan, jadi pass 1 -> pass 2 tidak perlu file hasil
  untuk seluruh dev set, dan sampel stratified yang sama dengan run kita bisa dipakai.

Lihat context/IMPLEMENTATION_DECISIONS.md poin 43.
"""

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.utils import timing

REPO_ROOT = Path(__file__).resolve().parents[2]
# Nama file yang ditulis data_preprocess.py DAIL (dan src/experiments/dail_preprocess.py)
TRAIN_FILE = "train_spider_and_others.json"
TEST_LINKING = Path("enc") / "test_schema-linking.jsonl"
TRAIN_LINKING = Path("enc") / "train_schema-linking.jsonl"
# Tokenizer di kode DAIL hanya untuk menghitung biaya, dan get_tokenizer()/count_tokens() DAIL
# selalu mengembalikan 0. Nilainya tidak berpengaruh; token dihitung dengan tokenizer Qwen.
DAIL_TOKENIZER_NAME = "gpt-3.5-turbo"
# Nama paket top-level milik DAIL. Kode DAIL meng-import-nya sebagai `utils.*`, `prompt.*`.
DAIL_PACKAGES = ("utils", "prompt", "llm")


def dail_path(cfg) -> Path:
    path = Path(cfg.dail_dir)
    return path if path.is_absolute() else REPO_ROOT / path


def dail_data_path(cfg) -> Path:
    path = Path(cfg.dail_data_dir)
    return path if path.is_absolute() else REPO_ROOT / path


def import_dail(cfg) -> dict:
    """
    Import modul DAIL dari external/dail_sql/. Paket DAIL bernama `utils` dan `prompt` (top-level),
    jadi folder itu dimasukkan ke sys.path. Kalau paket lain bernama sama sudah ter-import,
    berhenti dengan pesan jelas daripada diam-diam memakai modul yang salah.
    """
    root = dail_path(cfg).resolve()
    if not (root / "prompt" / "prompt_builder.py").is_file():
        raise FileNotFoundError(f"Kode DAIL tidak ditemukan di {root}")
    for name in DAIL_PACKAGES:
        module = sys.modules.get(name)
        if module is None:
            continue
        locations = list(getattr(module, "__path__", [])) or [getattr(module, "__file__", "") or ""]
        if not any(Path(loc).resolve().is_relative_to(root) for loc in locations if loc):
            raise ImportError(f"Modul '{name}' yang sudah ter-import bukan milik DAIL ({locations}). "
                              f"Restart kernel lalu jalankan ulang tanpa meng-import paket '{name}' lain.")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import prompt.PromptReprTemplate as repr_module
    import prompt.prompt_builder as prompt_builder
    import utils.data_builder as data_builder
    import utils.utils as dail_utils
    from utils.post_process import process_duplication
    return {"prompt_builder": prompt_builder, "repr_module": repr_module, "data_builder": data_builder,
            "utils": dail_utils, "process_duplication": process_duplication}


def check_preprocessed(cfg, n_dev: int | None = None) -> dict:
    """
    Pastikan hasil data_preprocess.py DAIL ada dan sejajar dengan dev.json / train. Kode DAIL
    mencocokkan baris linking dengan pertanyaan per POSISI, jadi jumlahnya harus sama persis.
    """
    data_dir = dail_data_path(cfg)
    paths = {"train": data_dir / TRAIN_FILE, "test_linking": data_dir / TEST_LINKING,
             "train_linking": data_dir / TRAIN_LINKING}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise FileNotFoundError("Hasil preprocessing DAIL belum ada: " + ", ".join(missing)
                                + ". Jalankan src/experiments/dail_preprocess.py (lihat notebooks/eval_dail.ipynb bagian 0).")
    counts = {
        "dev": n_dev if n_dev is not None else len(json.loads(Path(cfg.dev_json).read_text(encoding="utf-8"))),
        "train": len(json.loads(paths["train"].read_text(encoding="utf-8"))),
        "test_linking": sum(1 for line in open(paths["test_linking"], encoding="utf-8") if line.strip()),
        "train_linking": sum(1 for line in open(paths["train_linking"], encoding="utf-8") if line.strip()),
    }
    if counts["test_linking"] != counts["dev"] or counts["train_linking"] != counts["train"]:
        raise ValueError(f"Jumlah baris linking DAIL tidak sejajar: {counts}. Jalankan ulang dail_preprocess.py.")
    return counts


def _make_dataset(dail: dict, cfg):
    """SpiderDataset DAIL dengan path diarahkan ke data skripsi (Kaggle input read-only) dan data/dail/."""
    base = dail["data_builder"].BasicDataset
    data_dir = dail_data_path(cfg)

    class ThesisSpiderDataset(base):
        name = "spider"

        def __init__(self):
            # Atribut yang sama dengan BasicDataset.__init__, hanya lokasinya yang berbeda
            self.path_data = str(cfg.data_path)
            self.path_db = str(cfg.db_dir)
            self.test_json = str(cfg.dev_json)
            self.test_gold = str(cfg.gold_sql)
            self.train_json = str(data_dir / TRAIN_FILE)
            self.train_gold = str(Path(cfg.data_path) / "train_gold.sql")
            self.table_json = str(cfg.tables_json)
            self.path_test_schema_linking = str(data_dir / TEST_LINKING)
            self.path_train_schema_linking = str(data_dir / TRAIN_LINKING)
            self.mini_test_index_json = None
            self.pre_test_result = None
            self.databases = None

    return ThesisSpiderDataset()


def _timed(name: str, func: Callable) -> Callable:
    """Bungkus fungsi DAIL dengan timing.step(); input dan output tidak berubah."""
    def wrapper(*args, **kwargs):
        with timing.step(name):
            return func(*args, **kwargs)
    wrapper.__wrapped__ = func
    return wrapper


def question_key(item: dict) -> tuple:
    return item["db_id"], item["question"], item["query"]


@dataclass
class DailBaseline:
    cfg: Any
    dail: dict
    data: Any
    pre_prompt: Any
    final_prompt: Any
    targets: dict
    schemas: dict
    get_sql_for_database: Callable


def load_dail(cfg) -> DailBaseline:
    """
    Siapkan DAIL sekali per run: dataset + linking, lalu dua objek prompt (pass 1 dan pass 2).
    Membuat objek prompt = meng-encode seluruh pertanyaan train yang sudah di-mask dengan
    all-mpnet-base-v2 (CPU, persis seperti kode DAIL), jadi langkah ini makan waktu beberapa menit.
    """
    dail = import_dail(cfg)
    check_preprocessed(cfg)
    data = _make_dataset(dail, cfg)
    with timing.run_step("dail_load_databases"):
        data.get_databases()
    with timing.run_step("dail_load_dev"):
        dev_targets = data.get_test_json()
    targets = {}
    for target in dev_targets:
        targets.setdefault(question_key(target), target)

    def make_prompt(selector: str):
        cls = dail["prompt_builder"].prompt_factory(cfg.dail_prompt_repr, cfg.dail_k_shot, cfg.dail_example_type, selector)
        prompt = cls(data=data, tokenizer=DAIL_TOKENIZER_NAME)
        # Seleksi contoh dicatat sebagai tahap few_shot (nama sama dengan pipeline kita)
        prompt.get_examples = _timed("few_shot", prompt.get_examples)
        return prompt

    with timing.run_step("dail_index_pass1"):
        pre_prompt = make_prompt(cfg.dail_pre_selector)
    with timing.run_step("dail_index_pass2"):
        final_prompt = make_prompt(cfg.dail_selector)
    # SQLPrompt.format_question() memanggil get_sql_for_database() dari namespace modulnya;
    # dibungkus supaya waktu mengambil CREATE TABLE tercatat sebagai build_ddl.
    repr_module = dail["repr_module"]
    original_ddl = getattr(repr_module.get_sql_for_database, "__wrapped__", repr_module.get_sql_for_database)
    repr_module.get_sql_for_database = _timed("build_ddl", original_ddl)
    schemas = {s["db_id"]: s for s in data.get_table_json()}
    return DailBaseline(cfg, dail, data, pre_prompt, final_prompt, targets, schemas, original_ddl)


def dail_postprocess(dail: DailBaseline, text: str) -> str:
    """Cleaner SQL DAIL, disalin dari ask_llm.py (cabang n == 1, baris 80-90) apa adanya."""
    sql = " ".join(text.replace("\n", " ").split())
    sql = dail.dail["process_duplication"](sql)
    if sql.startswith("SELECT"):
        return sql
    if sql.startswith(" "):
        return "SELECT" + sql
    return "SELECT " + sql


def _generate(prompt: str, llm, tokenizer, cfg) -> tuple[str, int]:
    """Greedy decode, sama dengan generate_sql_with_token_count() tapi teks mentah dikembalikan
    (cleaner DAIL, bukan cleaner pipeline kita). Stop string = mode completion DAIL (cfg.dail_stop_strings)."""
    import torch  # di sini supaya modul ini bisa di-import (dan dites) tanpa torch

    with timing.step("prompt_tokenize"):
        inputs = tokenizer(prompt, return_tensors="pt").to(llm.device)
    stop = {"stop_strings": list(cfg.dail_stop_strings), "tokenizer": tokenizer} if cfg.dail_stop_strings else {}
    with timing.step("llm_generate"), torch.no_grad():
        outputs = llm.generate(
            **inputs,
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.temperature,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            **stop,
        )
    generated = outputs[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True), int(generated.shape[0])


def _one_pass(dail: DailBaseline, prompt_obj, target: dict, name: str, llm, tokenizer) -> dict:
    cfg = dail.cfg
    with timing.step(f"dail_{name}"):
        # build_prompt di sini termasuk build_ddl dan few_shot; finish_latency() mengurangkannya
        with timing.step("build_prompt"):
            formatted = prompt_obj.format(target=target, max_seq_len=cfg.dail_max_seq_len, max_ans_len=cfg.dail_max_ans_len,
                                          scope_factor=cfg.dail_scope_factor, cross_domain=False)
        raw, n_out = _generate(formatted["prompt"], llm, tokenizer, cfg)
        with timing.step("sql_clean"):
            sql = dail_postprocess(dail, raw)
    with timing.step("count_t_in"):
        n_in = len(tokenizer.encode(formatted["prompt"]))
    return {"prompt": formatted["prompt"], "n_examples": formatted["n_examples"], "raw": raw, "sql": sql,
            "n_in": n_in, "n_out": n_out}


def schema_context(dail: DailBaseline, db_id: str) -> str:
    """Blok CREATE TABLE yang masuk prompt DAIL (seluruh DB), untuk menghitung schema_tokens."""
    return "\n\n".join(dail.get_sql_for_database(dail.data.get_path_db(db_id)))


def run_single_dail(item: dict, dail: DailBaseline, llm, tokenizer) -> dict:
    """
    Satu pertanyaan lewat DAIL-SQL: pass 1 (seleksi pertanyaan ter-mask) -> SQL awal -> skeleton ->
    pass 2 (DAIL Selection) -> SQL final. n_in / n_out = jumlah kedua pass, karena DAIL memang
    memanggil LLM dua kali per pertanyaan.
    """
    key = question_key(item)
    if key not in dail.targets:
        raise KeyError(f"Pertanyaan tidak ada di dev.json versi DAIL: {key[:2]}")
    # Salinan dangkal cukup: yang diubah hanya kunci pre_skeleton. deepcopy tidak bisa dipakai karena
    # SqliteTable DAIL (dict dengan __getattr__ = dict.__getitem__) membuat copy.deepcopy() KeyError.
    target = dict(dail.targets[key])
    pass1 = _one_pass(dail, dail.pre_prompt, target, "pass1", llm, tokenizer)
    # Skeleton SQL awal, sama dengan get_pre_skeleton() DAIL. Kalau SQL awal tidak bisa di-parse,
    # skeleton kosong: selector pass 2 lalu jatuh ke urutan kemiripan pertanyaan saja.
    try:
        target["pre_skeleton"] = dail.dail["utils"].sql2skeleton(pass1["sql"], dail.schemas[item["db_id"]])
        skeleton_error = None
    except Exception as exc:  # noqa: BLE001 - dicatat, run tetap lanjut
        target["pre_skeleton"] = ""
        skeleton_error = repr(exc)
    pass2 = _one_pass(dail, dail.final_prompt, target, "pass2", llm, tokenizer)
    return {
        "pred": pass2["sql"],
        "n_in": pass1["n_in"] + pass2["n_in"],
        "n_out": pass1["n_out"] + pass2["n_out"],
        "pre_sql": pass1["sql"],
        "pre_skeleton": target["pre_skeleton"],
        "pre_skeleton_error": skeleton_error,
        "passes": {name: {k: p[k] for k in ("n_in", "n_out", "n_examples")} for name, p in (("pass1", pass1), ("pass2", pass2))},
        "prompts": {"pass1": pass1["prompt"], "pass2": pass2["prompt"]},
        "raw": {"pass1": pass1["raw"], "pass2": pass2["raw"]},
    }


def finish_latency(steps: dict | None) -> dict | None:
    """
    build_prompt dicatat mengelilingi format() DAIL, yang di dalamnya juga mencatat build_ddl dan
    few_shot. Supaya end_to_end tidak menghitungnya dua kali (dan build_prompt berarti hal yang sama
    dengan di pipeline kita), keduanya dikurangkan dulu, baru timing.finish_query().
    """
    if steps is None:
        return None
    steps = dict(steps)
    steps["build_prompt"] = max(0.0, steps.get("build_prompt", 0.0) - steps.get("build_ddl", 0.0) - steps.get("few_shot", 0.0))
    return timing.finish_query(steps)
