"""
dim6_ablation.py — Dimensi 6: Ablation Few-Shot

Referensi: EVALUATION_ANALYSIS_GUIDE.md Bagian 3, "Dimensi 6".

KEPUTUSAN (2026-08-26, lihat context/IMPLEMENTATION_DECISIONS.md poin 15):
guide minta "baca pola transisi EX vs k" tanpa angka threshold untuk
"signifikan" -- peneliti mengonfirmasi algoritma greedy/step-wise: mulai dari
k terkecil, TERUS naik ke k berikutnya selama lompatan EX ke k tersebut >=
STEP_THRESHOLD (poin persen). Begitu satu langkah gagal signifikan (termasuk
kalau EX-nya malah TURUN), berhenti di k SEBELUM langkah itu -- tidak lanjut
lagi walau ada langkah signifikan lagi setelahnya (sesuai semangat "diminishing
returns", bukan mencari titik optimal presisi matematis -- guide eksplisit
"bukan cuma ambil EX tertinggi mentah-mentah").

STEP_THRESHOLD = 2 poin persen, konsisten dengan threshold delta QVT yang
sudah WAJIB di guide Dimensi 4 (±2%) -- skala yang sama dipakai ulang, bukan
angka baru yang berdiri sendiri.

CATATAN: ex_per_k HARUS diisi dari hasil run nyata ablation.py + evaluation.py
(lihat src/experiments/ablation.py) -- modul ini murni menginterpretasi angka
yang sudah ada, tidak menjalankan eksperimen apapun sendiri.
"""

from typing import Dict, List, Optional

STEP_THRESHOLD = 2.0  # poin persen -- lihat catatan KEPUTUSAN di atas


def run_dimension_6(ex_per_k: Dict[int, float]) -> dict:
    """
    ex_per_k: {0: EX_persen, 1: EX_persen, 3: EX_persen, 5: EX_persen} (skala 0-100,
    konsisten dengan esm_ex_cm.aggregate_ex()'s convention).

    Return dict berisi tabel EX vs k, breakdown tiap langkah transisi, k_final,
    label pola, dan alasan dalam bahasa natural.
    """
    if not ex_per_k:
        raise ValueError("ex_per_k kosong -- Dimensi 6 butuh minimal satu nilai k untuk dianalisis.")

    ks = sorted(ex_per_k.keys())

    if len(ks) == 1:
        only_k = ks[0]
        return {
            "table": dict(ex_per_k),
            "steps": [],
            "k_final": only_k,
            "pattern": "single_k",
            "reasoning": (
                f"Cuma ada satu nilai k ({only_k}) di data -- tidak ada transisi "
                f"untuk dianalisis, k_final = {only_k} by default."
            ),
        }

    steps: List[dict] = []
    k_final = ks[0]
    stop_index: Optional[int] = None  # index langkah pertama yang GAGAL signifikan
    for i in range(len(ks) - 1):
        k_from, k_to = ks[i], ks[i + 1]
        delta = ex_per_k[k_to] - ex_per_k[k_from]
        significant = delta >= STEP_THRESHOLD
        steps.append({"from_k": k_from, "to_k": k_to, "delta": delta, "significant": significant})
        if stop_index is None:
            if significant:
                k_final = k_to  # terus lanjut
            else:
                stop_index = i  # berhenti di sini -- k_final sudah final (= ks[i])

    if stop_index is None:
        # Semua langkah signifikan sampai k terbesar -- EX terus naik jauh, belum
        # ada tanda plateau dalam rentang k yang diuji. Guide tidak punya nama
        # resmi untuk pola ini (cuma 3 pola dideskripsikan) -- dilabeli jujur,
        # bukan dipaksakan ke salah satu dari 3 pola resmi.
        pattern = "terus_naik_signifikan"
        reasoning = (
            f"EX terus naik signifikan (>= {STEP_THRESHOLD} poin) di SETIAP transisi "
            f"k sampai k={ks[-1]} -- belum ada tanda diminishing returns dalam rentang "
            f"k yang diuji. k_final = {k_final} (k terbesar yang diuji), tapi "
            f"pertimbangkan menguji k lebih besar dari {ks[-1]} kalau memungkinkan."
        )
    else:
        stopping_step = steps[stop_index]
        if stopping_step["delta"] < 0:
            pattern = "context_overload"
            reasoning = (
                f"EX justru TURUN {abs(stopping_step['delta']):.2f} poin dari k={stopping_step['from_k']} "
                f"ke k={stopping_step['to_k']} -- indikasi context overload/over-prompting. "
                f"k_final = {k_final} (k terakhir sebelum penurunan)."
            )
        elif stop_index == 0:
            pattern = "konsisten_tidak_bergantung_k"
            reasoning = (
                f"Perubahan EX kecil ({stopping_step['delta']:+.2f} poin, di bawah threshold "
                f"{STEP_THRESHOLD} poin) bahkan di transisi PERTAMA (k=0 -> k=1) -- sistem tidak "
                f"terlalu bergantung pada jumlah contoh. k_final = {k_final} (k paling murah)."
            )
        else:
            pattern = "diminishing_returns"
            reasoning = (
                f"EX naik signifikan sampai k={k_final}, lalu mendatar (transisi "
                f"k={stopping_step['from_k']} -> k={stopping_step['to_k']} cuma "
                f"{stopping_step['delta']:+.2f} poin, di bawah threshold {STEP_THRESHOLD} poin) "
                f"-- diminishing returns. k_final = {k_final}."
            )

    return {
        "table": dict(ex_per_k),
        "steps": steps,
        "k_final": k_final,
        "pattern": pattern,
        "reasoning": reasoning,
    }


def print_dimension_6(result: dict) -> None:
    print("=" * 70)
    print("Dimensi 6 -- Ablation Few-Shot (EX vs k)")
    print("=" * 70)
    for k in sorted(result["table"].keys()):
        print(f"  k={k:<3} EX={result['table'][k]:.2f}%")
    if result["steps"]:
        print()
        print("Transisi:")
        for s in result["steps"]:
            flag = "signifikan" if s["significant"] else "TIDAK signifikan"
            print(f"  k={s['from_k']} -> k={s['to_k']}: delta={s['delta']:+.2f} poin ({flag})")
    print()
    print(f"Pola: {result['pattern']}")
    print(f"k_final: {result['k_final']}")
    print(f"Alasan: {result['reasoning']}")
