"""Copia fiel de 59_compare_nano_ablation_by_label.py, lendo os CSVs gerados
por 77_reeval_cv_folds_maxdet2000.py (max_det=300 e max_det=2000) em vez dos
publicados por 51/58.

SANIDADE: a condicao max_det=300 deve reproduzir o publicado por
59_compare_nano_ablation_by_label.py (diferenca +6.02pp, IC95
[-1.49, +14.45], por polimero: PE +9.89, PET +16.03, PP -4.90, PS +18.05).

Alimenta mainV11.tex Discussao/RQ1, paragrafo da ablacao nano-only.

Uso:
    python3 78_compare_nano_ablation_maxdet2000.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import RESULTS_ROOT, save_experiment_summary, setup_experiment_logging

EXP_NAME = "78_compare_nano_ablation_maxdet2000"
EXP_DIR, LOG_PATH, LOGGER, RUNTIME_METADATA = setup_experiment_logging(EXP_NAME, __file__)

CV77_REPORTS = RESULTS_ROOT / "77_reeval_cv_folds_maxdet2000" / "reports"
CLASS_NAMES = ["PE", "PET", "PP", "PS"]
REGIME = "NANOPLASTIC"
N_BOOTSTRAP = 10_000
SEED = 42
MAX_DETS = (300, 2000)

PUBLISHED_OVERALL = {"diff_point": 0.06021917225236728,
                     "ci95": [-0.014947563633692776, 0.14452473185594536]}
PUBLISHED_BY_LABEL = {
    "PE": 0.09890109890109888, "PET": 0.1603010872595484,
    "PP": -0.048958794860434196, "PS": 0.18048780487804877,
}
SANITY_TOL = 0.002


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"{path} nao existe -- rode 77_reeval_cv_folds_maxdet2000.py antes")
    df = pd.read_csv(path)
    df = df[df["size"] == REGIME].copy()
    df["label"] = df["source_image"].str.split("_").str[0]
    df["matched"] = df["matched"].astype(str).str.lower().isin(("true", "1"))
    return df


def recall(df: pd.DataFrame) -> float:
    return float(df["matched"].mean()) if len(df) else float("nan")


def paired_diff_ci(a: pd.DataFrame, b: pd.DataFrame, seed=SEED, n_bootstrap=N_BOOTSTRAP):
    imgs = sorted(set(a["source_image"]) & set(b["source_image"]))
    ga = {s: g for s, g in a.groupby("source_image")}
    gb = {s: g for s, g in b.groupby("source_image")}
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sampled = rng.choice(imgs, size=len(imgs), replace=True)
        ra = np.concatenate([ga[s]["matched"].values for s in sampled]).mean()
        rb = np.concatenate([gb[s]["matched"].values for s in sampled]).mean()
        diffs[i] = ra - rb
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return {"diff_point": recall(a) - recall(b), "ci95": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0), "n_images": len(imgs)}


def run(max_det):
    mixed = load(CV77_REPORTS / f"cv_pooled_predictions_mixed_maxdet{max_det}.csv")
    nano = load(CV77_REPORTS / f"cv_pooled_predictions_nano_only_maxdet{max_det}.csv")

    overall = {"mixed_recall": recall(mixed), "nano_only_recall": recall(nano),
               "n_reference": int(len(mixed)), **paired_diff_ci(mixed, nano)}

    by_label = {}
    for lab in CLASS_NAMES:
        m, nn = mixed[mixed["label"] == lab], nano[nano["label"] == lab]
        if not len(m):
            continue
        by_label[lab] = {"n_reference": int(len(m)), "mixed_recall": recall(m),
                         "nano_only_recall": recall(nn), "diff_point": recall(m) - recall(nn)}
    return overall, by_label


def main():
    results = {}
    for md in MAX_DETS:
        print(f"\n{'='*72}\nmax_det={md}\n{'='*72}")
        overall, by_label = run(md)
        results[md] = (overall, by_label)
        print(f"  misto {overall['mixed_recall']*100:6.2f}%  nano-only {overall['nano_only_recall']*100:6.2f}%  "
              f"dif {overall['diff_point']*100:+6.2f}pp  IC95=[{overall['ci95'][0]*100:+.2f}, {overall['ci95'][1]*100:+.2f}]")
        for lab, r in by_label.items():
            print(f"    {lab:5s} {r['mixed_recall']*100:6.2f}% / {r['nano_only_recall']*100:6.2f}%  "
                  f"dif {r['diff_point']*100:+6.2f}pp")

    print(f"\n{'='*72}\nSANIDADE -- max_det=300 reproduz 59_compare_nano_ablation_by_label?\n{'='*72}")
    ov300, bl300 = results[300]
    ok = True
    d = abs(ov300["diff_point"] - PUBLISHED_OVERALL["diff_point"])
    dlo = abs(ov300["ci95"][0] - PUBLISHED_OVERALL["ci95"][0])
    dhi = abs(ov300["ci95"][1] - PUBLISHED_OVERALL["ci95"][1])
    if max(d, dlo, dhi) > SANITY_TOL:
        ok = False
    print(f"  overall   publicado={PUBLISHED_OVERALL['diff_point']*100:+.2f}pp "
          f"[{PUBLISHED_OVERALL['ci95'][0]*100:+.2f},{PUBLISHED_OVERALL['ci95'][1]*100:+.2f}]  "
          f"obtido={ov300['diff_point']*100:+.2f}pp [{ov300['ci95'][0]*100:+.2f},{ov300['ci95'][1]*100:+.2f}]")
    for lab, pub in PUBLISHED_BY_LABEL.items():
        obt = bl300[lab]["diff_point"]
        dd = abs(obt - pub)
        if dd > SANITY_TOL:
            ok = False
        print(f"  {lab:5s}     publicado={pub*100:+.2f}pp  obtido={obt*100:+.2f}pp  dif={dd*100:.3f}pp")
    print("\n  ->", "SIM" if ok else "NAO")
    if not ok:
        print("\n  !! nao reproduz -- o resultado de max_det=2000 NAO deve ser usado.")
        return

    ov2000, bl2000 = results[2000]
    save_experiment_summary(EXP_NAME, RUNTIME_METADATA, {
        "note": "comparacao CV-vs-CV (misto vs nano-only) com max_det explicito -- "
                "alimenta o paragrafo da ablacao nano-only na Discussao",
        "sanity_reproduces_published": ok,
        "maxdet_300": {"overall": ov300, "by_polymer_label": bl300},
        "maxdet_2000": {"overall": ov2000, "by_polymer_label": bl2000},
    })
    print(f"\nLog completo: {LOG_PATH}")


if __name__ == "__main__":
    main()
