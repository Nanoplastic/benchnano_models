"""Compara a ablacao nano-only com o treino misto na MESMA validacao cruzada,
no agregado e por rotulo de polimero.

O artigo compara hoje os 15,0% da ablacao (validacao cruzada, 9.399 nano, 79
micrografias) com os 31,1% do detector misto (teste travado, 2.630 nano, 14
micrografias). Sao protocolos de avaliacao diferentes, e o teste travado e
reconhecido no proprio artigo como um sorteio favoravel (43,0% contra 23,8%
no recall geral). O par correto e CV contra CV, que e o que este script mede.

Nao roda inferencia: reaproveita as predicoes por particula ja gravadas por
    51_eval_cv_folds_nano.py          -> folds nano-only
    58_eval_cv_folds_mixed_by_regime.py -> folds mistos
Ambas usam os mesmos 5 folds, as mesmas imagens held-out, imgsz=1024,
conf=0.15 e IoU=0.5, entao as duas condicoes compartilham um unico conjunto
de referencia.

O bootstrap da diferenca reamostra micrografias-fonte (particulas de uma
imagem ficam juntas), 10.000 replicas, seed 42 -- mesma convencao de
45_eval_cv_folds.py e 51_eval_cv_folds_nano.py.

Uso:
    python3 59_compare_nano_ablation_by_label.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import RESULTS_ROOT, save_experiment_summary, setup_experiment_logging

EXP_NAME = "59_compare_nano_ablation_by_label"
EXP_DIR, LOG_PATH, LOGGER, RUNTIME_METADATA = setup_experiment_logging(EXP_NAME, __file__)

SOURCES = {
    "nano_only": RESULTS_ROOT / "51_eval_cv_folds_nano" / "reports" / "cv_pooled_predictions_nano.csv",
    "mixed": RESULTS_ROOT / "58_eval_cv_folds_mixed_by_regime" / "reports" / "cv_pooled_predictions_mixed_by_regime.csv",
}
CLASS_NAMES = ["PE", "PET", "PP", "PS"]
REGIME = "NANOPLASTIC"
N_BOOTSTRAP = 10_000
SEED = 42


def load(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"{path} nao existe -- rode o script que a gera antes")
    df = pd.read_csv(path)
    df = df[df["size"] == REGIME].copy()
    # o prefixo do nome do arquivo e o rotulo de polimero (ex.: PS_PS-100x.png)
    df["label"] = df["source_image"].str.split("_").str[0]
    df["matched"] = df["matched"].astype(str).str.lower().isin(("true", "1"))
    return df


def recall(df: pd.DataFrame) -> float:
    return float(df["matched"].mean()) if len(df) else float("nan")


def paired_diff_ci(a: pd.DataFrame, b: pd.DataFrame, seed=SEED, n_bootstrap=N_BOOTSTRAP):
    """IC da diferenca de recall (a - b), reamostrando micrografias-fonte."""
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
    return {"diff_point": recall(a) - recall(b),
            "ci95": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0),
            "n_images": len(imgs)}


def main():
    mixed = load(SOURCES["mixed"])
    nano = load(SOURCES["nano_only"])

    n_m, n_n = len(mixed), len(nano)
    if n_m != n_n:
        LOGGER.warning("conjuntos de referencia diferem: misto=%d nano-only=%d", n_m, n_n)
    LOGGER.info("referencias %s: misto=%d nano-only=%d", REGIME, n_m, n_n)

    overall = {
        "mixed_recall": recall(mixed),
        "nano_only_recall": recall(nano),
        "n_reference": int(n_m),
        **paired_diff_ci(mixed, nano),
    }
    print(f"\n=== recall {REGIME}, 5 folds de validacao cruzada, "
          f"{overall['n_reference']} referencias em {overall['n_images']} micrografias ===")
    print(f"  treino misto      {overall['mixed_recall']*100:6.2f}%")
    print(f"  treino nano-only  {overall['nano_only_recall']*100:6.2f}%")
    print(f"  diferenca         {overall['diff_point']*100:+6.2f} pp  "
          f"IC95=[{overall['ci95'][0]*100:+.2f}, {overall['ci95'][1]*100:+.2f}]  "
          f"exclui zero? {'SIM' if overall['excludes_zero'] else 'NAO'}")

    by_label = {}
    print(f"\n=== por rotulo de polimero ===")
    print(f"  {'label':6s} {'n':>6s} {'misto':>8s} {'nano-only':>10s} {'dif':>8s}")
    for lab in CLASS_NAMES:
        m, nn = mixed[mixed["label"] == lab], nano[nano["label"] == lab]
        if not len(m):
            continue
        by_label[lab] = {"n_reference": int(len(m)),
                         "mixed_recall": recall(m),
                         "nano_only_recall": recall(nn),
                         "diff_point": recall(m) - recall(nn)}
        r = by_label[lab]
        print(f"  {lab:6s} {r['n_reference']:6d} {r['mixed_recall']*100:7.2f}% "
              f"{r['nano_only_recall']*100:9.2f}% {r['diff_point']*100:+7.2f} pp")

    save_experiment_summary(EXP_NAME, RUNTIME_METADATA, {
        "note": "resultado exploratorio/complementar -- par correto (CV vs CV) para a "
                "ablacao nano-only; substitui a comparacao do artigo entre 15,0% (CV) e "
                "31,1% (teste travado), que usa protocolos de avaliacao diferentes",
        "regime": REGIME,
        "sources": {k: str(v) for k, v in SOURCES.items()},
        "overall": overall,
        "by_polymer_label": by_label,
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
    })
    print(f"\nLog completo: {LOG_PATH}")


if __name__ == "__main__":
    main()
