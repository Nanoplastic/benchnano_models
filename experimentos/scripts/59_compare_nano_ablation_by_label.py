"""Ablação nano vs treino misto por rótulo / Nano ablation vs mixed training by label.

PT: Compara a ablação só com nanopartículas com o treino misto na MESMA validação cruzada,
    no agregado e por rótulo de polímero. O teste fixo (15 micrografias) é uma amostra
    favorável (recall geral de 43,0% contra 23,8% na validação cruzada), então a comparação
    usa validação cruzada nas duas condições.
    Não roda inferência: usa as predições por partícula já gravadas por
    51_eval_cv_folds_nano.py -> dobras só nano
    58_eval_cv_folds_mixed_by_regime.py -> dobras mistas
    Ambas usam as mesmas 5 dobras, as mesmas imagens held-out, imgsz=1024, conf=0,15 e
    IoU=0,5, então as duas condições compartilham o mesmo conjunto de referência.
    O bootstrap da diferença reamostra micrografias de origem (partículas de uma imagem
    ficam juntas), 10.000 réplicas, semente 42, como em 45_eval_cv_folds.py e
    51_eval_cv_folds_nano.py.
    Uso:
    python3 59_compare_nano_ablation_by_label.py
EN: Compares the nano-only ablation with mixed training on the SAME cross-validation,
    overall and per polymer label. The fixed test set (15 micrographs) is a favourable
    sample (overall recall 43.0% vs 23.8% in cross-validation), so the comparison uses
    cross-validation for both conditions.
    Runs no inference: uses the per-particle predictions already written by
    51_eval_cv_folds_nano.py -> nano-only folds
    58_eval_cv_folds_mixed_by_regime.py -> mixed folds
    Both use the same 5 folds, the same held-out images, imgsz=1024, conf=0.15 and IoU=0.5,
    so both conditions share the same reference set.
    The bootstrap of the difference resamples source micrographs (particles from one image
    stay together), 10,000 replicates, seed 42, as in 45_eval_cv_folds.py and
    51_eval_cv_folds_nano.py.
    Usage:
    python3 59_compare_nano_ablation_by_label.py
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import RAIZ_RESULTADOS, salvar_resumo_experimento, configurar_log_experimento

NOME_EXP = "59_compare_nano_ablation_by_label"
DIR_EXP, CAMINHO_LOG, REGISTRADOR, METADADOS_EXECUCAO = configurar_log_experimento(NOME_EXP, __file__)

FONTES = {
    "nano_only": RAIZ_RESULTADOS / "51_eval_cv_folds_nano" / "reports" / "cv_pooled_predictions_nano.csv",
    "mixed": RAIZ_RESULTADOS / "58_eval_cv_folds_mixed_by_regime" / "reports" / "cv_pooled_predictions_mixed_by_regime.csv",
}
NOMES_CLASSES = ["PE", "PET", "PP", "PS"]
REGIME_TAMANHO = "NANOPLASTIC"
N_BOOTSTRAP = 10_000
SEMENTE = 42


def carregar(caminho: Path) -> pd.DataFrame:
    if not caminho.exists():
        raise SystemExit(f"{caminho} nao existe -- rode o script que a gera antes")
    df = pd.read_csv(caminho)
    df = df[df["size"] == REGIME_TAMANHO].copy()
    # PT: o prefixo do nome do arquivo é o rótulo de polímero (ex.: PS_PS-100x.png)
    # EN: the file name prefix is the polymer label (e.g. PS_PS-100x.png)
    df["label"] = df["source_image"].str.split("_").str[0]
    df["matched"] = df["matched"].astype(str).str.lower().isin(("true", "1"))
    return df


def recall(df: pd.DataFrame) -> float:
    return float(df["matched"].mean()) if len(df) else float("nan")


def ic_diferenca_pareada(a: pd.DataFrame, b: pd.DataFrame, semente=SEMENTE, n_bootstrap=N_BOOTSTRAP):
    """PT: IC da diferença de recall (a - b), reamostrando micrografias de origem.
    EN: CI of the recall difference (a - b), resampling source micrographs.
    """
    imgs = sorted(set(a["source_image"]) & set(b["source_image"]))
    ga = {s: g for s, g in a.groupby("source_image")}
    gb = {s: g for s, g in b.groupby("source_image")}
    gerador = np.random.default_rng(semente)
    diferencas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        amostrados = gerador.choice(imgs, size=len(imgs), replace=True)
        ra = np.concatenate([ga[s]["matched"].values for s in amostrados]).mean()
        rb = np.concatenate([gb[s]["matched"].values for s in amostrados]).mean()
        diferencas[i] = ra - rb
    lo, hi = np.nanpercentile(diferencas, [2.5, 97.5])
    return {"diff_point": recall(a) - recall(b),
            "ci95": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0),
            "n_images": len(imgs)}


def principal():
    misto = carregar(FONTES["mixed"])
    nano = carregar(FONTES["nano_only"])

    n_m, n_n = len(misto), len(nano)
    if n_m != n_n:
        REGISTRADOR.warning("conjuntos de referencia diferem: misto=%d nano-only=%d", n_m, n_n)
    REGISTRADOR.info("referencias %s: misto=%d nano-only=%d", REGIME_TAMANHO, n_m, n_n)

    geral = {
        "mixed_recall": recall(misto),
        "nano_only_recall": recall(nano),
        "n_reference": int(n_m),
        **ic_diferenca_pareada(misto, nano),
    }
    print(f"\n=== recall {REGIME_TAMANHO}, 5 folds de validacao cruzada, "
          f"{geral['n_reference']} referencias em {geral['n_images']} micrografias ===")
    print(f"  treino misto      {geral['mixed_recall']*100:6.2f}%")
    print(f"  treino nano-only  {geral['nano_only_recall']*100:6.2f}%")
    print(f"  diferenca         {geral['diff_point']*100:+6.2f} pp  "
          f"IC95=[{geral['ci95'][0]*100:+.2f}, {geral['ci95'][1]*100:+.2f}]  "
          f"exclui zero? {'SIM' if geral['excludes_zero'] else 'NAO'}")

    by_label = {}
    print(f"\n=== por rotulo de polimero ===")
    print(f"  {'label':6s} {'n':>6s} {'misto':>8s} {'nano-only':>10s} {'dif':>8s}")
    for rot_lab in NOMES_CLASSES:
        m, nn = misto[misto["label"] == rot_lab], nano[nano["label"] == rot_lab]
        if not len(m):
            continue
        by_label[rot_lab] = {"n_reference": int(len(m)),
                         "mixed_recall": recall(m),
                         "nano_only_recall": recall(nn),
                         "diff_point": recall(m) - recall(nn)}
        r = by_label[rot_lab]
        print(f"  {rot_lab:6s} {r['n_reference']:6d} {r['mixed_recall']*100:7.2f}% "
              f"{r['nano_only_recall']*100:9.2f}% {r['diff_point']*100:+7.2f} pp")

    salvar_resumo_experimento(NOME_EXP, METADADOS_EXECUCAO, {
        "note": "resultado exploratorio/complementar -- par correto (CV vs CV) para a "
                "ablacao nano-only; substitui a comparacao entre 15,0% (CV) e "
                "31,1% (teste travado), que usa protocolos de avaliacao diferentes",
        "regime": REGIME_TAMANHO,
        "sources": {k: str(v) for k, v in FONTES.items()},
        "overall": geral,
        "by_polymer_label": by_label,
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEMENTE,
    })
    print(f"\nLog completo: {CAMINHO_LOG}")


if __name__ == "__main__":
    principal()
