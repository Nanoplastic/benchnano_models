"""Ablação nano vs misto com max_det=2000 / Nano vs mixed ablation with max_det=2000.

PT: Mesma comparação de 59_compare_nano_ablation_by_label.py, lendo os CSVs gerados por
    77_reeval_cv_folds_maxdet2000.py (max_det=300 e max_det=2000) em vez dos de 51/58.
    Checagem: a condição max_det=300 deve reproduzir o resultado de
    59_compare_nano_ablation_by_label.py (diferença +6,02 pp, IC 95% [-1,49; +14,45]; por
    polímero: PE +9,89, PET +16,03, PP -4,90, PS +18,05).
    Uso:
    python3 78_compare_nano_ablation_maxdet2000.py
EN: Same comparison as 59_compare_nano_ablation_by_label.py, reading the CSVs produced by
    77_reeval_cv_folds_maxdet2000.py (max_det=300 and max_det=2000) instead of those from
    51/58.
    Check: the max_det=300 condition must reproduce the result of
    59_compare_nano_ablation_by_label.py (difference +6.02 pp, 95% CI [-1.49, +14.45]; per
    polymer: PE +9.89, PET +16.03, PP -4.90, PS +18.05).
    Usage:
    python3 78_compare_nano_ablation_maxdet2000.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import RAIZ_RESULTADOS, salvar_resumo_experimento, configurar_log_experimento

NOME_EXP = "78_compare_nano_ablation_maxdet2000"
DIR_EXP, CAMINHO_LOG, REGISTRADOR, METADADOS_EXECUCAO = configurar_log_experimento(NOME_EXP, __file__)

RELATORIOS_CV77 = RAIZ_RESULTADOS / "77_reeval_cv_folds_maxdet2000" / "reports"
NOMES_CLASSES = ["PE", "PET", "PP", "PS"]
REGIME_TAMANHO = "NANOPLASTIC"
N_BOOTSTRAP = 10_000
SEMENTE = 42
MAX_DETS_TESTADOS = (300, 2000)

REFERENCIA_GERAL = {"diff_point": 0.06021917225236728,
                     "ci95": [-0.014947563633692776, 0.14452473185594536]}
REFERENCIA_POR_ROTULO = {
    "PE": 0.09890109890109888, "PET": 0.1603010872595484,
    "PP": -0.048958794860434196, "PS": 0.18048780487804877,
}
TOL_CHECAGEM = 0.002


def carregar(caminho: Path) -> pd.DataFrame:
    if not caminho.exists():
        raise SystemExit(f"{caminho} nao existe -- rode 77_reeval_cv_folds_maxdet2000.py antes")
    df = pd.read_csv(caminho)
    df = df[df["size"] == REGIME_TAMANHO].copy()
    df["label"] = df["source_image"].str.split("_").str[0]
    df["matched"] = df["matched"].astype(str).str.lower().isin(("true", "1"))
    return df


def recall(df: pd.DataFrame) -> float:
    return float(df["matched"].mean()) if len(df) else float("nan")


def ic_diferenca_pareada(a: pd.DataFrame, b: pd.DataFrame, semente=SEMENTE, n_bootstrap=N_BOOTSTRAP):
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
    return {"diff_point": recall(a) - recall(b), "ci95": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0), "n_images": len(imgs)}


def executar(max_det):
    misto = carregar(RELATORIOS_CV77 / f"cv_pooled_predictions_mixed_maxdet{max_det}.csv")
    nano = carregar(RELATORIOS_CV77 / f"cv_pooled_predictions_nano_only_maxdet{max_det}.csv")

    geral = {"mixed_recall": recall(misto), "nano_only_recall": recall(nano),
               "n_reference": int(len(misto)), **ic_diferenca_pareada(misto, nano)}

    by_label = {}
    for rot_lab in NOMES_CLASSES:
        m, nn = misto[misto["label"] == rot_lab], nano[nano["label"] == rot_lab]
        if not len(m):
            continue
        by_label[rot_lab] = {"n_reference": int(len(m)), "mixed_recall": recall(m),
                         "nano_only_recall": recall(nn), "diff_point": recall(m) - recall(nn)}
    return geral, by_label


def principal():
    resultados = {}
    for md in MAX_DETS_TESTADOS:
        print(f"\n{'='*72}\nmax_det={md}\n{'='*72}")
        geral, by_label = executar(md)
        resultados[md] = (geral, by_label)
        print(f"  misto {geral['mixed_recall']*100:6.2f}%  nano-only {geral['nano_only_recall']*100:6.2f}%  "
              f"dif {geral['diff_point']*100:+6.2f}pp  IC95=[{geral['ci95'][0]*100:+.2f}, {geral['ci95'][1]*100:+.2f}]")
        for rot_lab, r in by_label.items():
            print(f"    {rot_lab:5s} {r['mixed_recall']*100:6.2f}% / {r['nano_only_recall']*100:6.2f}%  "
                  f"dif {r['diff_point']*100:+6.2f}pp")

    print(f"\n{'='*72}\nSANIDADE -- max_det=300 reproduz 59_compare_nano_ablation_by_label?\n{'='*72}")
    ov300, bl300 = resultados[300]
    ok = True
    d = abs(ov300["diff_point"] - REFERENCIA_GERAL["diff_point"])
    dinf = abs(ov300["ci95"][0] - REFERENCIA_GERAL["ci95"][0])
    dsup = abs(ov300["ci95"][1] - REFERENCIA_GERAL["ci95"][1])
    if max(d, dinf, dsup) > TOL_CHECAGEM:
        ok = False
    print(f"  overall   referência={REFERENCIA_GERAL['diff_point']*100:+.2f}pp "
          f"[{REFERENCIA_GERAL['ci95'][0]*100:+.2f},{REFERENCIA_GERAL['ci95'][1]*100:+.2f}]  "
          f"obtido={ov300['diff_point']*100:+.2f}pp [{ov300['ci95'][0]*100:+.2f},{ov300['ci95'][1]*100:+.2f}]")
    for rot_lab, pub in REFERENCIA_POR_ROTULO.items():
        obt = bl300[rot_lab]["diff_point"]
        dd = abs(obt - pub)
        if dd > TOL_CHECAGEM:
            ok = False
        print(f"  {rot_lab:5s}     referência={pub*100:+.2f}pp  obtido={obt*100:+.2f}pp  dif={dd*100:.3f}pp")
    print("\n  ->", "SIM" if ok else "NAO")
    if not ok:
        print("\n  !! nao reproduz -- o resultado de max_det=2000 NAO deve ser usado.")
        return

    ov2000, bl2000 = resultados[2000]
    salvar_resumo_experimento(NOME_EXP, METADADOS_EXECUCAO, {
        "note": "comparacao CV-vs-CV (misto vs nano-only) com max_det explicito -- "
                "alimenta o paragrafo da ablacao nano-only na Discussao",
        "sanity_reproduces_published": ok,
        "maxdet_300": {"overall": ov300, "by_polymer_label": bl300},
        "maxdet_2000": {"overall": ov2000, "by_polymer_label": bl2000},
    })
    print(f"\nLog completo: {CAMINHO_LOG}")


if __name__ == "__main__":
    principal()
