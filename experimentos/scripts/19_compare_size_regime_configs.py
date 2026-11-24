"""Comparação das configurações de regime de tamanho / Size regime configuration comparison.

PT: Compara as três configurações de regime de tamanho (zero-shot, LoRA, ajuste fino
    completo) com o mesmo método estatístico de 11_compare_experiments_statistical.py:
    - IC 95% da acurácia de cada configuração via bootstrap por imagem de
    origem (partículas da mesma imagem não são independentes).
    - McNemar exato pareado em cada par (zero-shot x
    LoRA, zero-shot x completo, LoRA x completo).
    - Correção de Bonferroni para as 3 comparações: alfa_corrigido = 0,05/3 ≈ 0,0167.
    Requer as predições de teste geradas por 16, 17 e 18.
EN: Compares the three size regime configurations (zero-shot, LoRA, full fine-tuning) with
    the same statistical method as 11_compare_experiments_statistical.py:
    - 95% CI of each configuration's accuracy via source-image bootstrap
    (particles from the same image are not independent).
    - Paired exact McNemar on each pair (zero-shot x
    LoRA, zero-shot x full, LoRA x full).
    - Bonferroni correction for the 3 comparisons: corrected alpha = 0.05/3 ≈ 0.0167.
    Requires the test predictions produced by 16, 17 and 18.
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import DIR_RELATORIOS, RAIZ_RESULTADOS

CONFIGURACOES = {
    "zero-shot": "size_zeroshot_real_test",
    "LoRA": "size_lora_real_test",
    "full fine-tuning": "size_full_ft_real_test",
}
MANIFESTO = RAIZ_RESULTADOS / "15_build_size_regime_manifest" / "data" / "manifest_size_regime.csv"
ALFA = 0.05
N_BOOTSTRAP = 10_000
SEMENTE = 42


def carregar_predicoes(rotulo_exp: str) -> pd.DataFrame:
    caminho = DIR_RELATORIOS / f"{rotulo_exp}_predictions.csv"
    if not caminho.exists():
        raise FileNotFoundError(
            f"Não encontrei {caminho}. Rode primeiro o experimento correspondente "
            f"(16/17/18) até ele gerar essa predição de teste."
        )
    df = pd.read_csv(caminho)
    df["correct"] = df["correct"].astype(bool)
    return df[["image_path", "true_label", "pred_label", "correct"]]


def ic_bootstrap_por_imagem_origem(df: pd.DataFrame, manifesto: pd.DataFrame,
                                  n_bootstrap: int = N_BOOTSTRAP, alfa: float = ALFA,
                                  semente: int = SEMENTE) -> dict:
    mesclado = df.merge(manifesto[["image_path", "source_image"]], on="image_path", how="left")
    if mesclado["source_image"].isna().any():
        raise ValueError("Algum image_path não foi encontrado no manifest_size_regime.csv.")

    gerador = np.random.default_rng(semente)
    imagens_origem = mesclado["source_image"].unique()
    grupos = {s: g for s, g in mesclado.groupby("source_image")}
    acurs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        amostrados = gerador.choice(imagens_origem, size=len(imagens_origem), replace=True)
        linhas_tab = pd.concat([grupos[s] for s in amostrados], ignore_index=True)
        acurs[i] = linhas_tab["correct"].mean()

    ic_inf, ic_sup = np.percentile(acurs, [100 * alfa / 2, 100 * (1 - alfa / 2)])
    return {"n_source_images": len(imagens_origem), "ci_low": ic_inf, "ci_high": ic_sup}


def mcnemar_exato(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
    mesclado = df_a.merge(df_b, on="image_path", suffixes=("_a", "_b"), validate="one_to_one")
    so_a = int(((mesclado["correct_a"]) & (~mesclado["correct_b"])).sum())
    so_b = int(((~mesclado["correct_a"]) & (mesclado["correct_b"])).sum())
    n_discordantes = so_a + so_b
    valor_p = 1.0 if n_discordantes == 0 else stats.binomtest(
        min(so_a, so_b), n_discordantes, p=0.5, alternative="two-sided"
    ).pvalue
    return {"n": len(mesclado), "a_only": so_a, "b_only": so_b,
            "n_discordant": n_discordantes, "p_value": valor_p}


def principal():
    manifesto = pd.read_csv(MANIFESTO)

    print("=== Acurácia por configuração, com IC 95% (bootstrap por imagem-fonte) ===\n")
    preds = {}
    for nome, rotulo_exp in CONFIGURACOES.items():
        df = carregar_predicoes(rotulo_exp)
        preds[nome] = df
        acur = df["correct"].mean()
        ci = ic_bootstrap_por_imagem_origem(df, manifesto)
        print(f"{nome:18s} n={len(df):4d}  acc={acur:.3f}  "
              f"IC95%=[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]  "
              f"({ci['n_source_images']} imagens-fonte, {N_BOOTSTRAP} réplicas)")

    n_pares = len(list(combinations(CONFIGURACOES.keys(), 2)))
    alfa_corrigido = ALFA / n_pares
    print(f"\n=== McNemar exato pareado, {n_pares} pares -- correção de Bonferroni "
          f"(alpha={ALFA}/{n_pares}={alfa_corrigido:.4f}) ===\n")

    linhas_tab = []
    for nome_a, nome_b in combinations(CONFIGURACOES.keys(), 2):
        res = mcnemar_exato(preds[nome_a], preds[nome_b])
        significativo = res["p_value"] < alfa_corrigido
        print(f"{nome_a} vs {nome_b}: n={res['n']} discordantes={res['n_discordant']} "
              f"({nome_a} certo/{nome_b} errado={res['a_only']}, o inverso={res['b_only']}) "
              f"p={res['p_value']:.4f}  significativo (Bonferroni)? {'SIM' if significativo else 'não'}")
        linhas_tab.append({
            "config_a": nome_a, "config_b": nome_b, **res,
            "alpha_corrected": alfa_corrigido, "significant_bonferroni": significativo,
        })

    caminho_saida = DIR_RELATORIOS / "compare_size_regime_configs.csv"
    pd.DataFrame(linhas_tab).to_csv(caminho_saida, index=False)
    print(f"\nResumo salvo em {caminho_saida}")


if __name__ == "__main__":
    principal()
