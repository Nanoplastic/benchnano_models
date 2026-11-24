"""Comparações pareadas entre configurações controladas / Paired comparisons between controlled configurations.

PT: Gera a tabela de comparações entre configurações sob as dez repetições controladas (CSV
    e JSON em relatorio_final/).
    Usa os 10 valores por semente agregados por `final_report.py` (via
    `ctrl_common.carregar_todos_resumos_rep` + `ESPEC_METRICAS`). As dez repetições de duas
    configurações são AMOSTRAS INDEPENDENTES (sementes e modelos diferentes), não
    observações pareadas no mesmo conjunto de teste, por isso o teste é o t de Welch
    (variâncias não assumidas iguais) em vez do McNemar usado em
    `experimentos/scripts/19_compare_size_regime_configs.py`.
    Duas famílias de comparação, cada uma com sua  correção de Bonferroni:
    - Classificação de polímero com VLM (família de 3 pares): só real (500M) x real +
    sintético (500M) x ~2B, alfa_corrigido = 0,05/3 ≈ 0,0167.
    - Regime de tamanho, SmolVLM (família de 1 par): LoRA x ajuste completo, alfa = 0,05.
    Para este par também roda o teste U de Mann-Whitney (livre de distribuição) como
    checagem de robustez.
    Para cada par: diferença de médias, IC 95% da diferença (t de Welch), estatística t,
    graus de liberdade de Welch-Satterthwaite, valor p e g de Hedges (d de Cohen com
    correção de viés para amostra pequena).
    Uso:
    python3 compare_ctrl_repetitions.py
    Requer as 10 repetições completas dos 5 experimentos abaixo (mesma pré-condição de
    final_report.py).
EN: Builds the table of comparisons between configurations under the ten controlled
    repetitions (CSV and JSON in relatorio_final/).
    Uses the 10 per-seed values aggregated by `final_report.py` (via
    `ctrl_common.carregar_todos_resumos_rep` + `ESPEC_METRICAS`). The ten repetitions of two
    configurations are INDEPENDENT SAMPLES (different seeds and models), not paired
    observations on the same test set, so the test is Welch's t (unequal variances) instead
    of the McNemar used in `experimentos/scripts/19_compare_size_regime_configs.py`.
    Two comparison families, each with its own  Bonferroni correction:
    - VLM polymer classification (family of 3 pairs): real only (500M) x real + synthetic
    (500M) x ~2B, corrected alpha = 0.05/3 ≈ 0.0167.
    - Size regime, SmolVLM (family of 1 pair): LoRA x full fine-tuning, alpha = 0.05. For
    this pair the Mann-Whitney U test (distribution-free) is also run as a robustness check.
    For each pair: difference of means, 95% CI of the difference (Welch's t), t statistic,
    Welch-Satterthwaite degrees of freedom, p-value and Hedges' g (Cohen's d with
    small-sample bias correction).
    Usage:
    python3 compare_ctrl_repetitions.py
    Requires the 10 complete repetitions of the 5 experiments below (same precondition as
    final_report.py).
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctrl_common as cc

# PT: mesmos extratores de final_report.py, só para os experimentos que entram nas
#     comparações (o YOLO26L principal e o fim a fim pareado não têm uma segunda
#     configuração equivalente)
# EN: same extractors as final_report.py, only for the experiments in the comparisons (the
#     main YOLO26L and the paired end-to-end have no equivalent second configuration)
ESPEC_METRICAS = {
    "02_vlm_polymer_real_only": ("test_acc", lambda m: m["test_acc"]),
    "03_vlm_polymer_real_synth": ("test_acc", lambda m: m["test_acc"]),
    "04_vlm_polymer_bigvlm": ("test_acc", lambda m: m["test_acc"]),
    "05_smolvlm_size_lora": ("test_acc", lambda m: m["test_acc"]),
    "06_smolvlm_size_full": ("test_acc", lambda m: m["test_acc"]),
}

ROTULOS = {
    "02_vlm_polymer_real_only": "Real only LoRA (500M)",
    "03_vlm_polymer_real_synth": "Real+synthetic LoRA (500M)",
    "04_vlm_polymer_bigvlm": "Real only LoRA (~2B)",
    "05_smolvlm_size_lora": "LoRA",
    "06_smolvlm_size_full": "Full fine tuning",
}

# PT: as duas famílias de comparação, na ordem da tabela de saída; cada tupla é
#     (experimento_a, experimento_b)
# EN: the two comparison families, in output table order; each tuple is (experiment_a,
#     experiment_b)
FAMILIA_POLIMERO = [
    ("02_vlm_polymer_real_only", "03_vlm_polymer_real_synth"),
    ("02_vlm_polymer_real_only", "04_vlm_polymer_bigvlm"),
    ("03_vlm_polymer_real_synth", "04_vlm_polymer_bigvlm"),
]
FAMILIA_REGIME_TAMANHO = [
    ("05_smolvlm_size_lora", "06_smolvlm_size_full"),
]


def carregar_valores(experimento: str) -> list:
    nome_metrica, extrator = ESPEC_METRICAS[experimento]
    reps = cc.carregar_todos_resumos_rep(experimento)
    if len(reps) < cc.N_REPS:
        raise RuntimeError(
            f"{experimento}: só {len(reps)}/{cc.N_REPS} repetições encontradas. "
            f"Rode a fila completa (run_queue.py) antes desta comparação."
        )
    reps = sorted(reps, key=lambda r: r["repetition_index"])
    return [float(extrator(r["metrics"])) for r in reps]


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """PT: d de Cohen com desvio-padrão combinado, corrigido pelo fator de viés J de amostra
        pequena (g de Hedges).
    EN: Cohen's d with pooled standard deviation, corrected by the small-sample bias factor
        J (Hedges' g).
    """
    na, nb = len(a), len(b)
    sa2, sb2 = a.var(ddof=1), b.var(ddof=1)
    dp_combinado = np.sqrt(((na - 1) * sa2 + (nb - 1) * sb2) / (na + nb - 2))
    d = (a.mean() - b.mean()) / dp_combinado
    j = 1 - 3 / (4 * (na + nb) - 9)
    return float(d * j)


def comparar_welch(nome_a: str, a: list, nome_b: str, b: list, alfa: float) -> dict:
    """PT: Teste t de Welch (variâncias desiguais) entre duas amostras independentes de 10
        repetições, com IC 95% da diferença pelos graus de liberdade de Welch-Satterthwaite.
    EN: Welch's t test (unequal variances) between two independent samples of 10
        repetitions, with a 95% CI of the difference using Welch-Satterthwaite degrees of
        freedom.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    sa2, sb2 = a.var(ddof=1), b.var(ddof=1)

    diferenca = float(a.mean() - b.mean())
    se = float(np.sqrt(sa2 / na + sb2 / nb))
    df_welch = (sa2 / na + sb2 / nb) ** 2 / (
        (sa2 / na) ** 2 / (na - 1) + (sb2 / nb) ** 2 / (nb - 1)
    )
    t_stat, valor_p = stats.ttest_ind(a, b, equal_var=False)
    t_critico = stats.t.ppf(0.975, df_welch)
    ic_inf, ic_sup = diferenca - t_critico * se, diferenca + t_critico * se

    return {
        "config_a": nome_a, "config_b": nome_b,
        "n_a": na, "n_b": nb,
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "diff_pp": diferenca * 100,
        "ci95_diff_pp": [ic_inf * 100, ic_sup * 100],
        "t_stat": float(t_stat), "df_welch": float(df_welch),
        "p_value": float(valor_p),
        "hedges_g": hedges_g(a, b),
        "alpha_corrected": alfa,
        "significant_bonferroni": bool(valor_p < alfa),
    }


def familia_execucao(familia: list, valores_por_exp: dict, alfa_base_familia: float, tamanho_familia: int) -> list:
    alfa_corrigido = alfa_base_familia / tamanho_familia
    linhas_tab = []
    for exp_a, exp_b in familia:
        res = comparar_welch(
            ROTULOS[exp_a], valores_por_exp[exp_a],
            ROTULOS[exp_b], valores_por_exp[exp_b],
            alfa_corrigido,
        )
        linhas_tab.append(res)
        print(
            f"{res['config_a']:28s} vs {res['config_b']:22s} "
            f"diff={res['diff_pp']:+5.1f}pp  "
            f"CI=[{res['ci95_diff_pp'][0]:+.1f},{res['ci95_diff_pp'][1]:+.1f}]  "
            f"t={res['t_stat']:.2f} (df={res['df_welch']:.1f})  "
            f"p={res['p_value']:.4f}  g={res['hedges_g']:.2f}  "
            f"sig(Bonferroni, alpha={alfa_corrigido:.4f})? "
            f"{'SIM' if res['significant_bonferroni'] else 'não'}"
        )
    return linhas_tab


def principal():
    valores_por_exp = {exp: carregar_valores(exp) for exp in ESPEC_METRICAS}

    print("=== VLM pilot, rótulo de polímero (família de 3, alpha=0.05/3) ===\n")
    linhas_polimero = familia_execucao(FAMILIA_POLIMERO, valores_por_exp, alfa_base_familia=0.05, tamanho_familia=3)

    print("\n=== Regime de tamanho, SmolVLM (família de 1, alpha=0.05) ===\n")
    linhas_tamanho = familia_execucao(FAMILIA_REGIME_TAMANHO, valores_por_exp, alfa_base_familia=0.05, tamanho_familia=1)

    # PT: U de Mann-Whitney (livre de distribuição) para o par LoRA x ajuste completo, como
    #     checagem de robustez, já que o t de Welch fica perto do limiar de decisão para
    #     este par.
    # EN: Mann-Whitney U (distribution-free) for the LoRA x full fine-tuning pair, as a
    #     robustness check, since Welch's t is close to the decision threshold for this
    #     pair.
    vals_lora = np.asarray(valores_por_exp["05_smolvlm_size_lora"])
    vals_completo = np.asarray(valores_por_exp["06_smolvlm_size_full"])
    u_stat, mw_p = stats.mannwhitneyu(vals_lora, vals_completo, alternative="two-sided")
    print(f"\nMann-Whitney U (LoRA vs full fine tuning): U={u_stat:.1f}  p={mw_p:.4f}")
    linhas_tamanho[0]["mann_whitney_p_value"] = float(mw_p)

    todas_linhas = linhas_polimero + linhas_tamanho
    dir_saida = cc.RELATORIO_FINAL_DIR
    dir_saida.mkdir(parents=True, exist_ok=True)

    caminho_csv = dir_saida / "comparacoes_ctrl_repetitions.csv"
    pd.DataFrame(todas_linhas).to_csv(caminho_csv, index=False)

    import json
    caminho_json = dir_saida / "comparacoes_ctrl_repetitions.json"
    with open(caminho_json, "w", encoding="utf-8") as f:
        json.dump(
            {"polymer_family": linhas_polimero, "size_regime_family": linhas_tamanho,
             "polymer_alpha_corrected": 0.05 / 3, "size_regime_alpha": 0.05},
            f, indent=2, ensure_ascii=False,
        )

    print(f"\nSalvo em:\n  {caminho_csv}\n  {caminho_json}")


if __name__ == "__main__":
    principal()
