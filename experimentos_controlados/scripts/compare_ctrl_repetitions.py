"""Reproduz a Tabela~\\ref{tab:controlled_repetitions_tests} do artigo
(comparações pareadas entre configurações sob as dez repetições
controladas), que não tinha nenhum script correspondente no repositório --
achado durante a revisão de linguagem do artigo (checagem cruzada de todos
os números contra os experimentos, pedida pelo usuário).

Reaproveita os mesmos 10 valores por seed já agregados por
`final_report.py` (via `ctrl_common.load_all_rep_summaries` +
`METRIC_SPECS`), e aplica a mesma disciplina estatística de
`experimentos/scripts/19_compare_size_regime_configs.py`, estendida de
McNemar pareado (usado lá, pra predições no MESMO test set) para o teste
adequado aqui: as dez repetições de duas configurações são AMOSTRAS
INDEPENDENTES (seeds diferentes, modelos diferentes), não observações
pareadas no mesmo conjunto de teste -- por isso a Seção de Statistical
Analysis do artigo usa o teste $t$ de Welch (variâncias não assumidas
iguais) em vez de McNemar para comparações entre repetições controladas.

Duas famílias de comparação, cada uma com sua própria correção de
Bonferroni (Dunn1961MultipleComparisons, já citado no artigo):

  - VLM pilot, rótulo de polímero (família de 3 pares): real-only (500M) x
    real+sintético (500M) x ~2B, alpha_corrigido = 0.05/3 ~= 0.0167.
  - Regime de tamanho, SmolVLM (família de 1 par): LoRA x full fine-tuning,
    alpha = 0.05. Para este par, também roda o teste de Mann-Whitney U
    (livre de distribuição) como checagem de robustez, já que o artigo
    reporta os dois testes quando divergem perto do limiar.

Para cada par: diferença de médias, IC 95% da diferença (via t de Welch),
estatística t, graus de liberdade de Welch-Satterthwaite, valor p, e
Hedges' g (d de Cohen com correção de viés pra amostra pequena).

Uso:
    python3 compare_ctrl_repetitions.py

Requer que os 5 experimentos abaixo já tenham as 10 repetições completas
(mesma pré-condição de final_report.py).
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctrl_common as cc

# mesmos extratores de final_report.py, só para os experimentos que entram
# nas comparações pareadas (o principal YOLO26L e o fim-a-fim pareado não
# são comparados entre si -- não há uma segunda configuração equivalente)
METRIC_SPECS = {
    "02_vlm_polymer_real_only": ("test_acc", lambda m: m["test_acc"]),
    "03_vlm_polymer_real_synth": ("test_acc", lambda m: m["test_acc"]),
    "04_vlm_polymer_bigvlm": ("test_acc", lambda m: m["test_acc"]),
    "05_smolvlm_size_lora": ("test_acc", lambda m: m["test_acc"]),
    "06_smolvlm_size_full": ("test_acc", lambda m: m["test_acc"]),
}

LABELS = {
    "02_vlm_polymer_real_only": "Real only LoRA (500M)",
    "03_vlm_polymer_real_synth": "Real+synthetic LoRA (500M)",
    "04_vlm_polymer_bigvlm": "Real only LoRA (~2B)",
    "05_smolvlm_size_lora": "LoRA",
    "06_smolvlm_size_full": "Full fine tuning",
}

# as duas famílias de comparação, na ordem em que aparecem na tabela do
# artigo -- cada tupla é (experiment_a, experiment_b)
POLYMER_FAMILY = [
    ("02_vlm_polymer_real_only", "03_vlm_polymer_real_synth"),
    ("02_vlm_polymer_real_only", "04_vlm_polymer_bigvlm"),
    ("03_vlm_polymer_real_synth", "04_vlm_polymer_bigvlm"),
]
SIZE_REGIME_FAMILY = [
    ("05_smolvlm_size_lora", "06_smolvlm_size_full"),
]


def load_values(experiment: str) -> list:
    metric_name, extractor = METRIC_SPECS[experiment]
    reps = cc.load_all_rep_summaries(experiment)
    if len(reps) < cc.N_REPS:
        raise RuntimeError(
            f"{experiment}: só {len(reps)}/{cc.N_REPS} repetições encontradas. "
            f"Rode a fila completa (run_queue.py) antes desta comparação."
        )
    reps = sorted(reps, key=lambda r: r["repetition_index"])
    return [float(extractor(r["metrics"])) for r in reps]


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """d de Cohen com desvio-padrão combinado (pooled), corrigido pelo
    fator de viés J de amostra pequena -- vira Hedges' g. Mesma definição
    usada no texto do artigo ("the bias corrected standardized mean
    difference")."""
    na, nb = len(a), len(b)
    sa2, sb2 = a.var(ddof=1), b.var(ddof=1)
    pooled_sd = np.sqrt(((na - 1) * sa2 + (nb - 1) * sb2) / (na + nb - 2))
    d = (a.mean() - b.mean()) / pooled_sd
    j = 1 - 3 / (4 * (na + nb) - 9)
    return float(d * j)


def welch_compare(name_a: str, a: list, name_b: str, b: list, alpha: float) -> dict:
    """Teste t de Welch (variâncias desiguais) entre duas amostras
    independentes de 10 repetições cada, com IC 95% da diferença via
    graus de liberdade de Welch-Satterthwaite -- mesma estatística
    descrita na Seção de Statistical Analysis do artigo."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    sa2, sb2 = a.var(ddof=1), b.var(ddof=1)

    diff = float(a.mean() - b.mean())
    se = float(np.sqrt(sa2 / na + sb2 / nb))
    df_welch = (sa2 / na + sb2 / nb) ** 2 / (
        (sa2 / na) ** 2 / (na - 1) + (sb2 / nb) ** 2 / (nb - 1)
    )
    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)
    t_crit = stats.t.ppf(0.975, df_welch)
    ci_low, ci_high = diff - t_crit * se, diff + t_crit * se

    return {
        "config_a": name_a, "config_b": name_b,
        "n_a": na, "n_b": nb,
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "diff_pp": diff * 100,
        "ci95_diff_pp": [ci_low * 100, ci_high * 100],
        "t_stat": float(t_stat), "df_welch": float(df_welch),
        "p_value": float(p_value),
        "hedges_g": hedges_g(a, b),
        "alpha_corrected": alpha,
        "significant_bonferroni": bool(p_value < alpha),
    }


def run_family(family: list, values_by_exp: dict, family_alpha_base: float, family_size: int) -> list:
    alpha_corrected = family_alpha_base / family_size
    rows = []
    for exp_a, exp_b in family:
        res = welch_compare(
            LABELS[exp_a], values_by_exp[exp_a],
            LABELS[exp_b], values_by_exp[exp_b],
            alpha_corrected,
        )
        rows.append(res)
        print(
            f"{res['config_a']:28s} vs {res['config_b']:22s} "
            f"diff={res['diff_pp']:+5.1f}pp  "
            f"CI=[{res['ci95_diff_pp'][0]:+.1f},{res['ci95_diff_pp'][1]:+.1f}]  "
            f"t={res['t_stat']:.2f} (df={res['df_welch']:.1f})  "
            f"p={res['p_value']:.4f}  g={res['hedges_g']:.2f}  "
            f"sig(Bonferroni, alpha={alpha_corrected:.4f})? "
            f"{'SIM' if res['significant_bonferroni'] else 'não'}"
        )
    return rows


def main():
    values_by_exp = {exp: load_values(exp) for exp in METRIC_SPECS}

    print("=== VLM pilot, rótulo de polímero (família de 3, alpha=0.05/3) ===\n")
    polymer_rows = run_family(POLYMER_FAMILY, values_by_exp, family_alpha_base=0.05, family_size=3)

    print("\n=== Regime de tamanho, SmolVLM (família de 1, alpha=0.05) ===\n")
    size_rows = run_family(SIZE_REGIME_FAMILY, values_by_exp, family_alpha_base=0.05, family_size=1)

    # Mann-Whitney U (livre de distribuição) para o par LoRA x full
    # fine-tuning, reportado no artigo como checagem de robustez porque o
    # teste t de Welch fica perto do limiar de decisão para esse par.
    lora_vals = np.asarray(values_by_exp["05_smolvlm_size_lora"])
    full_vals = np.asarray(values_by_exp["06_smolvlm_size_full"])
    u_stat, mw_p = stats.mannwhitneyu(lora_vals, full_vals, alternative="two-sided")
    print(f"\nMann-Whitney U (LoRA vs full fine tuning): U={u_stat:.1f}  p={mw_p:.4f}")
    size_rows[0]["mann_whitney_p_value"] = float(mw_p)

    all_rows = polymer_rows + size_rows
    out_dir = cc.RELATORIO_FINAL_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "comparacoes_ctrl_repetitions.csv"
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)

    import json
    json_path = out_dir / "comparacoes_ctrl_repetitions.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"polymer_family": polymer_rows, "size_regime_family": size_rows,
             "polymer_alpha_corrected": 0.05 / 3, "size_regime_alpha": 0.05},
            f, indent=2, ensure_ascii=False,
        )

    latex_path = out_dir / "tabela_comparacoes_ctrl_repetitions.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(to_latex(polymer_rows, size_rows))

    print(f"\nSalvo em:\n  {csv_path}\n  {json_path}\n  {latex_path}")


def _fmt_p(p: float, sig: bool, extra_marker: str = "") -> str:
    txt = f"{p:.3f}{extra_marker}"
    return f"\\mathbf{{{txt}}}" if sig else txt


def to_latex(polymer_rows: list, size_rows: list) -> str:
    lines = [
        r"% Gerado automaticamente por experimentos_controlados/scripts/compare_ctrl_repetitions.py",
        r"\begin{table}[H]",
        r"\centering",
        r"\footnotesize",
        r"\renewcommand{\arraystretch}{1.2}",
        r"\begin{tabular}{lrrrrl}",
        r"\toprule",
        r"\textbf{Comparison} & \textbf{Difference} & \textbf{95\% CI of diff.} & "
        r"\textbf{$t$ (df)} & \textbf{$P$} & \textbf{Hedges' $g$} \\",
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{VLM pilot, polymer label (family of 3, "
        r"$\alpha=0.0167$)}} \\",
    ]
    for r in polymer_rows:
        lines.append(
            f"{r['config_a']} vs. {r['config_b']} & "
            f"${r['diff_pp']:+.1f}$ pp & "
            f"[${r['ci95_diff_pp'][0]:+.1f}$, ${r['ci95_diff_pp'][1]:+.1f}$] & "
            f"${r['t_stat']:.2f}$ ({r['df_welch']:.1f}) & "
            f"${_fmt_p(r['p_value'], r['significant_bonferroni'])}$ & "
            f"${r['hedges_g']:.2f}$ \\\\"
        )
    lines += [
        r"\midrule",
        r"\multicolumn{6}{l}{\textit{Size regime, SmolVLM (family of 1, $\alpha=0.05$)}} \\",
    ]
    for r in size_rows:
        lines.append(
            f"{r['config_a']} vs. {r['config_b']} & "
            f"${r['diff_pp']:+.1f}$ pp & "
            f"[${r['ci95_diff_pp'][0]:+.1f}$, ${r['ci95_diff_pp'][1]:+.1f}$] & "
            f"${r['t_stat']:.2f}$ ({r['df_welch']:.1f}) & "
            f"${_fmt_p(r['p_value'], r['significant_bonferroni'])}$ & "
            f"${r['hedges_g']:.2f}$ \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
