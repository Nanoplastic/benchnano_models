"""Regera fig_polymer_pilot.png (estudo de classificação de polímero
apenas-VLM, Tabela vlm_pilot_results) com a paleta Okabe-Ito.

Para as duas configurações com 4 execuções (LoRA apenas-real e LoRA
real+sintético), a "execução-base" (arquivo sem sufixo _run2/_run3/_run4) é
lida do working tree atual -- que inclui o retreino de 2026-08-23 dessas
duas configurações (ver Seção de Reprodutibilidade do main_pt.tex). A
Tabela vlm_pilot_results foi recalculada para refletir esse retreino
(médias 69,4%/69,7%, faixas 0,673--0,712/0,668--0,740); esta figura usa
exatamente os mesmos 4 arquivos de predição por configuração que geraram
esses números.
"""
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import OKABE_ITO, apply_style

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "resultados" / "reports"
FIGURES_DIR = ROOT.parent / "Paper_2___Modelos" / "figures"

apply_style()


def acc_from_file(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    correct = sum(1 for r in rows if r["correct"].strip().lower() in ("true", "1"))
    return correct / len(rows)


def four_run_stats(base_path, run_paths):
    accs = [acc_from_file(base_path)] + [acc_from_file(p) for p in run_paths]
    return float(np.mean(accs)), min(accs), max(accs)


mean_real_only, lo_real_only, hi_real_only = four_run_stats(
    REPORTS / "mina_real_only_lora_real_test_predictions.csv",
    [REPORTS / f"mina_real_only_lora_real_test_predictions_run{i}.csv" for i in (2, 3, 4)],
)
mean_real_synth, lo_real_synth, hi_real_synth = four_run_stats(
    REPORTS / "mina_real_plus_synthetic_lora_v2_real_test_predictions.csv",
    [REPORTS / f"mina_real_plus_synthetic_lora_v2_real_test_predictions_run{i}.csv" for i in (2, 3, 4)],
)
mean_bigvlm, lo_bigvlm, hi_bigvlm = four_run_stats(
    REPORTS / "mina_real_only_lora_bigvlm_real_test_predictions.csv",
    [REPORTS / f"mina_real_only_lora_bigvlm_real_test_predictions_run{i}.csv" for i in (2, 3, 4)],
)
zero_shot = acc_from_file(REPORTS / "zeroshot_real_test_predictions.csv")
describe_then_classify = acc_from_file(REPORTS / "describe_then_classify_mina_predictions.csv")

STRINGS = {
    "pt": {
        "labels": ["Zero-shot\n(500M)", "LoRA apenas\nreal (500M)", "LoRA real+\nsintético (500M)",
                   "LoRA apenas\nreal ($\\sim$2B)", "Descrever-depois-\nclassificar (500M)"],
        "chance": "nível de acaso (4 classes)",
        "ylabel": "Acurácia (teste travado, n=208)",
        "title": "Estudo de classificação de polímero apenas-VLM",
    },
    "en": {
        "labels": ["Zero-shot\n(500M)", "LoRA real\nonly (500M)", "LoRA real+\nsynthetic (500M)",
                   "LoRA real\nonly ($\\sim$2B)", "Describe-then-\nclassify (500M)"],
        "chance": "chance level (4 classes)",
        "ylabel": "Accuracy (locked test, n=208)",
        "title": "VLM only polymer classification study",
    },
}

means = [zero_shot, mean_real_only, mean_real_synth, mean_bigvlm, describe_then_classify]
err_lo = [0.0, mean_real_only - lo_real_only, mean_real_synth - lo_real_synth, mean_bigvlm - lo_bigvlm, 0.0]
err_hi = [0.0, hi_real_only - mean_real_only, hi_real_synth - mean_real_synth, hi_bigvlm - mean_bigvlm, 0.0]
colors = [OKABE_ITO["orange"], OKABE_ITO["sky_blue"], OKABE_ITO["bluish_green"], OKABE_ITO["blue"], OKABE_ITO["vermillion"]]

for lang, s in STRINGS.items():
    x = np.arange(len(s["labels"]))
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    bars = ax.bar(x, means, color=colors, edgecolor="white", linewidth=0.8, width=0.6,
                  yerr=[err_lo, err_hi], capsize=4,
                  error_kw={"linewidth": 1.2, "ecolor": OKABE_ITO["black"]})
    ax.axhline(0.25, color=OKABE_ITO["black"], linestyle="--", linewidth=1.0, alpha=0.6,
               label=s["chance"])

    for b, v, hi in zip(bars, means, err_hi):
        ax.text(b.get_x() + b.get_width() / 2, v + hi + 0.025, f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(s["labels"], fontsize=8.5)
    ax.set_ylabel(s["ylabel"])
    ax.set_ylim(0, 1.0)
    ax.set_title(s["title"])
    ax.legend(frameon=False, fontsize=8, loc="upper left")

    fig.tight_layout()
    out_path = FIGURES_DIR / f"fig_polymer_pilot_{lang}.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"salvo em {out_path}")
print({"real_only": (mean_real_only, lo_real_only, hi_real_only),
       "real_synth": (mean_real_synth, lo_real_synth, hi_real_synth),
       "bigvlm": (mean_bigvlm, lo_bigvlm, hi_bigvlm),
       "zero_shot": zero_shot, "describe_then_classify": describe_then_classify})
