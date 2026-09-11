"""Regera fig_size_pilot.png (teste de regime de tamanho do SmolVLM,
Tabela size_pilot_results) com a paleta Okabe-Ito.

IC 95% via bootstrap por imagem-fonte (10.000 réplicas, seed=42), mesma
lógica de 19_compare_size_regime_configs.py.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import OKABE_ITO, apply_style

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "resultados" / "reports"
MANIFEST = ROOT / "resultados" / "15_build_size_regime_manifest" / "data" / "manifest_size_regime.csv"
FIGURES_DIR = ROOT.parent / "Paper_2___Modelos" / "figures"

N_BOOTSTRAP = 10_000
SEED = 42
CONFIGS_BY_LANG = {
    "pt": {
        "Zero-shot\n(500M)": "size_zeroshot_real_test",
        "LoRA\n(500M)": "size_lora_real_test",
        "Full fine-tuning\n(500M)": "size_full_ft_real_test",
    },
    "en": {
        "Zero-shot\n(500M)": "size_zeroshot_real_test",
        "LoRA\n(500M)": "size_lora_real_test",
        "Full fine tuning\n(500M)": "size_full_ft_real_test",
    },
}

apply_style()

manifest = pd.read_csv(MANIFEST)


def load_predictions(tag):
    df = pd.read_csv(REPORTS / f"{tag}_predictions.csv")
    df["correct"] = df["correct"].astype(bool)
    return df[["image_path", "true_label", "pred_label", "correct"]]


def bootstrap_ci(df, seed=SEED, n_bootstrap=N_BOOTSTRAP):
    merged = df.merge(manifest[["image_path", "source_image"]], on="image_path", how="left")
    rng = np.random.default_rng(seed)
    source_images = merged["source_image"].unique()
    groups = {s: g for s, g in merged.groupby("source_image")}
    accs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sampled = rng.choice(source_images, size=len(source_images), replace=True)
        rows = pd.concat([groups[s] for s in sampled], ignore_index=True)
        accs[i] = rows["correct"].mean()
    lo, hi = np.percentile(accs, [2.5, 97.5])
    return lo, hi


colors = [OKABE_ITO["orange"], OKABE_ITO["sky_blue"], OKABE_ITO["bluish_green"]]
# estatísticas independem do idioma -- calcula uma vez usando as tags (iguais
# nos dois dicts), reaproveita pros dois gráficos
means, err_lo, err_hi = [], [], []
for tag in CONFIGS_BY_LANG["en"].values():
    df = load_predictions(tag)
    acc = df["correct"].mean()
    lo, hi = bootstrap_ci(df)
    means.append(acc)
    err_lo.append(acc - lo)
    err_hi.append(hi - acc)
    print(f"{tag:22s} acc={acc:.3f}  IC95=[{lo:.3f}, {hi:.3f}]")

TITLES = {
    "pt": "Teste de regime de tamanho do SmolVLM\n(condição caixa de referência)",
    "en": "SmolVLM size regime test\n(reference box condition)",
}
YLABELS = {"pt": "Acurácia (teste travado, n=208)", "en": "Accuracy (locked test, n=208)"}
CHANCE_LABEL = {"pt": "nível de acaso (teste balanceado)", "en": "chance level (balanced test)"}

for lang, configs in CONFIGS_BY_LANG.items():
    labels = list(configs.keys())
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    bars = ax.bar(x, means, color=colors, edgecolor="white", linewidth=0.8, width=0.55,
                  yerr=[err_lo, err_hi], capsize=5,
                  error_kw={"linewidth": 1.3, "ecolor": OKABE_ITO["black"]})
    ax.axhline(0.5, color=OKABE_ITO["black"], linestyle="--", linewidth=1.0, alpha=0.6,
               label=CHANCE_LABEL[lang])

    for b, v, hi in zip(bars, means, err_hi):
        ax.text(b.get_x() + b.get_width() / 2, v + hi + 0.025, f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(YLABELS[lang])
    ax.set_ylim(0, 1.12)
    ax.set_title(TITLES[lang])
    ax.legend(frameon=False, fontsize=8, loc="lower right")

    fig.tight_layout()
    out_path = FIGURES_DIR / f"fig_size_pilot_{lang}.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"salvo em {out_path}")
