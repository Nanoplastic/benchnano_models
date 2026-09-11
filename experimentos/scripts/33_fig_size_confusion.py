"""Regera fig_size_confusion.png (matrizes de confusão de regime de tamanho
do SmolVLM, condição caixa-referência n=208 e caixa-YOLO n=1482) com cmap
'viridis' no lugar de 'Blues'.

Fontes:
- caixa-referência: resultados/reports/size_lora_real_test_predictions.csv
- caixa-YOLO: resultados/20_predicted_box_and_end_to_end/reports/predicted_box_end_to_end.csv
  (filtrado a D_i & P_i, i.e. partícula casada espacialmente E com polímero correto)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import HEATMAP_CMAP, apply_style

ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT.parent / "Paper_2___Modelos" / "figures"
LABELS = ["MICROPLASTIC", "NANOPLASTIC"]

apply_style()


def confusion(true_labels, pred_labels, labels=LABELS):
    idx = {l: i for i, l in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        cm[idx[t], idx[p]] += 1
    return cm


# --- condição caixa-referência (n=208) ---
oracle = pd.read_csv(ROOT / "resultados" / "reports" / "size_lora_real_test_predictions.csv")
cm_oracle = confusion(oracle["true_label"], oracle["pred_label"])

# --- condição caixa-YOLO (n=1482, D_i & P_i) ---
e2e = pd.read_csv(ROOT / "resultados" / "20_predicted_box_and_end_to_end" / "reports" / "predicted_box_end_to_end.csv")
sub = e2e[(e2e["D_i"] == True) & (e2e["P_i"] == True)]
cm_yolo = confusion(sub["gt_regime"], sub["pred_regime"])

STRINGS = {
    "pt": {
        "titles": [f"Caixa de referência (n={cm_oracle.sum()})", f"Caixa do YOLO26L (n={cm_yolo.sum()})"],
        "xlabel": "Predito", "ylabel": "Referência",
        "suptitle": "Regime de tamanho: matrizes de confusão do SmolVLM (partição de teste travada)",
        "cbar": "proporção normalizada por linha",
    },
    "en": {
        "titles": [f"Reference box (n={cm_oracle.sum()})", f"YOLO26L box (n={cm_yolo.sum()})"],
        "xlabel": "Predicted", "ylabel": "Reference",
        "suptitle": "Size regime: SmolVLM confusion matrices (locked test partition)",
        "cbar": "row-normalized proportion",
    },
}

for lang, s in STRINGS.items():
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.5))
    for ax, cm, title in zip(axes, [cm_oracle, cm_yolo], s["titles"]):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)
        im = ax.imshow(cm_norm, cmap=HEATMAP_CMAP, vmin=0, vmax=1)
        ax.set_xticks(range(len(LABELS)))
        ax.set_yticks(range(len(LABELS)))
        ax.set_xticklabels(["MICRO", "NANO"])
        ax.set_yticklabels(["MICRO", "NANO"])
        ax.set_xlabel(s["xlabel"])
        ax.set_ylabel(s["ylabel"])
        ax.set_title(title, fontsize=10)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "black" if cm_norm[i, j] > 0.5 else "white"
                ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.1%})", ha="center", va="center",
                        color=color, fontsize=9)

    fig.suptitle(s["suptitle"], fontsize=11)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label=s["cbar"])
    out_path = FIGURES_DIR / f"fig_size_confusion_{lang}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"salvo em {out_path}")
