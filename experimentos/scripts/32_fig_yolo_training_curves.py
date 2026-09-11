"""Regera fig_yolo_training_curves.png com a paleta Okabe-Ito.

Fonte: runs_v2/microplastic_yolo26_v2/results.csv (log nativo do Ultralytics,
351 épocas completadas). Marca a época 209 (melhor checkpoint por mAP50 de
validação, conforme Seção "Treino e inferência do YOLO26L" do main_pt.tex).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import OKABE_ITO, apply_style

ROOT = Path(__file__).resolve().parent.parent
RESULTS_CSV = ROOT / "runs_v2" / "microplastic_yolo26_v2" / "results.csv"
FIGURES_DIR = ROOT.parent / "Paper_2___Modelos" / "figures"
BEST_EPOCH = 209

apply_style()

df = pd.read_csv(RESULTS_CSV)
df.columns = [c.strip() for c in df.columns]

TRAIN_COLOR = OKABE_ITO["blue"]
VAL_COLOR = OKABE_ITO["vermillion"]
METRIC_COLOR = OKABE_ITO["bluish_green"]

STRINGS = {
    "pt": {
        "panels": [
            ("Perda de caixa", "train/box_loss", "val/box_loss"),
            ("Perda de classificação", "train/cls_loss", "val/cls_loss"),
            ("Precisão (val)", "metrics/precision(B)", None),
            ("Recall (val)", "metrics/recall(B)", None),
            ("mAP@0,5 (val)", "metrics/mAP50(B)", None),
            ("mAP@0,5:0,95 (val)", "metrics/mAP50-95(B)", None),
        ],
        "train": "treino", "val": "validação", "xlabel": "Época",
        "annot": "época 209\n(melhor checkpoint)",
        "suptitle": "Curvas de treino e validação do YOLO26L (351 épocas completadas)",
    },
    "en": {
        "panels": [
            ("Box loss", "train/box_loss", "val/box_loss"),
            ("Classification loss", "train/cls_loss", "val/cls_loss"),
            ("Precision (val)", "metrics/precision(B)", None),
            ("Recall (val)", "metrics/recall(B)", None),
            ("mAP@0.5 (val)", "metrics/mAP50(B)", None),
            ("mAP@0.5:0.95 (val)", "metrics/mAP50-95(B)", None),
        ],
        "train": "train", "val": "validation", "xlabel": "Epoch",
        "annot": "epoch 209\n(best checkpoint)",
        "suptitle": "YOLO26L training and validation curves (351 epochs completed)",
    },
}

for lang, s in STRINGS.items():
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 6.0))
    axes = axes.ravel()

    for ax, (title, train_col, val_col) in zip(axes, s["panels"]):
        if val_col is not None:
            ax.plot(df["epoch"], df[train_col], color=TRAIN_COLOR, label=s["train"], linewidth=1.4)
            ax.plot(df["epoch"], df[val_col], color=VAL_COLOR, label=s["val"], linewidth=1.4)
            ax.legend(frameon=False, fontsize=8)
        else:
            ax.plot(df["epoch"], df[train_col], color=METRIC_COLOR, linewidth=1.6)
        ax.axvline(BEST_EPOCH, color=OKABE_ITO["black"], linestyle="--", linewidth=1.0, alpha=0.7)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(s["xlabel"])

    axes[0].text(BEST_EPOCH + 5, axes[0].get_ylim()[1] * 0.9, s["annot"],
                 fontsize=7, color=OKABE_ITO["black"])

    fig.suptitle(s["suptitle"], fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = FIGURES_DIR / f"fig_yolo_training_curves_{lang}.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"salvo em {out_path}")
