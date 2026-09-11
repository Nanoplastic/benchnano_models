"""Regera fig_yolo_perclass.png (Precisão/Recall/AP50 do YOLO26L por classe
de polímero) com a paleta Okabe-Ito no lugar do azul único anterior.

Fonte dos números: resultados/14_yolo_test_eval/summary.json (per_class_metrics),
já conferido contra a Tabela yolo_results do main_pt.tex.
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import OKABE_ITO, apply_style

ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT.parent / "Paper_2___Modelos" / "figures"

apply_style()

with open(ROOT / "resultados" / "14_yolo_test_eval" / "summary.json") as f:
    summary = json.load(f)

per_class = summary["metrics"]["per_class_metrics"]
classes = ["PE", "PET", "PP", "PS"]
precision = [per_class[c]["precision"] for c in classes]
recall = [per_class[c]["recall"] for c in classes]
ap50 = [per_class[c]["ap50"] for c in classes]

STRINGS = {
    "pt": {"precision": "Precisão", "recall": "Recall", "ylabel": "Valor da métrica",
           "title": "Desempenho do YOLO26L por classe de polímero\n(partição de teste travada, n=4610)"},
    "en": {"precision": "Precision", "recall": "Recall", "ylabel": "Metric value",
           "title": "YOLO26L performance by polymer class\n(locked test partition, n=4610)"},
}

for lang, s in STRINGS.items():
    metrics = [
        (s["precision"], precision, OKABE_ITO["orange"]),
        (s["recall"], recall, OKABE_ITO["sky_blue"]),
        ("AP$_{50}$", ap50, OKABE_ITO["bluish_green"]),
    ]

    x = np.arange(len(classes))
    n_bars = len(metrics)
    width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    for i, (label, values, color) in enumerate(metrics):
        offset = (i - (n_bars - 1) / 2) * width
        bars = ax.bar(x + offset, values, width=width, label=label, color=color,
                      edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel(s["ylabel"])
    ax.set_ylim(0, 1.0)
    ax.set_title(s["title"])
    ax.legend(frameon=False, loc="upper right", ncol=3, fontsize=9)

    fig.tight_layout()
    out_path = FIGURES_DIR / f"fig_yolo_perclass_{lang}.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"salvo em {out_path}")
