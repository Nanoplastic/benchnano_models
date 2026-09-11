"""Regera fig_end_to_end_funnel.png (funil de decomposição de erro fim-a-fim)
com a paleta Okabe-Ito.

Fonte: resultados/20_predicted_box_and_end_to_end/reports/predicted_box_and_end_to_end.csv
(4610 partículas de referência na partição de teste travada).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import OKABE_ITO, apply_style

ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = ROOT.parent / "Paper_2___Modelos" / "figures"

apply_style()

df = pd.read_csv(ROOT / "resultados" / "20_predicted_box_and_end_to_end" / "reports" / "predicted_box_end_to_end.csv")
n_total = len(df)
n_D = int(df["D_i"].sum())
n_DP = int(((df["D_i"] == True) & (df["P_i"] == True)).sum())
n_E = int((df["E_i"] == True).sum())

STRINGS = {
    "pt": {
        "stages": [
            ("Partículas de\nreferência", n_total),
            ("Detectadas\n$D$", n_D),
            ("Detectadas + polímero\ncorreto  $D \\cdot P$", n_DP),
            ("Fim-a-fim estrito\n$E = D \\cdot P \\cdot S$", n_E),
        ],
        "ylabel": "Número de partículas",
        "title": f"Decomposição de erro fim-a-fim\n(n={n_total} partículas de referência)",
    },
    "en": {
        "stages": [
            ("Reference\nparticles", n_total),
            ("Detected\n$D$", n_D),
            ("Detected + correct\npolymer  $D \\cdot P$", n_DP),
            ("Strict end to end\n$E = D \\cdot P \\cdot S$", n_E),
        ],
        "ylabel": "Number of particles",
        "title": f"End to end error decomposition\n(n={n_total} reference particles)",
    },
}

for lang, s in STRINGS.items():
    stages = s["stages"]
    labels = [t[0] for t in stages]
    values = [t[1] for t in stages]
    colors = [OKABE_ITO["black"], OKABE_ITO["orange"], OKABE_ITO["sky_blue"], OKABE_ITO["bluish_green"]]
    colors[0] = "#BBBBBB"  # barra de referência (total) em cinza neutro, não faz parte da paleta categórica

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", linewidth=0.8, width=0.6)
    for b, v in zip(bars, values):
        pct = v / n_total
        ax.text(b.get_x() + b.get_width() / 2, v + 60, f"{v}\n({pct:.1%})",
                ha="center", va="bottom", fontsize=9)

    ax.set_ylabel(s["ylabel"])
    ax.set_ylim(0, n_total * 1.18)
    ax.set_title(s["title"])

    fig.tight_layout()
    out_path = FIGURES_DIR / f"fig_end_to_end_funnel_{lang}.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"salvo em {out_path}")
print(stages)
