"""Gera fig_comparacao_mina_barras.png (mAP@0.5 de caixa por método x
protocolo, execução única) -- mesma paleta/estilo dos outros fig_*.py do
projeto (Okabe-Ito, ver experimentos/scripts/fig_style.py).

Fonte: experimentos_comparacao/relatorio_final/comparacao_mina.json,
gerado por 13_mina_compare_report.py -- rodar esse primeiro.

Uso:
    python3 14_fig_comparacao_mina_barras.py
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comp_common as cc

EXPERIMENTOS_SCRIPTS = cc.EXPERIMENTOS_DIR / "scripts"
sys.path.insert(0, str(EXPERIMENTOS_SCRIPTS))
from fig_style import OKABE_ITO, apply_style  # noqa: E402

FIGURES_DIR = cc.DATASETMINA_ROOT / "Paper_2___Modelos" / "figures"
METHOD_ORDER = ["Benchnano-seg (YOLO26L-seg)", "YOLOv10", "Faster R-CNN", "Mask R-CNN"]

STRINGS = {
    "pt": {
        "protocol_label": {"full": "Imagem inteira", "patches": "Recorte 256×256 (curated crop)"},
        "ylabel": "mAP@0.5 de caixa (%)",
        "title": "Comparação com o benchmark do MiNa",
        "group_a": "BenchNano",
        "group_b": "MiNa (retreinado no split travado)",
        "protocol_legend_title": "Protocolo",
        "method_legend": ["BenchNano-seg (nosso)", "MiNa (retreinado, mesmo split)"],
    },
    "en": {
        "protocol_label": {"full": "Full image", "patches": "256×256 crop (curated crop)"},
        "ylabel": "Box mAP@0.5 (%)",
        "title": "Comparison with the MiNa benchmark",
        "group_a": "BenchNano",
        "group_b": "MiNa (retrained on the locked split)",
        "protocol_legend_title": "Protocol",
        "method_legend": ["BenchNano-seg (ours)", "MiNa (retrained, same split)"],
    },
}

apply_style()


GROUP_GAP = 0.8  # espaço extra entre o grupo BenchNano (1 método) e o grupo MiNa (3 métodos)


def main():
    json_path = cc.RELATORIO_FINAL_DIR / "comparacao_mina.json"
    if not json_path.exists():
        raise SystemExit("rode 13_mina_compare_report.py primeiro.")
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    by_method_protocol = {(r["protocol"], r["method"]): r for r in data["box"]}

    for lang, s in STRINGS.items():
        # posições no eixo x: método 0 (Benchnano-seg) isolado à esquerda, com um
        # vão extra antes do grupo dos 3 métodos do MiNa retreinados
        x = np.array([0.0] + [1.0 + GROUP_GAP + i for i in range(len(METHOD_ORDER) - 1)])
        width = 0.35
        fig, ax = plt.subplots(figsize=(8.2, 5.4))

        colors = {"full": OKABE_ITO["blue"], "patches": OKABE_ITO["orange"]}
        for i, protocol in enumerate(("full", "patches")):
            values = []
            for method in METHOD_ORDER:
                r = by_method_protocol.get((protocol, method))
                values.append(r["map50"] * 100 if r and r["map50"] is not None else 0.0)
            offset = (i - 0.5) * width
            # sem hachura para o BenchNano (método 0), hachura para os 3 do MiNa
            hatches = [None] + ["///"] * (len(METHOD_ORDER) - 1)
            bars = ax.bar(x + offset, values, width=width, label=s["protocol_label"][protocol],
                          color=colors[protocol], edgecolor="white", linewidth=0.6)
            for b, v, h in zip(bars, values, hatches):
                b.set_hatch(h)
                if h:
                    b.set_edgecolor("white")
                if v > 0:
                    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}",
                            ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(METHOD_ORDER, rotation=10, ha="right")
        ax.set_ylabel(s["ylabel"])
        ax.set_ylim(0, 100)
        ax.set_title(s["title"], fontsize=11.5, pad=34)

        # rótulos de grupo (BenchNano vs. MiNa retreinado), entre o título e o gráfico
        ax.text(x[0], 1.02, s["group_a"], ha="center", va="bottom", fontsize=10, fontweight="bold",
                 transform=ax.get_xaxis_transform(), clip_on=False)
        ax.text(x[1:].mean(), 1.02, s["group_b"], ha="center", va="bottom",
                 fontsize=10, transform=ax.get_xaxis_transform(), clip_on=False)
        ax.axvline((x[0] + x[1]) / 2, color="grey", linewidth=0.7, linestyle=":", alpha=0.6,
                   ymax=1.0, clip_on=False)

        # duas legendas: protocolo (cor) e método/grupo (hachura)
        protocol_legend = ax.legend(frameon=False, loc="upper left", fontsize=9,
                                     title=s["protocol_legend_title"])
        ax.add_artist(protocol_legend)
        group_handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor="lightgrey", edgecolor="white", hatch=None,
                          label=s["method_legend"][0]),
            plt.Rectangle((0, 0), 1, 1, facecolor="lightgrey", edgecolor="white", hatch="///",
                          label=s["method_legend"][1]),
        ]
        ax.legend(handles=group_handles, frameon=False, loc="upper right", fontsize=8.5)

        fig.tight_layout(rect=(0, 0, 1, 0.92))
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = FIGURES_DIR / f"fig_comparacao_mina_barras_{lang}.png"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"salvo em {out_path}")


if __name__ == "__main__":
    main()
