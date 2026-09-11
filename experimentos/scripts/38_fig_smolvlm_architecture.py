"""Gera fig_smolvlm_architecture.png (PT, para Paper_2___Modelos/main_pt.tex)
e fig_smolvlm_architecture.png em legado/Paper_2___Modelos/figures/ (EN,
para legado/Paper_2___Modelos/main_V5.tex) -- diagrama esquemático da
arquitetura do SmolVLM (codificador visual SigLIP, conector com pixel
shuffle + projeção linear, backbone de linguagem SmolLM2), pedido pelo
professor para acompanhar a Seção "Arquitetura do SmolVLM"/"SmolVLM
architecture".

Não depende de dados de experimento -- é um diagrama esquemático, não um
gráfico de resultado. Conteúdo textual dos blocos segue a descrição já
escrita nos dois manuscritos e Marafioti et al. 2025 (SmolVLM).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import OKABE_ITO, apply_style

# repo root = .../datasetMINA (script está em experimentos/scripts/)
ROOT = Path(__file__).resolve().parent.parent.parent

TARGETS = {
    "pt": ROOT / "Paper_2___Modelos" / "figures" / "fig_smolvlm_architecture.png",
    "en": ROOT / "legado" / "Paper_2___Modelos" / "figures" / "fig_smolvlm_architecture.png",
}

LABELS = {
    "pt": dict(
        img="Imagem de\nentrada\n\nsubimagens de\nresolução fixa +\nvisão reduzida",
        siglip="Codificador\nvisual\nSigLIP\n\n(por subimagem)",
        shuffle="Pixel shuffle\n\nresolução espacial\n$\\rightarrow$ profundidade\nde canal ($r^2$)",
        proj="Projeção\nlinear\n\n$\\rightarrow$ espaço de\nembeddings do LM",
        connector="conector",
        concat="Concatenação\n\ntokens visuais +\ntokens textuais",
        lm="SmolLM2\n\nmodelo de\nlinguagem",
        text_tokens="Tokens textuais\ndo prompt",
        output="predição de\nregime de tamanho",
    ),
    "en": dict(
        img="Input image\n\nfixed-resolution\nsub-images +\ndown-scaled view",
        siglip="Visual\nencoder\nSigLIP\n\n(per sub-image)",
        shuffle="Pixel shuffle\n\nspatial resolution\n$\\rightarrow$ channel\ndepth ($r^2$)",
        proj="Linear\nprojection\n\n$\\rightarrow$ language-model\nembedding space",
        connector="connector",
        concat="Concatenation\n\nvisual tokens +\ntext tokens",
        lm="SmolLM2\n\nlanguage\nmodel",
        text_tokens="Prompt\ntext tokens",
        output="size-regime\nprediction",
    ),
}

apply_style()


def box(ax, xy, w, h, text, facecolor, fontsize=9.5, textcolor="black"):
    x, y = xy
    patch = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.1, edgecolor="black", facecolor=facecolor, alpha=0.92,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
             fontsize=fontsize, color=textcolor, zorder=3, linespacing=1.3)
    return (x, y, w, h)


def arrow(ax, p0, p1, color="black", style="-|>", lw=1.3, connectionstyle="arc3,rad=0.0"):
    a = FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=13,
                         linewidth=lw, color=color, zorder=1,
                         connectionstyle=connectionstyle)
    ax.add_patch(a)


def render(lang):
    L = LABELS[lang]
    out_path = TARGETS[lang]

    fig, ax = plt.subplots(figsize=(11.8, 5.2))
    ax.set_xlim(0, 15.4)
    ax.set_ylim(-2.3, 4.0)
    ax.axis("off")

    y_main = 1.3
    h_main = 1.5

    box(ax, (0.2, y_main), 2.0, h_main, L["img"], "#DDDDDD", fontsize=8.7)
    box(ax, (2.7, y_main), 2.0, h_main, L["siglip"], OKABE_ITO["sky_blue"], fontsize=9)
    box(ax, (5.2, y_main), 1.9, h_main, L["shuffle"], OKABE_ITO["orange"], fontsize=8.7)
    box(ax, (7.6, y_main), 1.8, h_main, L["proj"], OKABE_ITO["orange"], fontsize=8.7)

    conn_x0, conn_x1 = 5.2, 9.4
    ax.plot([conn_x0, conn_x0, conn_x1, conn_x1],
            [y_main + h_main + 0.06, y_main + h_main + 0.22,
             y_main + h_main + 0.22, y_main + h_main + 0.06],
            color="black", linewidth=1.0, zorder=1)
    ax.text((conn_x0 + conn_x1) / 2, y_main + h_main + 0.38, L["connector"],
            ha="center", va="bottom", fontsize=9.5, style="italic")

    box(ax, (9.9, y_main - 0.55), 1.6, h_main + 1.1, L["concat"],
        OKABE_ITO["bluish_green"], fontsize=8.7, textcolor="white")
    box(ax, (11.9, y_main - 0.55), 1.5, h_main + 1.1, L["lm"],
        OKABE_ITO["vermillion"], fontsize=9, textcolor="white")

    y_text = -1.7
    box(ax, (9.7, y_text), 2.0, 1.0, L["text_tokens"],
        OKABE_ITO["reddish_purple"], fontsize=8.7, textcolor="white")

    arrow(ax, (2.2, y_main + h_main / 2), (2.7, y_main + h_main / 2))
    arrow(ax, (4.7, y_main + h_main / 2), (5.2, y_main + h_main / 2))
    arrow(ax, (7.1, y_main + h_main / 2), (7.6, y_main + h_main / 2))
    arrow(ax, (9.4, y_main + h_main / 2), (9.9, y_main + h_main / 2))
    arrow(ax, (11.5, y_main + h_main / 2), (11.9, y_main + h_main / 2))

    arrow(ax, (10.7, y_text + 1.0), (10.7, y_main - 0.55),
          connectionstyle="arc3,rad=0.0")

    arrow(ax, (13.4, y_main + h_main / 2), (14.15, y_main + h_main / 2))
    ax.text(14.25, y_main + h_main / 2, L["output"],
            ha="left", va="center", fontsize=8.3, style="italic")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Salvo em {out_path}")


for lang in ("pt", "en"):
    render(lang)
