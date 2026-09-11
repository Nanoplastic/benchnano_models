"""Paleta e estilo compartilhados pelos scripts fig_*.py (30-36).

Paleta categórica: Okabe-Ito / "paleta Wong" (Wong, B. "Points of view:
Color blindness". Nature Methods 8, 441 (2011)), a paleta categórica mais
citada como padrão seguro para daltonismo em periódicos científicos.

Mapas de calor (matrizes de confusão): 'viridis' (perceptualmente uniforme,
seguro para daltonismo, default do Matplotlib desde a v2.0), no lugar do
'Blues' anterior.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito / Wong 2011 -- ordem pensada p/ uso sequencial em barras/séries
OKABE_ITO = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}
CATEGORICAL = [
    OKABE_ITO["blue"],
    OKABE_ITO["orange"],
    OKABE_ITO["bluish_green"],
    OKABE_ITO["vermillion"],
    OKABE_ITO["reddish_purple"],
    OKABE_ITO["sky_blue"],
    OKABE_ITO["yellow"],
]
HEATMAP_CMAP = "viridis"

FIGURES_DIR = None  # setado por cada script via set_figures_dir()


def set_figures_dir(path):
    global FIGURES_DIR
    FIGURES_DIR = path


def apply_style():
    plt.rcParams.update({
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.axisbelow": True,
        "savefig.dpi": 200,
        "figure.dpi": 200,
    })


apply_style()
