"""Regera fig_yolo_confusion_matrix.png com cmap 'viridis' no lugar de 'Blues'.

Recalcula a matriz de confusão via Ultralytics model.val() sobre o mesmo
checkpoint e partição de teste travada (15 imagens) usados em
14_yolo_test_eval.py -- a matriz não era salva como artefato numérico
separado, só como PNG do próprio Ultralytics.
"""
import functools
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import HEATMAP_CMAP, apply_style

from ultralytics import YOLO
from ultralytics.utils.metrics import ConfusionMatrix

# O validador do Ultralytics chama ConfusionMatrix.process_batch(...) sem
# passar iou_thres, caindo no default hardcoded da biblioteca (0.45). O
# artigo usa IoU=0.5 em toda a metodologia de casamento (Eq. iou_threshold,
# Secao "Reference box and predicted box evaluation"). Forcamos o mesmo
# 0.5 aqui para a matriz de confusao ficar consistente com o resto do
# artigo -- sem isso, a diagonal da matriz diverge da Table yolo_results
# em 1-2pp (achado confirmado lendo o codigo-fonte instalado do Ultralytics).
ConfusionMatrix.process_batch = functools.partialmethod(
    ConfusionMatrix.process_batch, iou_thres=0.5
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "runs_v2" / "microplastic_yolo26_v2" / "weights" / "best.pt"
TEST_DIR = ROOT / "datasets_yolo26_v2"
CLASS_NAMES = ["PE", "PET", "PP", "PS"]
FIGURES_DIR = ROOT.parent / "Paper_2___Modelos" / "figures"

apply_style()

data_yaml_test = TEST_DIR / "data_test_figs.yaml"
data_yaml_test.write_text(
    f"path: {TEST_DIR}\n"
    f"train: images/test\n"
    f"val: images/test\n"
    f"nc: {len(CLASS_NAMES)}\n"
    f"names:\n" + "\n".join(f"  - {c}" for c in CLASS_NAMES) + "\n"
)

model = YOLO(str(MODEL_PATH))
# plots=True é necessário para o Ultralytics de fato popular a matriz de
# confusão internamente (com plots=False ela fica zerada); direcionamos a
# saída padrão dele para uma pasta temporária só para não gerar lixo em
# runs/detect/ -- os PNGs que o Ultralytics geraria não são usados aqui,
# só a matriz numérica em metrics.confusion_matrix.matrix.
metrics = model.val(data=str(data_yaml_test), split="val", imgsz=1024, conf=0.15,
                     iou=0.5, plots=True, project="/tmp/yolo_val_for_figs",
                     name="cm", exist_ok=True, verbose=False)
cm = metrics.confusion_matrix.matrix  # linhas=previsto, colunas=referência; idx 4 = background

col_sums = cm.sum(axis=0, keepdims=True)
cm_norm = np.divide(cm, col_sums, out=np.zeros_like(cm), where=col_sums != 0)

STRINGS = {
    "pt": {"labels": CLASS_NAMES + ["background"], "xlabel": "Classe de referência",
           "ylabel": "Classe prevista",
           "title": "Matriz de confusão do YOLO26L (normalizada por coluna)\npartição de teste travada",
           "cbar": "proporção normalizada por coluna"},
    "en": {"labels": CLASS_NAMES + ["background"], "xlabel": "Reference class",
           "ylabel": "Predicted class",
           "title": "YOLO26L confusion matrix (column normalized)\nlocked test partition",
           "cbar": "column-normalized proportion"},
}

for lang, s in STRINGS.items():
    labels = s["labels"]
    fig, ax = plt.subplots(figsize=(7.0, 5.6))
    im = ax.imshow(cm_norm, cmap=HEATMAP_CMAP, vmin=0, vmax=1)

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel(s["xlabel"])
    ax.set_ylabel(s["ylabel"])
    ax.set_title(s["title"])

    text_threshold = 0.5
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val = cm_norm[i, j]
            color = "black" if val > text_threshold else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(s["cbar"])

    fig.tight_layout()
    out_path = FIGURES_DIR / f"fig_yolo_confusion_matrix_{lang}.png"
    fig.savefig(out_path)
    plt.close(fig)
    print(f"salvo em {out_path}")
print(cm)
