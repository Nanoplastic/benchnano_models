"""Gera fig_yolo_qualitative_examples.png: exemplos qualitativos de detecção
do YOLO26L sobre duas imagens reais da partição de teste travada, com
caixas coloridas por status (acerto / falso negativo / falso positivo).

Painel A: PS_PSL 10000X.png -- o caso extremo citado no texto (Seção
results_end_to_end, main_pt.tex linha ~611): 140 partículas de referência,
predominantemente nanoplástico, recall=0,229 (32/140 casadas).

Painel B: PE_PE 1000-1.png -- uma das duas imagens com recall perfeito
(1,000) citadas no mesmo parágrafo (n=8 partículas de referência).

Casamento por IoU (mesma classe, IOU_THRESHOLD=0.5, guloso por maior IoU
primeiro) -- mesma lógica de 14_yolo_test_eval.py.
"""
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import OKABE_ITO, apply_style

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "datasets_yolo26_v2"
MODEL_PATH = ROOT / "runs_v2" / "microplastic_yolo26_v2" / "weights" / "best.pt"
FIGURES_DIR = ROOT.parent / "Paper_2___Modelos" / "figures"
CLASS_NAMES = ["PE", "PET", "PP", "PS"]
IMG_SIZE = 1024
CONF_THRESHOLD = 0.15
IOU_THRESHOLD = 0.5

COLOR_TP = OKABE_ITO["bluish_green"]
COLOR_FN = OKABE_ITO["vermillion"]
COLOR_FP = OKABE_ITO["orange"]

apply_style()


def yolo_to_xyxy_px(cx, cy, w, h, img_w, img_h):
    return ((cx - w / 2) * img_w, (cy - h / 2) * img_h,
            (cx + w / 2) * img_w, (cy + h / 2) * img_h)


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_gt(label_path, img_w, img_h):
    out = []
    for line in open(label_path):
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        out.append((cls_id, yolo_to_xyxy_px(cx, cy, w, h, img_w, img_h)))
    return out


def match(model, img_path, gts, img_w, img_h):
    pred = model.predict(source=str(img_path), imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]
    preds = []
    if pred.boxes is not None and len(pred.boxes):
        xyxy = pred.boxes.xyxy.cpu().numpy()
        cls_ids = pred.boxes.cls.cpu().numpy().astype(int)
        for box, c in zip(xyxy, cls_ids):
            preds.append((int(c), tuple(box)))

    candidates = []
    for gi, (g_cls, g_box) in enumerate(gts):
        for pi, (p_cls, p_box) in enumerate(preds):
            if p_cls != g_cls:
                continue
            iou = iou_xyxy(g_box, p_box)
            if iou >= IOU_THRESHOLD:
                candidates.append((iou, gi, pi))
    candidates.sort(reverse=True)

    matched_gt, matched_pred = set(), set()
    for iou, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
    return gts, preds, matched_gt, matched_pred


def draw_panel(ax, img_path, label_path, title, title_suffix):
    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size
    gts = load_gt(label_path, img_w, img_h)
    gts, preds, matched_gt, matched_pred = match(model, img_path, gts, img_w, img_h)

    ax.imshow(np.asarray(img))
    for gi, (g_cls, box) in enumerate(gts):
        x1, y1, x2, y2 = box
        color = COLOR_TP if gi in matched_gt else COLOR_FN
        ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                         edgecolor=color, linewidth=1.3))
    for pi, (p_cls, box) in enumerate(preds):
        if pi in matched_pred:
            continue
        x1, y1, x2, y2 = box
        ax.add_patch(mpatches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                         edgecolor=COLOR_FP, linewidth=1.3, linestyle="--"))

    n_gt = len(gts)
    recall = len(matched_gt) / n_gt if n_gt else float("nan")
    ax.set_title(f"{title}\n({title_suffix.format(n_gt=n_gt, recall=recall)})", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


model = YOLO(str(MODEL_PATH))

STRINGS = {
    "pt": {
        "panel_a_title": "Caso de alta densidade (PS, ampliação 10000$\\times$)",
        "panel_b_title": "Caso de recall perfeito (PE, ampliação 1000$\\times$)",
        "suffix": "n={n_gt} partículas de referência, recall={recall:.3f}",
        "legend": ["acerto (IoU $\\geq$ 0,5)", "falso negativo (referência sem par)",
                   "falso positivo (predição sem par)"],
        "suptitle": "Exemplos qualitativos de detecção do YOLO26L (partição de teste travada)",
    },
    "en": {
        "panel_a_title": "High density case (PS, 10000$\\times$ magnification)",
        "panel_b_title": "Perfect recall case (PE, 1000$\\times$ magnification)",
        "suffix": "n={n_gt} reference particles, recall={recall:.3f}",
        "legend": ["match (IoU $\\geq$ 0.5)", "false negative (unmatched reference)",
                   "false positive (unmatched prediction)"],
        "suptitle": "YOLO26L qualitative detection examples (locked test partition)",
    },
}

for lang, s in STRINGS.items():
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2))

    draw_panel(
        axes[0],
        TEST_DIR / "images" / "test" / "PS_PSL 10000X.png",
        TEST_DIR / "labels" / "test" / "PS_PSL 10000X.txt",
        s["panel_a_title"], s["suffix"],
    )
    draw_panel(
        axes[1],
        TEST_DIR / "images" / "test" / "PE_PE 1000-1.png",
        TEST_DIR / "labels" / "test" / "PE_PE 1000-1.txt",
        s["panel_b_title"], s["suffix"],
    )

    legend_handles = [
        mpatches.Patch(edgecolor=COLOR_TP, fill=False, label=s["legend"][0]),
        mpatches.Patch(edgecolor=COLOR_FN, fill=False, label=s["legend"][1]),
        mpatches.Patch(edgecolor=COLOR_FP, fill=False, linestyle="--", label=s["legend"][2]),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=3, frameon=False, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(s["suptitle"], fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    out_path = FIGURES_DIR / f"fig_yolo_qualitative_examples_{lang}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"salvo em {out_path}")
