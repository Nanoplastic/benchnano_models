"""Recall do YOLO26L v2 estratificado pelo regime de tamanho físico
(NANOPLASTIC/MICROPLASTIC, limiar de 1um), não pela área normalizada da
caixa usada em 14_yolo_test_eval.py. Escrito para responder ao TBD da Seção
de Discussão que pede cruzar a queda de acurácia do SmolVLM no subconjunto
nano-only (Seção sec:results_size_pilot) com o recall de localização do
YOLO26L por tamanho.

Reaproveita a MESMA lógica de casamento por IoU (guloso, mesma classe,
IOU_THRESHOLD=0.5, conf=0.15) e o MESMO bootstrap por imagem (10000
replicas, seed=42) de 14_yolo_test_eval.py, só trocando o critério de bucket
de tamanho: em vez de area_norm, usa a calibração px->nm (100/ampliacao) já
validada em 15_build_size_regime_manifest.py e na Tabela de composição do
dataset (reproduziu exatamente as contagens 18329/2791/4610 e 5198/1571/2630
NANOPLASTIC por particao).
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import save_experiment_summary, setup_experiment_logging

from ultralytics import YOLO

EXP_NAME = "23_recall_by_size_regime"
EXP_DIR, LOG_PATH, LOGGER, RUNTIME_METADATA = setup_experiment_logging(EXP_NAME, __file__)

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "datasets_yolo26_v2"
MODEL_PATH = ROOT / "runs_v2" / "microplastic_yolo26_v2" / "weights" / "best.pt"

IMG_SIZE = 1024
CONF_THRESHOLD = 0.15
IOU_THRESHOLD = 0.5
N_BOOTSTRAP = 10_000
SEED = 42
NANO_MAX_NM = 1000.0

MAG_X = re.compile(r"(\d+)\s*[xX]")
MAG_DASH = re.compile(r"^\D*(\d+)-\d+")


def parse_magnification(name: str):
    m = MAG_X.findall(name)
    if m:
        return int(m[-1])
    m = MAG_DASH.match(name)
    if m:
        return int(m.group(1))
    return None


def yolo_to_xyxy(cx, cy, w, h):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


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


def load_gt(label_path: Path, W: int, H: int, um_per_px: float):
    if not label_path.exists():
        return []
    out = []
    for line in open(label_path):
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        px_size = max(w * W, h * H)
        size_nm = px_size * um_per_px * 1000.0
        regime = "NANOPLASTIC" if size_nm < NANO_MAX_NM else "MICROPLASTIC"
        out.append((cls_id, yolo_to_xyxy(cx, cy, w, h), regime))
    return out


def match_predictions(model, img_path: Path, gts):
    pred = model.predict(source=str(img_path), imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]

    preds = []
    if pred.boxes is not None and len(pred.boxes):
        xyxy_norm = pred.boxes.xyxyn.cpu().numpy()
        cls_ids = pred.boxes.cls.cpu().numpy().astype(int)
        for box, c in zip(xyxy_norm, cls_ids):
            preds.append((int(c), tuple(box)))

    candidates = []
    for gi, (g_cls, g_box, _regime) in enumerate(gts):
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

    return matched_gt


def main():
    if not MODEL_PATH.exists():
        print(f"Checkpoint nao existe em {MODEL_PATH}")
        sys.exit(1)

    LOGGER.info("=== recall do YOLO26L v2 por regime de tamanho fisico (test set travado) ===")
    model = YOLO(str(MODEL_PATH))

    bucket_gt = defaultdict(int)
    bucket_matched = defaultdict(int)
    per_image = []  # (image_name, regime -> (matched, gt))

    img_paths = sorted((TEST_DIR / "images/test").glob("*.png"))
    n_unresolved_mag = 0
    for img_path in img_paths:
        mag = parse_magnification(img_path.stem)
        if mag is None:
            n_unresolved_mag += 1
            continue
        with Image.open(img_path) as im:
            W, H = im.size
        um_per_px = 100.0 / mag

        label_path = TEST_DIR / "labels/test" / f"{img_path.stem}.txt"
        gts = load_gt(label_path, W, H, um_per_px)
        if not gts:
            continue

        matched_gt = match_predictions(model, img_path, gts)

        per_image_counts = defaultdict(lambda: [0, 0])  # regime -> [matched, gt]
        for gi, (g_cls, g_box, regime) in enumerate(gts):
            bucket_gt[regime] += 1
            per_image_counts[regime][1] += 1
            if gi in matched_gt:
                bucket_matched[regime] += 1
                per_image_counts[regime][0] += 1

        per_image.append((img_path.name, dict(per_image_counts)))
        LOGGER.info("imagem=%s gt=%d casados=%d", img_path.name, len(gts), len(matched_gt))

    print(f"{'regime':13s}{'GT':>6s}{'casados':>9s}{'recall':>9s}")
    regime_results = {}
    for regime in ("NANOPLASTIC", "MICROPLASTIC"):
        gt = bucket_gt[regime]
        m = bucket_matched[regime]
        recall = m / gt if gt else float("nan")
        regime_results[regime] = {"gt": gt, "matched": m, "recall": recall}
        print(f"{regime:13s}{gt:6d}{m:9d}{recall:9.3f}")

    # bootstrap por imagem, separado por regime (mesma logica de 14_yolo_test_eval.py)
    rng = np.random.default_rng(SEED)
    n_images = len(per_image)
    ci_results = {}
    for regime in ("NANOPLASTIC", "MICROPLASTIC"):
        recalls = np.empty(N_BOOTSTRAP)
        for b in range(N_BOOTSTRAP):
            idx = rng.integers(0, n_images, size=n_images)
            total_matched = sum(per_image[i][1].get(regime, [0, 0])[0] for i in idx)
            total_gt = sum(per_image[i][1].get(regime, [0, 0])[1] for i in idx)
            recalls[b] = total_matched / total_gt if total_gt else float("nan")
        ci_low, ci_high = np.nanpercentile(recalls, [2.5, 97.5])
        point = regime_results[regime]["recall"]
        ci_results[regime] = {"point": point, "low": float(ci_low), "high": float(ci_high)}
        print(f"IC 95% {regime}: [{ci_low:.3f}, {ci_high:.3f}]")

    save_experiment_summary(EXP_NAME, RUNTIME_METADATA, {
        "model_path": str(MODEL_PATH),
        "n_test_images": n_images,
        "n_unresolved_magnification": n_unresolved_mag,
        "size_regime_recall": regime_results,
        "size_regime_recall_bootstrap_ci95": ci_results,
        "iou_threshold": IOU_THRESHOLD,
        "conf_threshold": CONF_THRESHOLD,
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
    })
    print(f"\nLog completo: {LOG_PATH}")


if __name__ == "__main__":
    main()
