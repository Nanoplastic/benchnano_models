"""Variante de 14_yolo_test_eval.py (bucket por area normalizada) e
23_recall_by_size_regime.py (bucket por regime fisico px->nm) com
max_det=2000. Ambos usam casamento guloso, MESMA classe, IoU>=0.5,
conf=0.15 -- nenhum dos dois passava max_det, entao valia o padrao 300.

Alimenta mainV11.tex linha ~187 (recall por bucket de tamanho e por
regime fisico, single run).

SANIDADE (max_det=300 deve reproduzir):
  14_yolo_test_eval:          small=33.47%  medium=51.61%  large=20.00%
  23_recall_by_size_regime:   NANO=31.10% [15.55,60.40]  MICRO=36.87% [25.48,61.36]

Grava em resultados/71_reeval_bucket_regime_recall_maxdet2000/, nao
sobrescreve 14_yolo_test_eval nem 23_recall_by_size_regime.

Uso:
    python3 -u 71_reeval_bucket_regime_recall_maxdet2000.py
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

EXP_NAME = "71_reeval_bucket_regime_recall_maxdet2000"
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
MAX_DETS = (300, 2000)

PUBLISHED_BUCKET = {"small": 0.3347179711412331, "medium": 0.5161290322580645, "large": 0.2}
PUBLISHED_REGIME = {"NANOPLASTIC": {"recall": 0.31102661596958175, "low": 0.1555494405859077, "high": 0.6040493480251318},
                     "MICROPLASTIC": {"recall": 0.3686868686868687, "low": 0.25483942469339704, "high": 0.613550456316928}}
SANITY_TOL = 0.002

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


def classify_size(area_norm):
    if area_norm < 0.01:
        return "small"
    if area_norm < 0.05:
        return "medium"
    return "large"


def load_gt_both(label_path: Path, W: int, H: int, um_per_px):
    """Retorna (cls_id, xyxy, area_norm, regime) por particula."""
    out = []
    if not label_path.exists():
        return out
    for line in open(label_path):
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        area_norm = w * h
        regime = None
        if um_per_px is not None:
            px_size = max(w * W, h * H)
            size_nm = px_size * um_per_px * 1000.0
            regime = "NANOPLASTIC" if size_nm < NANO_MAX_NM else "MICROPLASTIC"
        out.append((cls_id, yolo_to_xyxy(cx, cy, w, h), area_norm, regime))
    return out


def match_predictions(model, img_path: Path, gts, max_det):
    pred = model.predict(source=str(img_path), imgsz=IMG_SIZE, conf=CONF_THRESHOLD,
                         max_det=max_det, verbose=False)[0]
    preds = []
    if pred.boxes is not None and len(pred.boxes):
        xyxy_norm = pred.boxes.xyxyn.cpu().numpy()
        cls_ids = pred.boxes.cls.cpu().numpy().astype(int)
        for box, c in zip(xyxy_norm, cls_ids):
            preds.append((int(c), tuple(box)))

    candidates = []
    for gi, (g_cls, g_box, _area, _regime) in enumerate(gts):
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


def run(model, max_det):
    bucket_gt, bucket_matched = defaultdict(int), defaultdict(int)
    regime_gt, regime_matched = defaultdict(int), defaultdict(int)
    per_image_bucket = []
    per_image_regime = []

    img_paths = sorted((TEST_DIR / "images/test").glob("*.png"))
    for img_path in img_paths:
        mag = parse_magnification(img_path.stem)
        um_per_px = 100.0 / mag if mag else None
        with Image.open(img_path) as im:
            W, H = im.size
        label_path = TEST_DIR / "labels/test" / f"{img_path.stem}.txt"
        gts = load_gt_both(label_path, W, H, um_per_px)
        if not gts:
            continue

        matched_gt = match_predictions(model, img_path, gts, max_det)

        pb = defaultdict(lambda: [0, 0])
        pr = defaultdict(lambda: [0, 0])
        for gi, (g_cls, g_box, area, regime) in enumerate(gts):
            size_bucket = classify_size(area)
            bucket_gt[size_bucket] += 1
            pb[size_bucket][1] += 1
            if gi in matched_gt:
                bucket_matched[size_bucket] += 1
                pb[size_bucket][0] += 1
            if regime:
                regime_gt[regime] += 1
                pr[regime][1] += 1
                if gi in matched_gt:
                    regime_matched[regime] += 1
                    pr[regime][0] += 1
        per_image_bucket.append((img_path.name, dict(pb)))
        per_image_regime.append((img_path.name, dict(pr)))
        LOGGER.info("max_det=%d imagem=%s gt=%d casados=%d", max_det, img_path.name, len(gts), len(matched_gt))

    bucket_results = {k: {"gt": bucket_gt[k], "matched": bucket_matched[k],
                          "recall": bucket_matched[k] / bucket_gt[k] if bucket_gt[k] else None}
                      for k in ("small", "medium", "large")}

    rng = np.random.default_rng(SEED)
    n_images = len(per_image_regime)
    regime_results = {}
    for regime in ("NANOPLASTIC", "MICROPLASTIC"):
        gt, m = regime_gt[regime], regime_matched[regime]
        recalls = np.empty(N_BOOTSTRAP)
        for b in range(N_BOOTSTRAP):
            idx = rng.integers(0, n_images, size=n_images)
            tm = sum(per_image_regime[i][1].get(regime, [0, 0])[0] for i in idx)
            tg = sum(per_image_regime[i][1].get(regime, [0, 0])[1] for i in idx)
            recalls[b] = tm / tg if tg else float("nan")
        lo, hi = np.nanpercentile(recalls, [2.5, 97.5])
        regime_results[regime] = {"gt": gt, "matched": m, "recall": m / gt if gt else None,
                                  "ci95_low": float(lo), "ci95_high": float(hi)}

    return bucket_results, regime_results


def main():
    if not MODEL_PATH.exists():
        raise SystemExit(f"checkpoint nao existe: {MODEL_PATH}")
    model = YOLO(str(MODEL_PATH))

    results = {}
    for md in MAX_DETS:
        print(f"\n{'='*72}\nmax_det={md}\n{'='*72}")
        bucket, regime = run(model, md)
        results[md] = {"bucket": bucket, "regime": regime}
        print("bucket:", {k: round(v["recall"] * 100, 2) if v["recall"] is not None else None
                          for k, v in bucket.items()})
        print("regime:", {k: (round(v["recall"] * 100, 2), round(v["ci95_low"] * 100, 2), round(v["ci95_high"] * 100, 2))
                          for k, v in regime.items()})

    b300 = results[300]["bucket"]
    r300 = results[300]["regime"]
    print(f"\n{'='*72}\nSANIDADE -- max_det=300 reproduz 14 e 23?\n{'='*72}")
    ok = True
    for k, pub in PUBLISHED_BUCKET.items():
        d = abs(b300[k]["recall"] - pub)
        if d > SANITY_TOL:
            ok = False
        print(f"   bucket {k:7s} publicado={pub*100:.2f}%  obtido={b300[k]['recall']*100:.2f}%  "
              f"dif={d*100:.2f}pp" + ("" if d <= SANITY_TOL else "  <-- DIVERGE"))
    for k, pub in PUBLISHED_REGIME.items():
        d_pt = abs(r300[k]["recall"] - pub["recall"])
        d_lo = abs(r300[k]["ci95_low"] - pub["low"])
        d_hi = abs(r300[k]["ci95_high"] - pub["high"])
        if max(d_pt, d_lo, d_hi) > SANITY_TOL:
            ok = False
        print(f"   regime {k:13s} publicado={pub['recall']*100:.2f}% [{pub['low']*100:.2f},{pub['high']*100:.2f}]  "
              f"obtido={r300[k]['recall']*100:.2f}% [{r300[k]['ci95_low']*100:.2f},{r300[k]['ci95_high']*100:.2f}]"
              + ("" if max(d_pt, d_lo, d_hi) <= SANITY_TOL else "  <-- DIVERGE"))
    print("   ->", "SIM" if ok else "NAO")
    if not ok:
        print("\n   !! nao reproduz -- o resultado de max_det=2000 NAO deve ser usado.")
        return

    print(f"\n{'='*72}\nmax_det=300 (sanidade)  vs  max_det=2000\n{'='*72}")
    b2000, r2000 = results[2000]["bucket"], results[2000]["regime"]
    for k in ("small", "medium", "large"):
        print(f"   bucket {k:7s}  {b300[k]['recall']*100:.2f}%  ->  {b2000[k]['recall']*100:.2f}%")
    for k in ("NANOPLASTIC", "MICROPLASTIC"):
        print(f"   regime {k:13s}  {r300[k]['recall']*100:.2f}% [{r300[k]['ci95_low']*100:.2f},{r300[k]['ci95_high']*100:.2f}]"
              f"  ->  {r2000[k]['recall']*100:.2f}% [{r2000[k]['ci95_low']*100:.2f},{r2000[k]['ci95_high']*100:.2f}]")

    save_experiment_summary(EXP_NAME, RUNTIME_METADATA, {
        "note": "reavaliacao do recall por bucket de tamanho (area_norm, 14_yolo_test_eval) "
                "e por regime fisico (px->nm, 23_recall_by_size_regime) com max_det explicito",
        "model_path": str(MODEL_PATH),
        "conf_threshold": CONF_THRESHOLD, "iou_threshold": IOU_THRESHOLD,
        "sanity_reproduces_published": ok,
        "results": {str(k): v for k, v in results.items()},
    })
    print(f"\nLog completo: {LOG_PATH}")


if __name__ == "__main__":
    main()
