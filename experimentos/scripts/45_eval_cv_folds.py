"""Avalia os 5 checkpoints da cross-validation (item 3), cada um SÓ nas
imagens held-out do seu próprio fold (nunca vistas em treino nem em seleção
de checkpoint desse fold) -- agrega os 5 resultados pra ter uma estimativa
de recall por tamanho sem viés sobre as 102 imagens-fonte do dataset
inteiro (155 partículas médias, 30 grandes -- ver 43_build_cv_folds.py),
bem maior que as 31/5 da partição de teste travada única usada no artigo.

Casamento por IoU: casador guloso mesma-classe, IoU>=0.5 (mesma aproximação
de 14_yolo_test_eval.py / 25_bootstrap_yolo_detection_metrics.py -- não é
bit-a-bit idêntico ao validador interno do ultralytics que gerou os números
publicados no artigo).

Resultado marcado como EXPLORATÓRIO/complementar -- o teste travado de 15
imagens continua sendo o resultado principal do artigo; isso só testa se um
conjunto de avaliação maior fecha o intervalo de confiança.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import save_experiment_summary, setup_experiment_logging

from ultralytics import YOLO

EXP_NAME = "45_eval_cv_folds"
EXP_DIR, LOG_PATH, LOGGER, RUNTIME_METADATA = setup_experiment_logging(EXP_NAME, __file__)

ROOT = Path(__file__).resolve().parent.parent
CV_DATASETS_DIR = ROOT / "datasets_yolo26_v2_cv"
CV_RUNS_DIR = ROOT / "runs_v2_cv"
CLASS_NAMES = ["PE", "PET", "PP", "PS"]
K_FOLDS = 5

IMG_SIZE = 1024
CONF_THRESHOLD = 0.15
IOU_THRESHOLD = 0.5
N_BOOTSTRAP = 10_000
SEED = 42

SMALL_THRESHOLD = 0.01
MEDIUM_THRESHOLD = 0.05

# numeros ja publicados no artigo (partição de teste travada, 15 imagens),
# pra comparacao lado a lado -- ver main_pt.tex linha ~767
PUBLISHED_RECALL_CI95 = [0.247, 0.518]
PUBLISHED_N_MEDIUM = 31
PUBLISHED_N_LARGE = 5


def classify_size(area_norm: float) -> str:
    if area_norm < SMALL_THRESHOLD:
        return "small"
    if area_norm < MEDIUM_THRESHOLD:
        return "medium"
    return "large"


def yolo_to_xyxy_px(cx, cy, w, h, W, H):
    return ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)


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


def load_gt(label_path: Path, W: int, H: int):
    out = []
    if not label_path.exists():
        return out
    for line in open(label_path):
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        xyxy = yolo_to_xyxy_px(cx, cy, w, h, W, H)
        out.append({"cls_id": cls_id, "xyxy": xyxy, "size": classify_size(w * h)})
    return out


def match_same_class(gts, preds):
    candidates = []
    for gi, g in enumerate(gts):
        for pi, p in enumerate(preds):
            if g["cls_id"] != p["cls_id"]:
                continue
            iou = iou_xyxy(g["xyxy"], p["xyxy"])
            if iou >= IOU_THRESHOLD:
                candidates.append((iou, gi, pi))
    candidates.sort(reverse=True)
    matched_gt = set()
    matched_pred = set()
    for iou, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
    return matched_gt


def bootstrap_recall_ci(records, size_filter, seed=SEED, n_bootstrap=N_BOOTSTRAP):
    """records: list of dict(source_image, size, matched). Bootstrap por
    imagem-fonte (mesmo padrao do resto do projeto)."""
    sub = [r for r in records if size_filter(r["size"])]
    if not sub:
        return None
    df = pd.DataFrame(sub)
    imgs = df["source_image"].unique()
    groups = {s: g for s, g in df.groupby("source_image")}
    rng = np.random.default_rng(seed)
    recalls = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sampled = rng.choice(imgs, size=len(imgs), replace=True)
        rep = pd.concat([groups[s] for s in sampled], ignore_index=True)
        recalls[i] = rep["matched"].mean()
    ci = np.nanpercentile(recalls, [2.5, 97.5])
    return {
        "n": len(sub),
        "n_images": len(imgs),
        "recall_point": float(df["matched"].mean()),
        "ci95": [float(ci[0]), float(ci[1])],
    }


def main():
    missing = []
    for k in range(K_FOLDS):
        ckpt = CV_RUNS_DIR / f"fold{k}" / "weights" / "best.pt"
        if not ckpt.exists():
            missing.append(str(ckpt))
    if missing:
        LOGGER.error("checkpoints faltando: %s -- rode 44_train_cv_fold.py --fold K pra cada fold antes", missing)
        for m in missing:
            print(f"FALTANDO: {m}")
        sys.exit(1)

    LOGGER.info("=== avaliando %d folds de cross-validation (held-out only) ===", K_FOLDS)

    all_records = []  # source_image, size, matched, fold
    per_fold_summary = []

    for k in range(K_FOLDS):
        ckpt_path = CV_RUNS_DIR / f"fold{k}" / "weights" / "best.pt"
        img_dir = CV_DATASETS_DIR / f"fold{k}" / "images" / "test"
        label_dir = CV_DATASETS_DIR / f"fold{k}" / "labels" / "test"
        model = YOLO(str(ckpt_path))

        img_paths = sorted(img_dir.glob("*.png"))
        n_gt_fold = 0
        n_matched_fold = 0
        for img_path in img_paths:
            image = Image.open(img_path).convert("RGB")
            W, H = image.size
            gts = load_gt(label_dir / f"{img_path.stem}.txt", W, H)

            pred = model.predict(source=str(img_path), imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]
            preds = []
            if pred.boxes is not None and len(pred.boxes):
                xyxy_px = pred.boxes.xyxy.cpu().numpy()
                cls_ids = pred.boxes.cls.cpu().numpy().astype(int)
                for box, c in zip(xyxy_px, cls_ids):
                    preds.append({"cls_id": int(c), "xyxy": tuple(box)})

            matched_gt = match_same_class(gts, preds)
            for gi, g in enumerate(gts):
                is_matched = gi in matched_gt
                all_records.append({
                    "source_image": img_path.name, "fold": k,
                    "size": g["size"], "matched": is_matched,
                })
                n_gt_fold += 1
                n_matched_fold += int(is_matched)

            LOGGER.info("fold=%d imagem=%s gt=%d matched=%d", k, img_path.name, len(gts), len(matched_gt))

        recall_fold = n_matched_fold / n_gt_fold if n_gt_fold else float("nan")
        per_fold_summary.append({"fold": k, "n_images": len(img_paths), "n_gt": n_gt_fold, "recall": recall_fold})
        print(f"fold {k}: {len(img_paths)} imagens held-out, n_gt={n_gt_fold}, recall={recall_fold:.3f}")

    df_all = pd.DataFrame(all_records)
    out_csv = EXP_DIR / "reports" / "cv_pooled_predictions.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(out_csv, index=False)

    print(f"\n=== Recall pooled por tamanho (5 folds, cada imagem avaliada 1x fora do treino) ===")
    bucket_results = {}
    for size in ("small", "medium", "large"):
        res = bootstrap_recall_ci(all_records, lambda s, sz=size: s == sz)
        bucket_results[size] = res
        if res:
            print(f"{size}: n={res['n']} ({res['n_images']} imagens) recall={res['recall_point']:.3f} "
                  f"IC95%=[{res['ci95'][0]:.3f}, {res['ci95'][1]:.3f}]")

    overall = bootstrap_recall_ci(all_records, lambda s: True)
    print(f"\ngeral: n={overall['n']} recall={overall['recall_point']:.3f} IC95%={overall['ci95']}")

    print(f"\n=== Comparação com o número publicado (partição de teste travada, n={PUBLISHED_N_MEDIUM} médio / {PUBLISHED_N_LARGE} grande) ===")
    print(f"publicado (recall geral casado por tamanho): IC95%={PUBLISHED_RECALL_CI95}")
    if bucket_results.get("medium"):
        print(f"CV pooled médio: n={bucket_results['medium']['n']} (vs {PUBLISHED_N_MEDIUM} publicado) IC95%={bucket_results['medium']['ci95']}")
    if bucket_results.get("large"):
        print(f"CV pooled grande: n={bucket_results['large']['n']} (vs {PUBLISHED_N_LARGE} publicado) IC95%={bucket_results['large']['ci95']}")

    save_experiment_summary(EXP_NAME, RUNTIME_METADATA, {
        "note": "resultado exploratorio/complementar -- o teste travado de 15 imagens continua sendo o resultado principal do artigo",
        "k_folds": K_FOLDS,
        "per_fold_summary": per_fold_summary,
        "bucket_results": bucket_results,
        "overall": overall,
        "published_comparison": {
            "published_recall_ci95_overall": PUBLISHED_RECALL_CI95,
            "published_n_medium": PUBLISHED_N_MEDIUM,
            "published_n_large": PUBLISHED_N_LARGE,
        },
        "iou_threshold": IOU_THRESHOLD,
        "conf_threshold": CONF_THRESHOLD,
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
    })
    print(f"\nCSV pooled: {out_csv}")
    print(f"Log completo: {LOG_PATH}")


if __name__ == "__main__":
    main()
