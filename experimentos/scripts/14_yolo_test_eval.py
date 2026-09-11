"""Avaliação final do YOLO26L v2 (retreinado sem vazamento, ver
13_train_yolo26_corrected.py) no split de teste travado -- 15 imagens em
datasets_yolo26_v2/images/test, nunca vistas em treino/val. Rodar só DEPOIS
que 13_train_yolo26_corrected.py terminar e salvar runs_v2/microplastic_
yolo26_v2/weights/best.pt.

Duas partes:

1. Métricas padrão via `model.val()` do Ultralytics (precision/recall/mAP por
   classe e agregado) -- primeira avaliação de verdade em dado nunca visto
   pelo modelo.

2. Recall por tamanho de partícula com casamento por IoU de verdade,
   corrigindo o bug de scripts_thiago/test.py (`analyze_particle_size_recall`
   somava `len(prediction.boxes)` -- total de caixas da imagem -- em TODOS os
   buckets de tamanho presentes, sem casar cada detecção com o ground-truth
   do tamanho certo, deixando o "recall" reportado sem sentido). Aqui cada
   caixa prevista é casada 1-para-1 com o ground-truth de MESMA CLASSE por
   IoU >= IOU_THRESHOLD (guloso, maior IoU primeiro), e o recall é calculado
   por bucket a partir desse casamento.

Amostra pequena (n=15 imagens) -- ver Jain cap. 13.9 (Determining Sample
Size) e cap. 13.4 (Comparing Two Alternatives), citados no
graphify-out/GRAPH_REPORT.md como leitura já feita neste projeto: qualquer
número aqui vem com incerteza grande e não deve ser tratado como definitivo
sem um intervalo de confiança. Por isso a Parte 1 também roda um bootstrap
por IMAGEM (não por instância) para reportar IC 95% em precision/recall/
mAP50, na mesma lógica de unidade de reamostragem já usada em
11_compare_experiments_statistical.py.
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import save_experiment_summary, setup_experiment_logging

from ultralytics import YOLO

EXP_NAME = "14_yolo_test_eval"
EXP_DIR, LOG_PATH, LOGGER, RUNTIME_METADATA = setup_experiment_logging(EXP_NAME, __file__)

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "datasets_yolo26_v2"
MODEL_PATH = ROOT / "runs_v2" / "microplastic_yolo26_v2" / "weights" / "best.pt"
CLASS_NAMES = ["PE", "PET", "PP", "PS"]

IMG_SIZE = 1024
CONF_THRESHOLD = 0.15
IOU_THRESHOLD = 0.5
N_BOOTSTRAP = 10_000
SEED = 42

# mesmos limiares de scripts_thiago/test.py, pra comparabilidade
SMALL_THRESHOLD = 0.01
MEDIUM_THRESHOLD = 0.05


def classify_size(area_norm: float) -> str:
    if area_norm < SMALL_THRESHOLD:
        return "small"
    if area_norm < MEDIUM_THRESHOLD:
        return "medium"
    return "large"


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


def load_gt(label_path: Path):
    if not label_path.exists():
        return []
    out = []
    for line in open(label_path):
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        out.append((cls_id, yolo_to_xyxy(cx, cy, w, h), w * h))
    return out


def match_predictions(model, img_path: Path, gts):
    """Roda o modelo numa imagem e casa predicoes com GT por IoU (mesma classe)."""
    pred = model.predict(source=str(img_path), imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]

    preds = []
    if pred.boxes is not None and len(pred.boxes):
        xyxy_norm = pred.boxes.xyxyn.cpu().numpy()
        cls_ids = pred.boxes.cls.cpu().numpy().astype(int)
        for box, c in zip(xyxy_norm, cls_ids):
            preds.append((int(c), tuple(box)))

    candidates = []
    for gi, (g_cls, g_box, _) in enumerate(gts):
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

    n_fp = len(preds) - len(matched_pred)
    return matched_gt, n_fp, len(preds)


def main():
    if not MODEL_PATH.exists():
        msg = f"Checkpoint nao existe em {MODEL_PATH} -- rode 13_train_yolo26_corrected.py primeiro."
        LOGGER.error(msg)
        print(msg)
        sys.exit(1)

    LOGGER.info("=== avaliacao final YOLO26L v2 no test set travado (15 imagens) ===")
    LOGGER.info("model_path=%s", MODEL_PATH)

    model = YOLO(str(MODEL_PATH))

    # --- Parte 1: metricas padrao via model.val() ------------------------
    data_yaml_test = TEST_DIR / "data_test.yaml"
    data_yaml_test.write_text(
        f"path: {TEST_DIR}\n"
        f"train: images/test\n"
        f"val: images/test\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n" + "\n".join(f"  - {c}" for c in CLASS_NAMES) + "\n"
    )
    print("=== Metricas padrao (model.val) no test set travado (15 imagens) ===")
    metrics = model.val(data=str(data_yaml_test), split="val", imgsz=IMG_SIZE,
                         conf=CONF_THRESHOLD, iou=0.7, plots=True)
    agg = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }
    per_class = {}
    for i, name in metrics.names.items():
        per_class[name] = {
            "precision": float(metrics.box.p[i]),
            "recall": float(metrics.box.r[i]),
            "ap50": float(metrics.box.ap50[i]),
            "ap50_95": float(metrics.box.ap[i]),
        }
        print(f"  {name:6s} P={metrics.box.p[i]:.3f}  R={metrics.box.r[i]:.3f}  "
              f"AP50={metrics.box.ap50[i]:.3f}  AP50-95={metrics.box.ap[i]:.3f}")
    print(f"Agregado: {agg}")
    LOGGER.info("metricas_agregadas=%s", agg)
    LOGGER.info("metricas_por_classe=%s", per_class)

    # --- Parte 2: recall por tamanho com casamento por IoU + bootstrap ---
    print(f"\n=== Recall por tamanho de particula (IoU >= {IOU_THRESHOLD}, casamento real) ===")
    LOGGER.info("iniciando recall por tamanho com casamento por IoU (threshold=%.2f)", IOU_THRESHOLD)

    bucket_gt = defaultdict(int)
    bucket_matched = defaultdict(int)
    per_image_recall = []  # (image_name, matched_total, gt_total) -- p/ bootstrap por imagem

    img_paths = sorted((TEST_DIR / "images/test").glob("*.png"))
    for img_path in img_paths:
        label_path = TEST_DIR / "labels/test" / f"{img_path.stem}.txt"
        gts = load_gt(label_path)
        if not gts:
            continue

        matched_gt, n_fp, n_pred = match_predictions(model, img_path, gts)

        for gi, (g_cls, g_box, area) in enumerate(gts):
            size = classify_size(area)
            bucket_gt[size] += 1
            if gi in matched_gt:
                bucket_matched[size] += 1

        per_image_recall.append((img_path.name, len(matched_gt), len(gts)))
        LOGGER.info("imagem=%s gt=%d casados=%d falsos_positivos=%d predicoes=%d",
                    img_path.name, len(gts), len(matched_gt), n_fp, n_pred)

    print(f"{'tamanho':8s}{'GT':>6s}{'casados':>9s}{'recall':>9s}")
    size_results = {}
    for size in ("small", "medium", "large"):
        gt = bucket_gt[size]
        m = bucket_matched[size]
        recall = m / gt if gt else float("nan")
        size_results[size] = {"gt": gt, "matched": m, "recall": recall}
        print(f"{size:8s}{gt:6d}{m:9d}{recall:9.3f}")

    # bootstrap por imagem (nao por instancia) para IC 95% do recall global
    rng = np.random.default_rng(SEED)
    n_images = len(per_image_recall)
    recalls = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n_images, size=n_images)
        sampled = [per_image_recall[i] for i in idx]
        total_matched = sum(s[1] for s in sampled)
        total_gt = sum(s[2] for s in sampled)
        recalls[b] = total_matched / total_gt if total_gt else float("nan")
    ci_low, ci_high = np.nanpercentile(recalls, [2.5, 97.5])
    point_recall = sum(s[1] for s in per_image_recall) / sum(s[2] for s in per_image_recall)
    print(f"\nRecall global (casamento por IoU): {point_recall:.3f}  "
          f"IC 95% (bootstrap por imagem, n_img={n_images}, {N_BOOTSTRAP} replicas): "
          f"[{ci_low:.3f}, {ci_high:.3f}]")
    LOGGER.info("recall_global=%.4f ic95_low=%.4f ic95_high=%.4f n_imagens=%d",
                point_recall, ci_low, ci_high, n_images)

    save_experiment_summary(EXP_NAME, RUNTIME_METADATA, {
        "model_path": str(MODEL_PATH),
        "n_test_images": n_images,
        "aggregate_metrics": agg,
        "per_class_metrics": per_class,
        "size_stratified_recall": size_results,
        "global_recall_bootstrap_ci95": {"point": point_recall, "low": float(ci_low), "high": float(ci_high)},
        "iou_threshold": IOU_THRESHOLD,
        "conf_threshold": CONF_THRESHOLD,
    })
    print(f"\nLog completo: {LOG_PATH}")


if __name__ == "__main__":
    main()
