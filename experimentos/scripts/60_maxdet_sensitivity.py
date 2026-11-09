"""Mede o efeito do limite de deteccoes por imagem (max_det) sobre o recall
do detector no teste travado.

Os Metodos do artigo afirmam que a inferencia usa "at most 2,000 detections
per image". Nenhum dos scripts de avaliacao (14, 20, 23, 42) passa `max_det`
para `model.predict()`, entao todos rodaram com o padrao do Ultralytics,
que e 300 (ultralytics/cfg/default.yaml). Os 2.000 eram parametro de TREINO.

Duas micrografias do teste travado pararam exatamente em 300 predicoes --
PP_PP-1000x1 (1.268 referencias) e PS_PS-100x (957) -- que sao as duas mais
densas e somam 2.225 das 4.610 particulas do teste. Nelas o recall maximo
alcancavel era 23,7% e 31,3%, independentemente do modelo.

O limite e sobreponivel: para modelos end2end o predictor propaga o valor
para a cabeca (ultralytics/engine/predictor.py:426,
`set_head_attr(max_det=max(args.max_det, 300))`), e o top-k usa
k = min(max_det, n_anchors). Entao 300 e o padrao, nao um teto da
arquitetura.

Este script roda o MESMO checkpoint e a MESMA logica de casamento de
14_yolo_test_eval.py (IoU>=0.5, guloso, mesma classe) e de
20_predicted_box_and_end_to_end.py (casamento espacial, classe-agnostico),
variando so o max_det.

SANIDADE: com max_det=300 deve reproduzir o recall global de 0.3358 e a
taxa D de 0.3544 ja publicados. Se nao reproduzir, o resto nao vale.

Uso:
    python3 60_maxdet_sensitivity.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import save_experiment_summary, setup_experiment_logging

from ultralytics import YOLO

EXP_NAME = "60_maxdet_sensitivity"
EXP_DIR, LOG_PATH, LOGGER, RUNTIME_METADATA = setup_experiment_logging(EXP_NAME, __file__)

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "datasets_yolo26_v2"
MODEL_PATH = ROOT / "runs_v2" / "microplastic_yolo26_v2" / "weights" / "best.pt"
CLASS_NAMES = ["PE", "PET", "PP", "PS"]

IMG_SIZE = 1024
CONF_THRESHOLD = 0.15
IOU_THRESHOLD = 0.5
MAX_DET_VALUES = [300, 2000]

# valores publicados, com o max_det padrao de 300
PUBLISHED = {"global_recall": 0.33579, "d_rate": 0.35445}


def iou_xyxy(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(label_path: Path):
    gts = []
    if not label_path.exists():
        return gts
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c = int(p[0])
        cx, cy, w, h = (float(v) for v in p[1:5])
        gts.append((c, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    return gts


def match(gts, preds, class_aware: bool):
    """Casamento guloso por IoU decrescente, cada referencia usada uma vez."""
    cands = []
    for gi, (g_cls, g_box) in enumerate(gts):
        for pi, (p_cls, p_box) in enumerate(preds):
            if class_aware and p_cls != g_cls:
                continue
            iou = iou_xyxy(g_box, p_box)
            if iou >= IOU_THRESHOLD:
                cands.append((iou, gi, pi))
    cands.sort(reverse=True)
    mg, mp = set(), set()
    for _, gi, pi in cands:
        if gi in mg or pi in mp:
            continue
        mg.add(gi)
        mp.add(pi)
    return mg


def run(max_det: int):
    model = YOLO(str(MODEL_PATH))
    img_dir = TEST_DIR / "images" / "test"
    lbl_dir = TEST_DIR / "labels" / "test"

    per_image, tot_gt = [], 0
    n_aware = n_agnostic = 0
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in (".png", ".jpg", ".jpeg", ".tif"):
            continue
        gts = load_gt(lbl_dir / f"{img.stem}.txt")
        if not gts:
            continue
        pred = model.predict(source=str(img), imgsz=IMG_SIZE, conf=CONF_THRESHOLD,
                             max_det=max_det, verbose=False)[0]
        preds = []
        if pred.boxes is not None and len(pred.boxes):
            for box, c in zip(pred.boxes.xyxyn.cpu().numpy(),
                              pred.boxes.cls.cpu().numpy().astype(int)):
                preds.append((int(c), tuple(box)))
        ma = match(gts, preds, class_aware=True)
        mg = match(gts, preds, class_aware=False)
        n_aware += len(ma)
        n_agnostic += len(mg)
        tot_gt += len(gts)
        per_image.append({"image": img.name, "n_gt": len(gts), "n_pred": len(preds),
                          "matched_class_aware": len(ma), "matched_spatial": len(mg),
                          "truncated": len(preds) >= max_det})
    return {"max_det": max_det, "n_gt": tot_gt,
            "global_recall": n_aware / tot_gt, "d_rate": n_agnostic / tot_gt,
            "per_image": per_image}


def main():
    results = {}
    for md in MAX_DET_VALUES:
        LOGGER.info("avaliando com max_det=%d", md)
        r = run(md)
        results[md] = r
        trunc = [p["image"] for p in r["per_image"] if p["truncated"]]
        print(f"\nmax_det={md:5d}  recall global={r['global_recall']:.4f}  "
              f"taxa D={r['d_rate']:.4f}  predicoes totais="
              f"{sum(p['n_pred'] for p in r['per_image']):5d}  truncadas={len(trunc)}")
        for t in trunc:
            print(f"               truncada: {t}")

    base = results[300]
    ok = (abs(base["global_recall"] - PUBLISHED["global_recall"]) < 1e-3 and
          abs(base["d_rate"] - PUBLISHED["d_rate"]) < 1e-3)
    print("\n" + "=" * 70)
    print("SANIDADE -- max_det=300 reproduz o publicado?", "SIM" if ok else "NAO")
    print(f"   recall global  publicado={PUBLISHED['global_recall']:.4f}  obtido={base['global_recall']:.4f}")
    print(f"   taxa D         publicado={PUBLISHED['d_rate']:.4f}  obtido={base['d_rate']:.4f}")
    if not ok:
        print("\n   !! nao reproduz -- o resultado com max_det=2000 NAO deve ser usado.")
        return

    hi = results[2000]
    print("\n" + "=" * 70)
    print("EFEITO DE max_det=2000")
    print(f"   recall global  {base['global_recall']*100:5.2f}%  ->  {hi['global_recall']*100:5.2f}%"
          f"   ({(hi['global_recall']-base['global_recall'])*100:+.2f} pp)")
    print(f"   taxa D         {base['d_rate']*100:5.2f}%  ->  {hi['d_rate']*100:5.2f}%"
          f"   ({(hi['d_rate']-base['d_rate'])*100:+.2f} pp)")
    print("\n   por micrografia truncada em 300:")
    bi = {p["image"]: p for p in base["per_image"]}
    for p in hi["per_image"]:
        b = bi[p["image"]]
        if not b["truncated"]:
            continue
        print(f"   {p['image']:24s} gt={p['n_gt']:5d}  pred {b['n_pred']:4d}->{p['n_pred']:5d}"
              f"  recall {b['matched_spatial']/p['n_gt']*100:5.2f}% -> {p['matched_spatial']/p['n_gt']*100:5.2f}%")

    save_experiment_summary(EXP_NAME, RUNTIME_METADATA, {
        "note": "sensibilidade ao max_det. Os Metodos do artigo afirmam 2.000 deteccoes por "
                "imagem, mas os scripts de avaliacao nao passam max_det e rodaram com o padrao "
                "300 do Ultralytics. Duas micrografias do teste travado foram truncadas.",
        "model_path": str(MODEL_PATH),
        "published_with_max_det_300": PUBLISHED,
        "sanity_reproduces_published": ok,
        "conf_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
        "results": {str(k): v for k, v in results.items()},
    })
    print(f"\nLog completo: {LOG_PATH}")


if __name__ == "__main__":
    main()
