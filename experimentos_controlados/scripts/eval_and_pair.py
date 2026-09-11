"""Funções de avaliação reaproveitadas pelos 6 scripts `NN_..._10x.py`, mais
o modo `--pair` que roda a avaliação fim-a-fim pareada (YOLO26L rep i +
SmolVLM regime-de-tamanho rep i), espelhando a lógica de
experimentos/scripts/20_predicted_box_and_end_to_end.py.

Reimplementado aqui (em vez de importar o script 20 direto) porque esses
scripts fixam os caminhos de checkpoint como constante de módulo
(YOLO_MODEL_PATH, SIZE_LORA_ADAPTER) -- não dá pra apontar pra um checkpoint
de repetição diferente sem reescrever o arquivo original, o que violaria a
regra de nunca modificar nada em experimentos/scripts/. A lógica de
matching por IoU, cálculo de D_i/P_i/S_i/E_i e o parsing de magnificação são
copiados literalmente do script 20 (mesmo comportamento, só parametrizado).
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctrl_common as cc

sys.path.insert(0, str(cc.EXPERIMENTOS_SCRIPTS))
from vlm_common import (
    MODEL_ID, SIZE_CLASSIFY_PROMPT, SIZE_LABELS, classify_image,
    extract_label, load_base_model,
)

from ultralytics import YOLO

TEST_DIR = cc.DATASETMINA_ROOT / "experimentos" / "datasets_yolo26_v2"
CLASS_NAMES = ["PE", "PET", "PP", "PS"]
IMG_SIZE = 1024
CONF_THRESHOLD = 0.15
IOU_THRESHOLD = 0.5
CROP_MARGIN_PX = 10
N_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 42

MAG_X = re.compile(r"(\d+)\s*[xX]")
MAG_DASH = re.compile(r"^\D*(\d+)-\d+")


# --------------------------------------------------------------------------
# Avaliação padrão do YOLO26L no split de teste travado (espelha a Parte 1
# de 14_yolo_test_eval.py -- métricas agregadas + por classe via model.val())
# --------------------------------------------------------------------------

def yolo_eval_test_set(model_path) -> dict:
    model_path = Path(model_path)
    model = YOLO(str(model_path))

    data_yaml_test = TEST_DIR / "data_test.yaml"
    if not data_yaml_test.exists():
        data_yaml_test.write_text(
            f"path: {TEST_DIR}\n"
            f"train: images/test\n"
            f"val: images/test\n"
            f"nc: {len(CLASS_NAMES)}\n"
            f"names:\n" + "\n".join(f"  - {c}" for c in CLASS_NAMES) + "\n"
        )

    metrics = model.val(data=str(data_yaml_test), split="val", imgsz=IMG_SIZE,
                         conf=CONF_THRESHOLD, iou=0.7, plots=False, verbose=False)
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
    return {"model_path": str(model_path), "aggregate_metrics": agg, "per_class_metrics": per_class}


# --------------------------------------------------------------------------
# Avaliação fim-a-fim pareada (espelha experimentos/scripts/20_predicted_
# box_and_end_to_end.py, parametrizada por checkpoint)
# --------------------------------------------------------------------------

def parse_magnification(name: str):
    m = MAG_X.findall(name)
    if m:
        return int(m[-1])
    m = MAG_DASH.match(name)
    if m:
        return int(m.group(1))
    return None


def _yolo_to_xyxy_px(cx, cy, w, h, W, H):
    return ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)


def _iou_xyxy(a, b):
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


def _load_gt(label_path: Path, W: int, H: int, magnification):
    out = []
    if not label_path.exists():
        return out
    um_per_px = 100.0 / magnification if magnification else None
    for line in open(label_path):
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        xyxy = _yolo_to_xyxy_px(cx, cy, w, h, W, H)
        px_size = max(w * W, h * H)
        size_nm = px_size * um_per_px * 1000.0 if um_per_px else None
        regime = None
        if size_nm is not None:
            regime = "NANOPLASTIC" if size_nm < 1000 else "MICROPLASTIC"
        out.append({"cls_id": cls_id, "xyxy": xyxy, "size_nm": size_nm, "regime": regime})
    return out


def end_to_end_eval(yolo_model_path, size_lora_adapter_path) -> dict:
    yolo_model_path = Path(yolo_model_path)
    size_lora_adapter_path = Path(size_lora_adapter_path)
    if not yolo_model_path.exists():
        raise FileNotFoundError(f"checkpoint YOLO não existe: {yolo_model_path}")
    if not size_lora_adapter_path.exists():
        raise FileNotFoundError(f"adapter LoRA não existe: {size_lora_adapter_path}")

    from peft import PeftModel

    yolo_model = YOLO(str(yolo_model_path))
    processor, base_model = load_base_model(MODEL_ID)
    vlm_model = PeftModel.from_pretrained(base_model, size_lora_adapter_path)
    vlm_model.config.use_cache = True

    img_paths = sorted((TEST_DIR / "images/test").glob("*.png"))
    rows = []

    for img_path in img_paths:
        original_name = img_path.stem.split("_", 1)[1] if "_" in img_path.stem else img_path.stem
        magnification = parse_magnification(original_name)
        if magnification is None:
            continue

        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        label_path = TEST_DIR / "labels/test" / f"{img_path.stem}.txt"
        gts = [g for g in _load_gt(label_path, W, H, magnification) if g["regime"] is not None]

        pred = yolo_model.predict(source=str(img_path), imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]
        preds = []
        if pred.boxes is not None and len(pred.boxes):
            xyxy_px = pred.boxes.xyxy.cpu().numpy()
            cls_ids = pred.boxes.cls.cpu().numpy().astype(int)
            for box, c in zip(xyxy_px, cls_ids):
                preds.append({"cls_id": int(c), "xyxy": tuple(box)})

        candidates = []
        for gi, g in enumerate(gts):
            for pi, p in enumerate(preds):
                iou = _iou_xyxy(g["xyxy"], p["xyxy"])
                if iou >= IOU_THRESHOLD:
                    candidates.append((iou, gi, pi))
        candidates.sort(reverse=True)
        matched_gt, matched_pred = {}, set()
        for iou, gi, pi in candidates:
            if gi in matched_gt or pi in matched_pred:
                continue
            matched_gt[gi] = pi
            matched_pred.add(pi)

        for gi, g in enumerate(gts):
            D_i = gi in matched_gt
            P_i, S_i, pred_regime = False, None, None
            if D_i:
                p = preds[matched_gt[gi]]
                P_i = (p["cls_id"] == g["cls_id"])
                if P_i:
                    x1, y1, x2, y2 = p["xyxy"]
                    x1 = max(0, x1 - CROP_MARGIN_PX)
                    y1 = max(0, y1 - CROP_MARGIN_PX)
                    x2 = min(W, x2 + CROP_MARGIN_PX)
                    y2 = min(H, y2 + CROP_MARGIN_PX)
                    crop = image.crop((x1, y1, x2, y2))
                    raw = classify_image(processor, vlm_model, crop, prompt=SIZE_CLASSIFY_PROMPT)
                    pred_regime = extract_label(raw, labels=SIZE_LABELS)
                    S_i = (pred_regime == g["regime"])
            E_i = bool(D_i and P_i and (S_i is True))
            rows.append({"source_image": img_path.name, "gt_regime": g["regime"],
                         "pred_regime": pred_regime, "D_i": D_i, "P_i": P_i, "S_i": S_i, "E_i": E_i})

    df = pd.DataFrame(rows)
    n = len(df)
    d_rate = float(df["D_i"].mean())
    p_rate_given_d = float(df.loc[df["D_i"], "P_i"].mean()) if df["D_i"].any() else float("nan")
    s_rate_given_dp = float(df.loc[df["D_i"] & df["P_i"], "S_i"].mean()) if (df["D_i"] & df["P_i"]).any() else float("nan")
    e_rate = float(df["E_i"].mean())

    n_classified = int((df["D_i"] & df["P_i"]).sum())
    acc_predicted_box = float(df.loc[df["D_i"] & df["P_i"], "S_i"].mean()) if n_classified > 0 else float("nan")

    return {
        "n_gt_particles": n,
        "D_rate": d_rate,
        "P_rate_given_D": p_rate_given_d,
        "S_rate_given_DP": s_rate_given_dp if not pd.isna(s_rate_given_dp) else None,
        "E_rate_strict_end_to_end": e_rate,
        "n_classified_by_smolvlm": n_classified,
        "smolvlm_accuracy_predicted_box": acc_predicted_box if not pd.isna(acc_predicted_box) else None,
        "per_particle": df.to_dict(orient="records"),
    }


# --------------------------------------------------------------------------
# Modo --pair: roda o fim-a-fim pareado (rep i do YOLO + rep i do SmolVLM
# regime de tamanho) pra cada uma das 10 repetições, depois que os dois
# experimentos de origem já tiverem as 10 repetições prontas.
# --------------------------------------------------------------------------

PAIRED_EXPERIMENT = "07_end_to_end_paired"


def run_pairing():
    state = cc.load_queue_state()
    yolo_reps = cc.load_all_rep_summaries("01_yolo26_main")
    size_reps = cc.load_all_rep_summaries("05_smolvlm_size_lora")

    if len(yolo_reps) < cc.N_REPS or len(size_reps) < cc.N_REPS:
        print(f"[{PAIRED_EXPERIMENT}] esperando: 01_yolo26_main tem {len(yolo_reps)}/10, "
              f"05_smolvlm_size_lora tem {len(size_reps)}/10 repetições. Rode os dois primeiro.")
        return

    for seed in cc.SEEDS:
        if cc.is_rep_done(state, PAIRED_EXPERIMENT, seed):
            print(f"[{PAIRED_EXPERIMENT}] rep {seed:02d} já concluída, pulando.")
            continue

        yolo_summary = next(r for r in yolo_reps if r["repetition_index"] == seed)
        size_summary = next(r for r in size_reps if r["repetition_index"] == seed)
        yolo_ckpt = Path(yolo_summary["metadata"]["best_weights"])
        size_adapter = Path(size_summary["metadata"]["adapter_dir"])

        print(f"\n=== {PAIRED_EXPERIMENT} — repetição {seed:02d}/9 (YOLO+SmolVLM pareados) ===")
        cc.set_current_job(PAIRED_EXPERIMENT, seed)

        metrics = end_to_end_eval(yolo_ckpt, size_adapter)
        per_particle = metrics.pop("per_particle")
        pd.DataFrame(per_particle).to_csv(
            cc.rep_dir(PAIRED_EXPERIMENT, seed) / "predicted_box_end_to_end.csv", index=False)

        print(f"[{PAIRED_EXPERIMENT}] rep {seed:02d}: E_rate={metrics['E_rate_strict_end_to_end']:.3f} "
              f"D_rate={metrics['D_rate']:.3f}")
        cc.save_rep_summary(PAIRED_EXPERIMENT, seed, seed, metrics, extra_metadata={
            "yolo_checkpoint": str(yolo_ckpt), "size_lora_adapter": str(size_adapter),
        })
        cc.mark_rep_done(state, PAIRED_EXPERIMENT, seed)
        cc.clear_current_job()


if __name__ == "__main__":
    run_pairing()
