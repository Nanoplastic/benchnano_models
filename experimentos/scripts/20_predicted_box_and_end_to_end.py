"""Condição predicted-box (Camada 1 + Camada 2 juntas) e avaliação fim-a-fim
-- as duas peças do roadmap que dependiam uma da outra, calculadas no mesmo
script porque usam o mesmo dado de base.

Passo a passo, por imagem do teste travado (datasets_yolo26_v2/images/test):

1. Roda o YOLO26L (best.pt retreinado, ver 13_train_yolo26_corrected.py) e
   pega as caixas previstas + classe de polímero prevista.
2. Casa cada caixa prevista com o ground-truth por IoU >= IOU_THRESHOLD,
   SEM exigir classe igual nesse casamento espacial -- D_i (eq. detection_
   correct do artigo) é definido só pela geometria. Casamento guloso (maior
   IoU primeiro), um-para-um.
3. Pra cada par casado espacialmente: P_i = 1 se a classe de polímero
   prevista bate com a real (eq. polymer_correct).
4. Só pros pares com D_i=1 E P_i=1: recorta a região da caixa PREVISTA
   (não a de referência -- é isso que testa o efeito do erro de localização
   do YOLO) com a mesma margem de 10px usada nos recortes de referência, e
   manda pro SmolVLM (checkpoint LoRA, empatou estatisticamente com o full
   fine-tuning mas usa 100x menos parâmetro -- ver 19_compare_size_regime_
   configs.py) decidir NANOPLASTIC/MICROPLASTIC. S_i = 1 se bater com o
   regime de tamanho de referência (mesma fórmula px->nm de sempre).
5. E_i = D_i * P_i * S_i por partícula de referência (eq. end_to_end_correct
   do artigo). Reportado também cada estágio isolado, pra não contar erro
   do YOLO como falha do SmolVLM.

Compara com a condição reference-box (sec:results_size_pilot, script 17) pra
medir o efeito da localização do YOLO na segunda etapa -- essa é a pergunta
que RQ1 faz diretamente.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import (
    CHECKPOINTS_DIR, DEVICE, MODEL_ID, SIZE_CLASSIFY_PROMPT, SIZE_LABELS,
    classify_image, extract_label, load_base_model, save_experiment_summary,
    setup_experiment_logging,
)

from ultralytics import YOLO

EXP_NAME = "20_predicted_box_and_end_to_end"
EXP_DIR, LOG_PATH, LOGGER, RUNTIME_METADATA = setup_experiment_logging(EXP_NAME, __file__)

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "datasets_yolo26_v2"
YOLO_MODEL_PATH = ROOT / "runs_v2" / "microplastic_yolo26_v2" / "weights" / "best.pt"
SIZE_LORA_ADAPTER = CHECKPOINTS_DIR / "size_lora"
CLASS_NAMES = ["PE", "PET", "PP", "PS"]

IMG_SIZE = 1024
CONF_THRESHOLD = 0.15
IOU_THRESHOLD = 0.5
CROP_MARGIN_PX = 10  # mesma margem usada em 01_extract_real_crops.py
N_BOOTSTRAP = 10_000
SEED = 42

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


def load_gt(label_path: Path, W: int, H: int, magnification):
    """Retorna lista de dicts: cls_id, xyxy_px, size_nm, regime."""
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
        xyxy = yolo_to_xyxy_px(cx, cy, w, h, W, H)
        px_size = max(w * W, h * H)
        size_nm = px_size * um_per_px * 1000.0 if um_per_px else None
        regime = None
        if size_nm is not None:
            regime = "NANOPLASTIC" if size_nm < 1000 else "MICROPLASTIC"
        out.append({"cls_id": cls_id, "xyxy": xyxy, "size_nm": size_nm, "regime": regime})
    return out


def main():
    if not YOLO_MODEL_PATH.exists():
        LOGGER.error("checkpoint YOLO nao existe: %s", YOLO_MODEL_PATH)
        sys.exit(1)
    if not SIZE_LORA_ADAPTER.exists():
        LOGGER.error("adapter LoRA nao existe: %s -- rode 17_size_finetune_lora.py primeiro", SIZE_LORA_ADAPTER)
        sys.exit(1)

    LOGGER.info("=== condicao predicted-box + avaliacao fim-a-fim ===")
    yolo_model = YOLO(str(YOLO_MODEL_PATH))

    from peft import PeftModel
    processor, base_model = load_base_model(MODEL_ID)
    vlm_model = PeftModel.from_pretrained(base_model, SIZE_LORA_ADAPTER)
    vlm_model.config.use_cache = True

    img_paths = sorted((TEST_DIR / "images/test").glob("*.png"))

    rows = []
    n_gt_total = 0
    for img_path in img_paths:
        cls_prefix = img_path.stem.split("_", 1)[0]
        original_name = img_path.stem.split("_", 1)[1] if "_" in img_path.stem else img_path.stem
        magnification = parse_magnification(original_name)
        if magnification is None:
            LOGGER.warning("magnificacao nao resolvida para %s -- pulando imagem", img_path.name)
            continue

        image = Image.open(img_path).convert("RGB")
        W, H = image.size

        label_path = TEST_DIR / "labels/test" / f"{img_path.stem}.txt"
        gts = load_gt(label_path, W, H, magnification)
        gts = [g for g in gts if g["regime"] is not None]
        n_gt_total += len(gts)

        pred = yolo_model.predict(source=str(img_path), imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]
        preds = []
        if pred.boxes is not None and len(pred.boxes):
            xyxy_px = pred.boxes.xyxy.cpu().numpy()
            cls_ids = pred.boxes.cls.cpu().numpy().astype(int)
            for box, c in zip(xyxy_px, cls_ids):
                preds.append({"cls_id": int(c), "xyxy": tuple(box)})

        # casamento espacial guloso, SEM exigir classe igual (D_i puramente geometrico)
        candidates = []
        for gi, g in enumerate(gts):
            for pi, p in enumerate(preds):
                iou = iou_xyxy(g["xyxy"], p["xyxy"])
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
            P_i = False
            S_i = None
            pred_regime = None
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
            rows.append({
                "source_image": img_path.name, "class_prefix": cls_prefix,
                "gt_regime": g["regime"], "pred_regime": pred_regime,
                "D_i": D_i, "P_i": P_i, "S_i": S_i, "E_i": E_i,
            })

        LOGGER.info("imagem=%s gt=%d casados=%d", img_path.name, len(gts), len(matched_gt))
        print(f"{img_path.name}: gt={len(gts)} casados_espacialmente={len(matched_gt)}")

    df = pd.DataFrame(rows)
    out_csv = EXP_DIR / "reports" / "predicted_box_end_to_end.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    n = len(df)
    d_rate = df["D_i"].mean()
    p_rate_given_d = df.loc[df["D_i"], "P_i"].mean() if df["D_i"].any() else float("nan")
    s_rate_given_dp = df.loc[df["D_i"] & df["P_i"], "S_i"].mean() if (df["D_i"] & df["P_i"]).any() else float("nan")
    e_rate = df["E_i"].mean()

    print(f"\n=== Resumo (n={n} partículas de referência, {len(img_paths)} imagens) ===")
    print(f"D_i (localização, IoU>=0.5): {d_rate:.3f}")
    print(f"P_i | D_i (polímero correto, dado localizado): {p_rate_given_d:.3f}")
    print(f"S_i | D_i,P_i (tamanho correto, dado localizado+polímero certo): {s_rate_given_dp:.3f}")
    print(f"E_i (fim-a-fim estrito, D*P*S): {e_rate:.3f}")

    n_classified = int((df["D_i"] & df["P_i"]).sum())
    if n_classified > 0:
        sub = df[df["D_i"] & df["P_i"]].copy()
        acc_predicted_box = sub["S_i"].mean()
        print(f"\nAcurácia do SmolVLM SÓ nos casos classificados (condição predicted-box, n={n_classified}): {acc_predicted_box:.3f}")
        print("Matriz de confusão (predicted-box):")
        print(pd.crosstab(sub["gt_regime"], sub["pred_regime"]))

        rng = np.random.default_rng(SEED)
        imgs = sub["source_image"].unique()
        groups = {s: g for s, g in sub.groupby("source_image")}
        accs = np.empty(N_BOOTSTRAP)
        for i in range(N_BOOTSTRAP):
            sampled = rng.choice(imgs, size=len(imgs), replace=True)
            rep = pd.concat([groups[s] for s in sampled], ignore_index=True)
            accs[i] = rep["S_i"].mean()
        ci_low, ci_high = np.nanpercentile(accs, [2.5, 97.5])
        print(f"IC 95% (bootstrap por imagem, {N_BOOTSTRAP} réplicas, {len(imgs)} imagens): [{ci_low:.3f}, {ci_high:.3f}]")
    else:
        acc_predicted_box = float("nan")
        ci_low = ci_high = float("nan")

    save_experiment_summary(EXP_NAME, RUNTIME_METADATA, {
        "n_gt_particles": n,
        "D_rate": float(d_rate),
        "P_rate_given_D": float(p_rate_given_d),
        "S_rate_given_DP": float(s_rate_given_dp) if not pd.isna(s_rate_given_dp) else None,
        "E_rate_strict_end_to_end": float(e_rate),
        "n_classified_by_smolvlm": n_classified,
        "smolvlm_accuracy_predicted_box": float(acc_predicted_box) if not pd.isna(acc_predicted_box) else None,
        "smolvlm_accuracy_ci95": [float(ci_low), float(ci_high)] if not pd.isna(ci_low) else None,
        "iou_threshold": IOU_THRESHOLD,
        "conf_threshold": CONF_THRESHOLD,
        "crop_margin_px": CROP_MARGIN_PX,
    })
    print(f"\nLog completo: {LOG_PATH}")
    print(f"CSV detalhado por partícula: {out_csv}")


if __name__ == "__main__":
    main()
