"""Camada 3 (`sec:digital_representation`) para a condição predicted-box: a
mesma avaliação fim a fim de `20_predicted_box_and_end_to_end.py`, mas
persistindo um `ParticleRecord` por partícula de referência em vez de só as
taxas agregadas.

Por que um script novo, e não editar o 20: o 20 já sustenta os números
publicados no artigo (D_rate, P_rate, S_rate, E_rate da Tabela
`smolvlm_results`/decomposição fim a fim) -- não sobrescrever esse script nem
seu CSV de saída (`reports/predicted_box_end_to_end.csv`) é a mesma regra que
já seguimos pros outros experimentos. Este script recalcula exatamente a
mesma coisa (mesmos checkpoints, mesmo limiar de confiança/IoU, mesma lógica
de casamento) e imprime os mesmos quatro números no final -- rodar os dois
deve dar D/P/S/E idênticos, é a forma de conferir que a Camada 3 não mudou a
análise, só persistiu campos que antes eram descartados.

O que muda de fato, em relação ao 20:
  - a caixa PREVISTA e a confiança do YOLO por partícula são mantidas (antes
    eram descartadas depois de decidir D_i/P_i);
  - cada partícula de referência ganha um `particle_index` estável (posição
    no arquivo de rótulo de teste daquela imagem). NÃO é o id de anotação
    original do MiNa: uma tentativa de recuperar esse id casando por IoU
    contra os JSONs COCO brutos (`MPDataset/Full_Images/COCO Format`) foi
    testada e descartada -- em imagens densas a contagem de anotações não
    bate com o número de partículas do split de teste (mesma classe de
    colisão de nome de arquivo que o artigo já documenta ter corrigido em
    outro ponto do pipeline), então o casamento por IoU saía errado pra
    ~80% das partículas. `particle_index` é estável entre execuções porque
    o arquivo de rótulo de teste é estático, só não é rastreável até uma
    anotação MiNa fora deste split;
  - tudo isso vira um `ParticleRecord` (ver `particle_record.py`) e é
    ACRESCENTADO (nunca sobrescrito) a
    `resultados/particle_records/predicted_box_end_to_end.jsonl`, com um
    `run_id` próprio por execução -- reprocessar com um checkpoint novo, ou
    um prompt novo, não apaga os registros de execuções anteriores.

Uso:
    python3 55_predicted_box_and_end_to_end_records.py
"""
import re
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from particle_record import (
    ParticleRecord, PARTICLE_RECORDS_DIR, append_records,
    collect_model_versions, evaluation_status, make_run_id,
)
from vlm_common import (
    CHECKPOINTS_DIR, MODEL_ID, SIZE_CLASSIFY_PROMPT, SIZE_LABELS,
    classify_image, extract_label, load_base_model, save_experiment_summary,
    setup_experiment_logging,
)

from ultralytics import YOLO

EXP_NAME = "55_predicted_box_and_end_to_end_records"
EXP_DIR, LOG_PATH, LOGGER, RUNTIME_METADATA = setup_experiment_logging(EXP_NAME, __file__)

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "datasets_yolo26_v2"
YOLO_MODEL_PATH = ROOT / "runs_v2" / "microplastic_yolo26_v2" / "weights" / "best.pt"
SIZE_LORA_ADAPTER = CHECKPOINTS_DIR / "size_lora"
CLASS_NAMES = ["PE", "PET", "PP", "PS"]

IMG_SIZE = 1024
CONF_THRESHOLD = 0.15
IOU_THRESHOLD = 0.5
CROP_MARGIN_PX = 10

OUT_PATH = PARTICLE_RECORDS_DIR / "predicted_box_end_to_end.jsonl"

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

    LOGGER.info("=== condicao predicted-box + avaliacao fim-a-fim, com registros de particula (Camada 3) ===")
    model_versions = collect_model_versions({
        "yolo_checkpoint": str(YOLO_MODEL_PATH),
        "size_lora_adapter": str(SIZE_LORA_ADAPTER),
        "model_id": MODEL_ID,
        "conf_threshold": CONF_THRESHOLD,
        "iou_threshold": IOU_THRESHOLD,
    })
    run_id = make_run_id(model_versions, tag=EXP_NAME)
    LOGGER.info("run_id=%s", run_id)

    yolo_model = YOLO(str(YOLO_MODEL_PATH))

    from peft import PeftModel
    processor, base_model = load_base_model(MODEL_ID)
    vlm_model = PeftModel.from_pretrained(base_model, SIZE_LORA_ADAPTER)
    vlm_model.config.use_cache = True

    img_paths = sorted((TEST_DIR / "images/test").glob("*.png"))

    rows = []
    records = []
    for img_path in img_paths:
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

        pred = yolo_model.predict(source=str(img_path), imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]
        preds = []
        if pred.boxes is not None and len(pred.boxes):
            xyxy_px = pred.boxes.xyxy.cpu().numpy()
            cls_ids = pred.boxes.cls.cpu().numpy().astype(int)
            confs = pred.boxes.conf.cpu().numpy()
            for box, c, conf in zip(xyxy_px, cls_ids, confs):
                preds.append({"cls_id": int(c), "xyxy": tuple(float(v) for v in box), "conf": float(conf)})

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
            P_i, S_i, pred_regime = False, None, None
            box_pred, conf_pred, polymer_pred = None, None, None
            if D_i:
                p = preds[matched_gt[gi]]
                box_pred, conf_pred = p["xyxy"], p["conf"]
                polymer_pred = CLASS_NAMES[p["cls_id"]]
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
            status = evaluation_status(D_i, P_i, S_i)

            rows.append({"source_image": img_path.name, "gt_regime": g["regime"],
                         "pred_regime": pred_regime, "D_i": D_i, "P_i": P_i, "S_i": S_i, "E_i": E_i})

            records.append(ParticleRecord(
                source_image=img_path.name,
                particle_index=gi,
                magnification=float(magnification),
                box_ref_xyxy=tuple(float(v) for v in g["xyxy"]),
                polymer_label_ref=CLASS_NAMES[g["cls_id"]],
                size_nm_ref=g["size_nm"],
                size_regime_ref=g["regime"],
                detected=D_i,
                box_pred_xyxy=box_pred,
                polymer_label_pred=polymer_pred,
                detector_confidence=conf_pred,
                polymer_correct=P_i,
                size_regime_pred=pred_regime,
                size_correct=S_i,
                end_to_end_correct=E_i,
                evaluation_status=status,
                run_id=run_id,
                created_at_utc=model_versions["timestamp_utc"],
                model_versions=model_versions,
            ))

        LOGGER.info("imagem=%s gt=%d casados=%d", img_path.name, len(gts), len(matched_gt))
        print(f"{img_path.name}: gt={len(gts)} casados_espacialmente={len(matched_gt)}")

    out_records_path = append_records(records, OUT_PATH)

    df = pd.DataFrame(rows)
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
    print("\nEstes quatro números devem bater com os de "
          "20_predicted_box_and_end_to_end.py -- se não baterem, a mudança "
          "introduziu uma diferença analítica e não deve ser usada sem investigar.")

    n_classified = int((df["D_i"] & df["P_i"]).sum())
    acc_predicted_box = df.loc[df["D_i"] & df["P_i"], "S_i"].mean() if n_classified > 0 else float("nan")

    save_experiment_summary(EXP_NAME, RUNTIME_METADATA, {
        "run_id": run_id,
        "n_gt_particles": n,
        "D_rate": float(d_rate),
        "P_rate_given_D": float(p_rate_given_d),
        "S_rate_given_DP": float(s_rate_given_dp) if not pd.isna(s_rate_given_dp) else None,
        "E_rate_strict_end_to_end": float(e_rate),
        "n_classified_by_smolvlm": n_classified,
        "smolvlm_accuracy_predicted_box": float(acc_predicted_box) if not pd.isna(acc_predicted_box) else None,
        "particle_records_path": str(out_records_path),
    })
    print(f"\nLog completo: {LOG_PATH}")
    print(f"Registros de partícula (Camada 3, acrescentados, nunca sobrescritos): {out_records_path}")


if __name__ == "__main__":
    main()
