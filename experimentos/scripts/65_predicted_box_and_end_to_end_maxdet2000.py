"""Copia fiel de 20_predicted_box_and_end_to_end.py (single run, mesmos
checkpoints -- runs_v2/microplastic_yolo26_v2/best.pt e checkpoints/size_lora)
com UMA mudanca de inferencia: max_det. Esta e a corrida que de fato alimenta
a Fig. 3 (detector), o painel direito da Fig. 4a (matriz caixa-YOLO, n=1482)
e o funil fim-a-fim -- nao a 07_end_to_end_paired dos experimentos
controlados, que usa os checkpoints de seed e serve só a Tabela 1.

Roda DUAS vezes: primeiro com max_det=300 (sanidade -- deve reproduzir
D=35.4% P|D=90.7% S|D,P=90.8% E=29.2%, n=1482, do summary.json publicado),
depois com max_det=2000. Só grava o resultado de 2000 se a sanidade passar.

Grava em resultados/65_predicted_box_and_end_to_end_maxdet2000/, portanto
NAO sobrescreve 20_predicted_box_and_end_to_end (o publicado).

Uso:
    python3 -u 65_predicted_box_and_end_to_end_maxdet2000.py
"""
import json
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

EXP_NAME = "65_predicted_box_and_end_to_end_maxdet2000"
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

PUBLISHED_SUMMARY = (Path(__file__).resolve().parent.parent / "resultados"
                     / "20_predicted_box_and_end_to_end" / "summary.json")
MAX_DETS = (300, 2000)
SANITY_TOL = 0.002


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


def run_once(max_det: int):
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

        pred = yolo_model.predict(source=str(img_path), imgsz=IMG_SIZE, conf=CONF_THRESHOLD,
                                  max_det=max_det, verbose=False)[0]
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
    out_csv = EXP_DIR / "reports" / f"predicted_box_end_to_end_maxdet{max_det}.csv"

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

    metrics = {
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
        "max_det": max_det,
    }
    return metrics, df, out_csv


def main():
    if not PUBLISHED_SUMMARY.exists():
        raise SystemExit(f"nao existe: {PUBLISHED_SUMMARY}")
    published = json.loads(PUBLISHED_SUMMARY.read_text())["metrics"]

    results = {}
    for md in MAX_DETS:
        print(f"\n{'='*72}\nmax_det={md}\n{'='*72}")
        LOGGER.info("iniciando run_once(max_det=%d)", md)
        metrics, df, out_csv = run_once(md)
        results[md] = (metrics, df, out_csv)

    m300 = results[300][0]
    keys = ["D_rate", "P_rate_given_D", "S_rate_given_DP", "E_rate_strict_end_to_end"]
    print(f"\n{'='*72}\nSANIDADE -- max_det=300 reproduz 20_predicted_box_and_end_to_end?\n{'='*72}")
    ok = True
    for k in keys:
        d = abs(m300[k] - published[k])
        flag = "" if d <= SANITY_TOL else "  <-- DIVERGE"
        if d > SANITY_TOL:
            ok = False
        print(f"   {k:26s} publicado={published[k]*100:6.2f}%  obtido={m300[k]*100:6.2f}%  "
              f"dif={d*100:5.2f}pp{flag}")
    if m300["n_classified_by_smolvlm"] != published["n_classified_by_smolvlm"]:
        print(f"   n_classified_by_smolvlm      publicado={published['n_classified_by_smolvlm']}  "
              f"obtido={m300['n_classified_by_smolvlm']}  <-- DIVERGE")
        ok = False
    print("   ->", "SIM" if ok else "NAO")
    if not ok:
        print("\n   !! nao reproduz -- o resultado de max_det=2000 NAO deve ser usado.")
        return

    m2000, df2000, out_csv2000 = results[2000]
    out_csv2000.parent.mkdir(parents=True, exist_ok=True)
    df2000.to_csv(out_csv2000, index=False)
    print(f"\nCSV detalhado por particula (max_det=2000): {out_csv2000}")

    print(f"\n{'='*72}\nmax_det=300 (sanidade)  vs  max_det=2000\n{'='*72}")
    for k in keys:
        print(f"   {k:26s} {m300[k]*100:6.2f}%  ->  {m2000[k]*100:6.2f}%"
              f"   ({(m2000[k]-m300[k])*100:+.2f} pp)")
    print(f"   n_classified_by_smolvlm      {m300['n_classified_by_smolvlm']:5d}  ->  "
          f"{m2000['n_classified_by_smolvlm']:5d}")

    save_experiment_summary(EXP_NAME, RUNTIME_METADATA, {
        "note": "reavaliacao fim-a-fim do single run com max_det explicito -- alimenta "
                "Fig. 3, o painel direito da Fig. 4a e o funil. O publicado "
                "(20_predicted_box_and_end_to_end) rodou com o padrao 300 do Ultralytics.",
        "published_reference": {k: published[k] for k in keys + ["n_classified_by_smolvlm"]},
        "sanity_reproduces_published": ok,
        "maxdet_300": m300,
        "maxdet_2000": m2000,
        "csv_maxdet_2000": str(out_csv2000),
    })
    print(f"\nLog completo: {LOG_PATH}")


if __name__ == "__main__":
    main()
