"""Análise de erro por classe -- por que PE tem a menor precisão (15,2%) e a
maior recall (85,7%), e por que PS tem o menor AP@0,5 (21,6%) e recall de
26,6%) no detector YOLO26L ORIGINAL (Discussão, main_pt.tex linha ~766).

Roda no checkpoint ORIGINAL (runs_v2/microplastic_yolo26_v2/weights/best.pt),
não no SAHI fine-tunado do experimento 41 -- é o resultado publicado que está
sendo explicado aqui, não a melhoria proposta.

Casamento por IoU: casador guloso mesma-classe, IoU>=0.5 (mesma aproximação
já usada em 25_bootstrap_yolo_detection_metrics.py e
37_fig_yolo_qualitative_examples.py). Isso NÃO é bit-a-bit idêntico ao
validador interno do ultralytics que gerou os números exatos citados no
texto (model.val() em 14_yolo_test_eval.py) -- os totais agregados aqui
devem bater em ordem de grandeza, não em decimal exato.

Duas hipóteses testadas, marcadas como EXPLORATÓRIAS (geram evidência para
decidir o que escrever na Discussão, não são resultado confirmatório do
artigo):

1. PE, falso positivo: é confusão com outro polímero (a caixa cai em cima de
   uma partícula real de outra classe) ou detecção espúria em fundo/artefato
   (sem nenhum GT de nenhuma classe por perto)?
2. PS, falso negativo: é concentrado em partículas pequenas (viés de tamanho
   já documentado no artigo pra recall geral) e/ou é "sem proposta nenhuma
   ali" vs. "detectado só que suprimido/com outra classe"?
"""
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import OKABE_ITO, apply_style
from vlm_common import save_experiment_summary, setup_experiment_logging

from ultralytics import YOLO

EXP_NAME = "42_class_error_analysis"
EXP_DIR, LOG_PATH, LOGGER, RUNTIME_METADATA = setup_experiment_logging(EXP_NAME, __file__)

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "datasets_yolo26_v2"
MODEL_PATH = ROOT / "runs_v2" / "microplastic_yolo26_v2" / "weights" / "best.pt"
CLASS_NAMES = ["PE", "PET", "PP", "PS"]

IMG_SIZE = 1024
CONF_THRESHOLD = 0.15
IOU_THRESHOLD = 0.5
CROP_MARGIN_PX = 10
N_GALLERY = 6
N_BOOTSTRAP = 10_000
SEED = 42

apply_style()


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
        area_norm = w * h
        out.append({"cls_id": cls_id, "xyxy": xyxy, "area_norm": area_norm})
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
    matched_gt, matched_pred = {}, {}
    for iou, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt[gi] = pi
        matched_pred[pi] = gi
    return matched_gt, matched_pred


def crop_with_margin(image, xyxy, W, H, margin=CROP_MARGIN_PX):
    x1, y1, x2, y2 = xyxy
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(W, x2 + margin)
    y2 = min(H, y2 + margin)
    return image.crop((x1, y1, x2, y2))


def bootstrap_median_diff_by_image(records_a, records_b, seed=SEED, n_bootstrap=N_BOOTSTRAP):
    """records_a/b: list of (source_image, value). Bootstrap pareado por
    imagem-fonte na diferenca de mediana (b - a), reamostrando a UNIAO de
    imagens presentes em qualquer um dos dois grupos."""
    df_a = pd.DataFrame(records_a, columns=["source_image", "value"])
    df_b = pd.DataFrame(records_b, columns=["source_image", "value"])
    imgs = sorted(set(df_a["source_image"]) | set(df_b["source_image"]))
    groups_a = {s: g["value"].values for s, g in df_a.groupby("source_image")}
    groups_b = {s: g["value"].values for s, g in df_b.groupby("source_image")}
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sampled = rng.choice(imgs, size=len(imgs), replace=True)
        vals_a = np.concatenate([groups_a[s] for s in sampled if s in groups_a]) if any(s in groups_a for s in sampled) else np.array([])
        vals_b = np.concatenate([groups_b[s] for s in sampled if s in groups_b]) if any(s in groups_b for s in sampled) else np.array([])
        diffs[i] = (np.median(vals_b) if len(vals_b) else np.nan) - (np.median(vals_a) if len(vals_a) else np.nan)
    ci = np.nanpercentile(diffs, [2.5, 97.5])
    point = (np.median(df_b["value"]) if len(df_b) else float("nan")) - (np.median(df_a["value"]) if len(df_a) else float("nan"))
    return {
        "diff_median_point": float(point),
        "ci95": [float(ci[0]), float(ci[1])],
        "excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
    }


def main():
    if not MODEL_PATH.exists():
        LOGGER.error("checkpoint nao existe: %s", MODEL_PATH)
        sys.exit(1)

    LOGGER.info("=== analise de erro por classe (PE falso positivo, PS falso negativo) ===")
    model = YOLO(str(MODEL_PATH))

    img_paths = sorted((TEST_DIR / "images/test").glob("*.png"))

    pe_fp_rows = []
    ps_rows = []  # FN e TP juntos, coluna status

    pe_fp_gallery_candidates = []  # (confidence, img_path, xyxy)
    ps_fn_gallery_candidates = []  # (area_norm, img_path, xyxy)

    for img_path in img_paths:
        image = Image.open(img_path).convert("RGB")
        W, H = image.size
        label_path = TEST_DIR / "labels/test" / f"{img_path.stem}.txt"
        gts = load_gt(label_path, W, H)

        pred = model.predict(source=str(img_path), imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]
        preds = []
        if pred.boxes is not None and len(pred.boxes):
            xyxy_px = pred.boxes.xyxy.cpu().numpy()
            cls_ids = pred.boxes.cls.cpu().numpy().astype(int)
            confs = pred.boxes.conf.cpu().numpy()
            for box, c, conf in zip(xyxy_px, cls_ids, confs):
                x1, y1, x2, y2 = box
                area_norm = ((x2 - x1) / W) * ((y2 - y1) / H)
                preds.append({"cls_id": int(c), "xyxy": tuple(box), "conf": float(conf), "area_norm": area_norm})

        matched_gt, matched_pred = match_same_class(gts, preds)

        # --- PE: falsos positivos ---
        for pi, p in enumerate(preds):
            if p["cls_id"] != CLASS_NAMES.index("PE"):
                continue
            if pi in matched_pred:
                continue
            best_iou, best_cls = 0.0, None
            for g in gts:
                if g["cls_id"] == p["cls_id"]:
                    continue
                iou = iou_xyxy(p["xyxy"], g["xyxy"])
                if iou > best_iou:
                    best_iou, best_cls = iou, CLASS_NAMES[g["cls_id"]]
            pe_fp_rows.append({
                "source_image": img_path.name, "confidence": p["conf"],
                "box_area_norm": p["area_norm"], "best_iou_other_class_gt": best_iou,
                "overlapping_class": best_cls,
            })
            pe_fp_gallery_candidates.append((p["conf"], img_path, p["xyxy"], W, H))

        # --- PS: falsos negativos vs. verdadeiros positivos ---
        for gi, g in enumerate(gts):
            if g["cls_id"] != CLASS_NAMES.index("PS"):
                continue
            is_matched = gi in matched_gt
            any_pred_overlap = False
            if not is_matched:
                for p in preds:
                    if iou_xyxy(g["xyxy"], p["xyxy"]) > 0:
                        any_pred_overlap = True
                        break
            ps_rows.append({
                "source_image": img_path.name, "status": "TP" if is_matched else "FN",
                "box_area_norm": g["area_norm"],
                "any_pred_overlap": any_pred_overlap if not is_matched else None,
            })
            if not is_matched:
                ps_fn_gallery_candidates.append((g["area_norm"], img_path, g["xyxy"], W, H))

        LOGGER.info("imagem=%s n_gt=%d n_pred=%d", img_path.name, len(gts), len(preds))

    df_pe_fp = pd.DataFrame(pe_fp_rows)
    df_ps = pd.DataFrame(ps_rows)

    reports_dir = EXP_DIR / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    pe_fp_csv = reports_dir / "pe_false_positives.csv"
    ps_csv = reports_dir / "ps_false_negatives_vs_matched.csv"
    df_pe_fp.to_csv(pe_fp_csv, index=False)
    df_ps.to_csv(ps_csv, index=False)

    n_pe_fp = len(df_pe_fp)
    frac_pe_fp_overlap_other = float((df_pe_fp["best_iou_other_class_gt"] > 0).mean()) if n_pe_fp else float("nan")
    print(f"\n=== PE falsos positivos (n={n_pe_fp}) ===")
    print(f"fracao que sobrepoe (IoU>0) GT de outra classe: {frac_pe_fp_overlap_other:.3f}")
    if n_pe_fp:
        print("classe mais frequente sobreposta:")
        print(df_pe_fp.loc[df_pe_fp['best_iou_other_class_gt'] > 0, 'overlapping_class'].value_counts())

    n_ps_fn = int((df_ps["status"] == "FN").sum())
    n_ps_tp = int((df_ps["status"] == "TP").sum())
    frac_ps_fn_no_proposal = float((df_ps.loc[df_ps["status"] == "FN", "any_pred_overlap"] == False).mean()) if n_ps_fn else float("nan")
    print(f"\n=== PS falsos negativos (n={n_ps_fn}) vs. verdadeiros positivos (n={n_ps_tp}) ===")
    print(f"fracao de FN sem NENHUMA proposta sobrepondo (qualquer classe): {frac_ps_fn_no_proposal:.3f}")
    print(f"area_norm mediana -- FN: {df_ps.loc[df_ps['status']=='FN','box_area_norm'].median():.6f}  "
          f"TP: {df_ps.loc[df_ps['status']=='TP','box_area_norm'].median():.6f}")

    ps_fn_records = list(df_ps.loc[df_ps["status"] == "FN", ["source_image", "box_area_norm"]].itertuples(index=False, name=None))
    ps_tp_records = list(df_ps.loc[df_ps["status"] == "TP", ["source_image", "box_area_norm"]].itertuples(index=False, name=None))
    bootstrap_ps_area = bootstrap_median_diff_by_image(ps_tp_records, ps_fn_records) if (ps_fn_records and ps_tp_records) else None
    if bootstrap_ps_area:
        print(f"\nbootstrap (exploratorio) diff mediana area_norm FN-TP: {bootstrap_ps_area['diff_median_point']:+.6f} "
              f"IC95%={bootstrap_ps_area['ci95']} exclui_zero={'sim' if bootstrap_ps_area['excludes_zero'] else 'nao'}")

    # --- galeria qualitativa ---
    pe_fp_gallery_candidates.sort(key=lambda x: x[0], reverse=True)
    ps_fn_gallery_candidates.sort(key=lambda x: x[0], reverse=True)
    pe_top = pe_fp_gallery_candidates[:N_GALLERY]
    ps_top = ps_fn_gallery_candidates[:N_GALLERY]

    fig, axes = plt.subplots(2, N_GALLERY, figsize=(2.2 * N_GALLERY, 4.6))
    for col in range(N_GALLERY):
        ax = axes[0, col]
        if col < len(pe_top):
            conf, img_path, xyxy, W, H = pe_top[col]
            image = Image.open(img_path).convert("RGB")
            crop = crop_with_margin(image, xyxy, W, H)
            ax.imshow(np.asarray(crop))
            ax.set_title(f"conf={conf:.2f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("PE falso positivo\n(maior confiança)", fontsize=9)

        ax = axes[1, col]
        if col < len(ps_top):
            area, img_path, xyxy, W, H = ps_top[col]
            image = Image.open(img_path).convert("RGB")
            crop = crop_with_margin(image, xyxy, W, H)
            ax.imshow(np.asarray(crop))
            ax.set_title(f"area={area:.4f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        if col == 0:
            ax.set_ylabel("PS falso negativo\n(maior área)", fontsize=9)

    fig.suptitle("Galeria de inspeção -- PE falso positivo / PS falso negativo (exploratório)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    gallery_path = reports_dir / "pe_fp_ps_fn_gallery.png"
    fig.savefig(gallery_path, bbox_inches="tight")
    plt.close(fig)

    save_experiment_summary(EXP_NAME, RUNTIME_METADATA, {
        "note": "analise exploratoria -- gera hipotese, nao e resultado confirmatorio do artigo",
        "model_path": str(MODEL_PATH),
        "n_pe_false_positives": n_pe_fp,
        "frac_pe_fp_overlapping_other_class_gt": frac_pe_fp_overlap_other,
        "n_ps_false_negatives": n_ps_fn,
        "n_ps_true_positives": n_ps_tp,
        "frac_ps_fn_with_no_prediction_overlap": frac_ps_fn_no_proposal,
        "ps_area_norm_median_fn": float(df_ps.loc[df_ps["status"] == "FN", "box_area_norm"].median()) if n_ps_fn else None,
        "ps_area_norm_median_tp": float(df_ps.loc[df_ps["status"] == "TP", "box_area_norm"].median()) if n_ps_tp else None,
        "bootstrap_ps_area_norm_median_diff_fn_vs_tp": bootstrap_ps_area,
        "iou_threshold": IOU_THRESHOLD,
        "conf_threshold": CONF_THRESHOLD,
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
    })
    print(f"\nCSVs: {pe_fp_csv}, {ps_csv}")
    print(f"Galeria: {gallery_path}")
    print(f"Log completo: {LOG_PATH}")


if __name__ == "__main__":
    main()
