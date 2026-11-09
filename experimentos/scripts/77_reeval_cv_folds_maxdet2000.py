"""Reavalia os 10 checkpoints de validacao cruzada (5 folds MISTO +
5 folds NANO-ONLY) com max_det explicito -- mesmo problema latente ja
corrigido no detector principal e na Secao de Custo Computacional: nenhum
dos scripts originais (45/51/58) passava max_det, entao valia o padrao 300
do Ultralytics, apesar dos Metodos (e do checkpoint de treino) declararem
2.000.

A cross-validation cobre os 102 micrografias usaveis (nao so as 15 do teste
travado), entao ha micrografias densas diferentes das ja conhecidas
(PP_PP-1000x1, PS_PS-100x) truncando em 300 aqui -- confirmado por sondagem
previa: 15 das ~102 imagens held-out (soma dos 5 folds mistos) e 2 das ~100
(soma dos 5 folds nano-only) batem o teto, algumas indo ate quase 3x mais
predicoes a max_det=2000.

Alimenta mainV11.tex secao Discussion / RQ1, paragrafo da ablacao nano-only
(achado C de uma sessao anterior): "15.0% pooled recall ... against 21.0%
for mixed-size training ... [-1.5, +14.5] ... 20.8% to 4.8% for PET ...
21.7% to 3.6% for PS ... 20.3% to 25.2%".

SANIDADE (max_det=300 deve reproduzir):
  nano-only : overall=6.19%  NANO=15.01% (n=9399)  MICRO=1.11% (n=16331)
  misto     : overall=23.78% NANO=21.03% (n=9399)  MICRO=25.36% (n=16331)

Grava em resultados/77_reeval_cv_folds_maxdet2000/, nao sobrescreve
51_eval_cv_folds_nano nem 58_eval_cv_folds_mixed_by_regime.

Uso:
    python3 -u 77_reeval_cv_folds_maxdet2000.py
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import save_experiment_summary, setup_experiment_logging

from ultralytics import YOLO

EXP_NAME = "77_reeval_cv_folds_maxdet2000"
EXP_DIR, LOG_PATH, LOGGER, RUNTIME_METADATA = setup_experiment_logging(EXP_NAME, __file__)

ROOT = Path(__file__).resolve().parent.parent
K_FOLDS = 5
IMG_SIZE = 1024
CONF_THRESHOLD = 0.15
IOU_THRESHOLD = 0.5
N_BOOTSTRAP = 10_000
SEED = 42
NANO_MAX_NM = 1000.0
MAX_DETS = (300, 2000)

CONDITIONS = {
    "nano_only": {
        "runs_dir": ROOT / "runs_v2_cv_nano",
        "data_dir": ROOT / "datasets_yolo26_v2_cv_nano",
        "published": {"overall": 0.061912164788184995,
                      "NANOPLASTIC": (0.1501223534418555, 9399),
                      "MICROPLASTIC": (0.011144449207029576, 16331)},
    },
    "mixed": {
        "runs_dir": ROOT / "runs_v2_cv",
        "data_dir": ROOT / "datasets_yolo26_v2_cv",
        "published": {"overall": 0.23777691410804508,
                      "NANOPLASTIC": (0.2103415256942228, 9399),
                      "MICROPLASTIC": (0.25356683607862346, 16331)},
    },
}
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


def yolo_to_xyxy(cx, cy, w, h, W, H):
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


def load_gt(label_path: Path, W: int, H: int, um_per_px):
    out = []
    if not label_path.exists():
        return out
    for line in open(label_path):
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        px_size = max(w * W, h * H)
        size_nm = px_size * um_per_px * 1000.0
        regime = "NANOPLASTIC" if size_nm < NANO_MAX_NM else "MICROPLASTIC"
        out.append({"cls_id": cls_id, "xyxy": yolo_to_xyxy(cx, cy, w, h, W, H), "size": regime})
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
    matched_gt, matched_pred = set(), set()
    for iou, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue
        matched_gt.add(gi)
        matched_pred.add(pi)
    return matched_gt


def bootstrap_recall_ci(records, size_filter, seed=SEED, n_bootstrap=N_BOOTSTRAP):
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
    return {"n": len(sub), "n_images": len(imgs), "recall_point": float(df["matched"].mean()),
            "ci95": [float(ci[0]), float(ci[1])]}


def run_condition(cond_key, max_det):
    cfg = CONDITIONS[cond_key]
    runs_dir, data_dir = cfg["runs_dir"], cfg["data_dir"]

    all_records = []
    per_fold = []
    for k in range(K_FOLDS):
        ckpt = runs_dir / f"fold{k}" / "weights" / "best.pt"
        img_dir = data_dir / f"fold{k}" / "images" / "test"
        label_dir = data_dir / f"fold{k}" / "labels" / "test"
        if not ckpt.exists():
            raise SystemExit(f"checkpoint nao existe: {ckpt}")
        model = YOLO(str(ckpt))

        img_paths = sorted(img_dir.glob("*.png"))
        n_gt_fold = n_matched_fold = 0
        for img_path in img_paths:
            mag = parse_magnification(img_path.stem)
            if mag is None:
                continue
            um_per_px = 100.0 / mag
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                W, H = im.size
            gts = load_gt(label_dir / f"{img_path.stem}.txt", W, H, um_per_px)

            pred = model.predict(source=str(img_path), imgsz=IMG_SIZE, conf=CONF_THRESHOLD,
                                 max_det=max_det, verbose=False)[0]
            preds = []
            if pred.boxes is not None and len(pred.boxes):
                xyxy_px = pred.boxes.xyxy.cpu().numpy()
                cls_ids = pred.boxes.cls.cpu().numpy().astype(int)
                for box, c in zip(xyxy_px, cls_ids):
                    preds.append({"cls_id": int(c), "xyxy": tuple(box)})

            matched_gt = match_same_class(gts, preds)
            for gi, g in enumerate(gts):
                is_matched = gi in matched_gt
                all_records.append({"source_image": img_path.name, "fold": k,
                                    "size": g["size"], "matched": is_matched})
                n_gt_fold += 1
                n_matched_fold += int(is_matched)
            LOGGER.info("cond=%s max_det=%d fold=%d imagem=%s gt=%d matched=%d n_pred=%d",
                        cond_key, max_det, k, img_path.name, len(gts), len(matched_gt), len(preds))

        per_fold.append({"fold": k, "n_images": len(img_paths), "n_gt": n_gt_fold,
                         "recall": n_matched_fold / n_gt_fold if n_gt_fold else None})

    bucket = {}
    for regime in ("NANOPLASTIC", "MICROPLASTIC"):
        bucket[regime] = bootstrap_recall_ci(all_records, lambda s, r=regime: s == r)
    overall = bootstrap_recall_ci(all_records, lambda s: True)

    # salva os registros por particula -- reusado por 78_compare_nano_ablation_maxdet2000.py
    # (mesmo formato de 51/58: source_image, size, matched)
    out_csv = EXP_DIR / "reports" / f"cv_pooled_predictions_{cond_key}_maxdet{max_det}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_records).to_csv(out_csv, index=False)

    return {"max_det": max_det, "per_fold": per_fold, "bucket_results": bucket, "overall": overall,
            "csv_path": str(out_csv)}


def main():
    results = {}
    for cond_key in CONDITIONS:
        results[cond_key] = {}
        for md in MAX_DETS:
            print(f"\n{'='*72}\n{cond_key} -- max_det={md}\n{'='*72}")
            LOGGER.info("iniciando %s max_det=%d", cond_key, md)
            r = run_condition(cond_key, md)
            results[cond_key][md] = r
            print(f"  overall recall={r['overall']['recall_point']*100:.2f}%  "
                  f"NANO={r['bucket_results']['NANOPLASTIC']['recall_point']*100:.2f}%  "
                  f"MICRO={r['bucket_results']['MICROPLASTIC']['recall_point']*100:.2f}%")

    print(f"\n{'='*72}\nSANIDADE -- max_det=300 reproduz 51 e 58?\n{'='*72}")
    ok = True
    for cond_key, cfg in CONDITIONS.items():
        pub = cfg["published"]
        r300 = results[cond_key][300]
        d_overall = abs(r300["overall"]["recall_point"] - pub["overall"])
        flag = "" if d_overall <= SANITY_TOL else "  <-- DIVERGE"
        if d_overall > SANITY_TOL:
            ok = False
        print(f"  {cond_key:10s} overall  publicado={pub['overall']*100:.2f}%  "
              f"obtido={r300['overall']['recall_point']*100:.2f}%  dif={d_overall*100:.3f}pp{flag}")
        for regime in ("NANOPLASTIC", "MICROPLASTIC"):
            pub_r, pub_n = pub[regime]
            obt = r300["bucket_results"][regime]
            d = abs(obt["recall_point"] - pub_r)
            n_ok = obt["n"] == pub_n
            flag = "" if (d <= SANITY_TOL and n_ok) else "  <-- DIVERGE"
            if d > SANITY_TOL or not n_ok:
                ok = False
            print(f"  {cond_key:10s} {regime:13s} publicado={pub_r*100:.2f}% (n={pub_n})  "
                  f"obtido={obt['recall_point']*100:.2f}% (n={obt['n']})  dif={d*100:.3f}pp{flag}")
    print("\n  ->", "SIM" if ok else "NAO")
    if not ok:
        print("\n  !! nao reproduz -- o resultado de max_det=2000 NAO deve ser usado.")
        return

    print(f"\n{'='*72}\nmax_det=300 (sanidade)  vs  max_det=2000\n{'='*72}")
    for cond_key in CONDITIONS:
        r300, r2000 = results[cond_key][300], results[cond_key][2000]
        print(f"\n  {cond_key}")
        print(f"    overall   {r300['overall']['recall_point']*100:6.2f}%  ->  {r2000['overall']['recall_point']*100:6.2f}%")
        for regime in ("NANOPLASTIC", "MICROPLASTIC"):
            a, b = r300["bucket_results"][regime], r2000["bucket_results"][regime]
            print(f"    {regime:13s} {a['recall_point']*100:6.2f}% [{a['ci95'][0]*100:.1f},{a['ci95'][1]*100:.1f}]  ->  "
                  f"{b['recall_point']*100:6.2f}% [{b['ci95'][0]*100:.1f},{b['ci95'][1]*100:.1f}]")

    # diferenca pareada entre misto e nano-only a max_det=2000, por micrografia-fonte
    # (bootstrap sobre imagens, igual 59_compare_nano_ablation_by_label.py)
    nano_2000 = results["nano_only"][2000]["bucket_results"]["NANOPLASTIC"]
    mixed_2000 = results["mixed"][2000]["bucket_results"]["NANOPLASTIC"]
    print(f"\n  diferenca NANO (misto - nano_only) a max_det=2000: "
          f"{(mixed_2000['recall_point']-nano_2000['recall_point'])*100:+.2f} pp")

    save_experiment_summary(EXP_NAME, RUNTIME_METADATA, {
        "note": "reavaliacao dos 10 folds de CV (5 mistos + 5 nano-only) com max_det "
                "explicito -- alimenta achado C (ablacao nano-only) na Discussao. "
                "Varias micrografias held-out truncavam em 300 predicoes.",
        "conf_threshold": CONF_THRESHOLD, "iou_threshold": IOU_THRESHOLD,
        "sanity_reproduces_published": ok,
        "results": {cond_key: {str(md): r for md, r in by_md.items()} for cond_key, by_md in results.items()},
    })
    print(f"\nLog completo: {LOG_PATH}")


if __name__ == "__main__":
    main()
