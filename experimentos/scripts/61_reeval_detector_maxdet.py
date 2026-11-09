"""Reavalia o detector no teste travado com um limite explicito de deteccoes
por imagem (`max_det`), para o single run e para as 10 repeticoes controladas.

Motivo: os Metodos do artigo afirmam "at most 2,000 detections per image", e
`max_det=2000` esta de fato nos 5 scripts de TREINO (13, 44, 48, 50 e
01_yolo26_main_10x) e fica gravado no checkpoint. Mas nenhum script de
AVALIACAO passa o parametro, e o predictor do Ultralytics sobrescreve o
atributo da cabeca a cada chamada:

    ultralytics/engine/predictor.py:426
    model.set_head_attr(max_det=max(self.args.max_det, 300), ...)

Sem o parametro, `args.max_det` vale o padrao 300 e o 2000 do checkpoint e
descartado. Duas micrografias do teste travado pararam exatamente em 300
predicoes -- PP_PP-1000x1 (1.268 referencias) e PS_PS-100x (957) --, que sao
as duas mais densas e somam 2.225 das 4.610 particulas do teste.

Reproduz a logica de 14_yolo_test_eval.py (val() + casamento guloso por IoU)
e de 23_recall_by_size_regime.py (regime fisico nano/micro), variando so o
max_det. Tambem recalcula a correlacao densidade x recall, que hoje nao
existe em script nenhum.

NAO cobre: a cadeia fim-a-fim (D.P.S) nem as linhas pareadas da Tabela 1 --
essas precisam do SmolVLM e dependem de qual adaptador de tamanho sera
adotado (Eq. 1 ou Eq. 2), decisao ainda aberta. Ver eval_and_pair.py.

SANIDADE: com --max-det 300 o single run deve reproduzir os valores
publicados (P=0.6069 R=0.4668 mAP50=0.2802; recall global 0.3358; taxa D
0.3544). O script checa e aborta se divergir.

Uso:
    python3 61_reeval_detector_maxdet.py                  # sanidade 300 + 2000
    python3 61_reeval_detector_maxdet.py --max-det 2000   # so 2000
    python3 61_reeval_detector_maxdet.py --skip-seeds     # so o single run
"""
import argparse
import math
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import save_experiment_summary, setup_experiment_logging

from ultralytics import YOLO

EXP_NAME = "61_reeval_detector_maxdet"
EXP_DIR, LOG_PATH, LOGGER, RUNTIME_METADATA = setup_experiment_logging(EXP_NAME, __file__)

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "datasets_yolo26_v2"
SINGLE_RUN = ROOT / "runs_v2" / "microplastic_yolo26_v2" / "weights" / "best.pt"
CTRL_RUNS = ROOT.parent / "experimentos_controlados" / "runs" / "01_yolo26_main"
CLASS_NAMES = ["PE", "PET", "PP", "PS"]

IMG_SIZE = 1024
CONF_THRESHOLD = 0.15
VAL_IOU = 0.7          # NMS do val(), como em 14_yolo_test_eval.py
MATCH_IOU = 0.5        # casamento predicao x referencia
NANO_MAX_NM = 1000.0
SEEDS = range(10)

PUBLISHED_300 = {"precision": 0.60694, "recall": 0.46683, "map50": 0.28025,
                 "global_recall": 0.33579, "d_rate": 0.35445}

MAG_X = re.compile(r"(\d+)\s*[xX]")
MAG_DASH = re.compile(r"^\D*(\d+)-\d+")


def parse_magnification(name: str):
    m = MAG_X.findall(name)
    if m:
        return int(m[-1])
    m = MAG_DASH.match(name)
    return int(m.group(1)) if m else None


def test_yaml() -> Path:
    """data.yaml com caminho absoluto correto (o do repo aponta para o path antigo)."""
    p = Path(tempfile.mkdtemp()) / "data_test.yaml"
    p.write_text(
        f"path: {TEST_DIR}\ntrain: images/test\nval: images/test\n"
        f"nc: {len(CLASS_NAMES)}\nnames:\n" + "\n".join(f"  - {c}" for c in CLASS_NAMES) + "\n"
    )
    return p


def iou_xyxy(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(label_path: Path, mag):
    """Referencias com classe, caixa normalizada e regime fisico."""
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cls_id = int(p[0])
        cx, cy, w, h = (float(v) for v in p[1:5])
        box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        regime = None
        if mag:
            size_nm = max(w * 1280, h * 960) * (100.0 / mag) * 1000.0
            regime = "NANOPLASTIC" if size_nm < NANO_MAX_NM else "MICROPLASTIC"
        out.append({"cls_id": cls_id, "box": box, "regime": regime})
    return out


def greedy_match(gts, preds, class_aware: bool):
    cands = []
    for gi, g in enumerate(gts):
        for pi, p in enumerate(preds):
            if class_aware and p["cls_id"] != g["cls_id"]:
                continue
            iou = iou_xyxy(g["box"], p["box"])
            if iou >= MATCH_IOU:
                cands.append((iou, gi, pi))
    cands.sort(reverse=True)
    mg, mp = set(), set()
    for _, gi, pi in cands:
        if gi in mg or pi in mp:
            continue
        mg.add(gi)
        mp.add(pi)
    return mg


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    r = sxy / math.sqrt(sxx * syy)
    t = r * math.sqrt((n - 2) / (1 - r * r))
    return r, t, n


def evaluate(ckpt: Path, max_det: int, yaml_path: Path):
    """val() + casamento por particula, com max_det explicito."""
    model = YOLO(str(ckpt))
    m = model.val(data=str(yaml_path), split="val", imgsz=IMG_SIZE,
                  conf=CONF_THRESHOLD, iou=VAL_IOU, max_det=max_det,
                  plots=False, verbose=False)
    agg = {"precision": float(m.box.mp), "recall": float(m.box.mr),
           "map50": float(m.box.map50), "map50_95": float(m.box.map)}
    per_class = {}
    for i, name in enumerate(CLASS_NAMES):
        try:
            p, r, ap50, ap = m.box.class_result(i)
            per_class[name] = {"precision": float(p), "recall": float(r),
                               "ap50": float(ap50), "ap50_95": float(ap)}
        except Exception:
            pass

    img_dir, lbl_dir = TEST_DIR / "images" / "test", TEST_DIR / "labels" / "test"
    per_image, regime = [], {"NANOPLASTIC": [0, 0], "MICROPLASTIC": [0, 0]}
    n_aware = n_spatial = n_gt = 0
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in (".png", ".jpg", ".jpeg", ".tif"):
            continue
        mag = parse_magnification(img.name)
        gts = load_gt(lbl_dir / f"{img.stem}.txt", mag)
        if not gts:
            continue
        pred = model.predict(source=str(img), imgsz=IMG_SIZE, conf=CONF_THRESHOLD,
                             max_det=max_det, verbose=False)[0]
        preds = []
        if pred.boxes is not None and len(pred.boxes):
            for box, c in zip(pred.boxes.xyxyn.cpu().numpy(),
                              pred.boxes.cls.cpu().numpy().astype(int)):
                preds.append({"cls_id": int(c), "box": tuple(box)})
        ma = greedy_match(gts, preds, class_aware=True)
        ms = greedy_match(gts, preds, class_aware=False)
        for gi, g in enumerate(gts):
            if g["regime"]:
                regime[g["regime"]][1] += 1
                if gi in ms:
                    regime[g["regime"]][0] += 1
        n_aware += len(ma); n_spatial += len(ms); n_gt += len(gts)
        per_image.append({"image": img.name, "n_gt": len(gts), "n_pred": len(preds),
                          "matched_class_aware": len(ma), "matched_spatial": len(ms),
                          "truncated": len(preds) >= max_det})

    gt_counts = [p["n_gt"] for p in per_image]
    recalls = [p["matched_spatial"] / p["n_gt"] for p in per_image]
    r, t, n = pearson(gt_counts, recalls)
    return {"max_det": max_det, "aggregate_metrics": agg, "per_class_metrics": per_class,
            "n_gt": n_gt, "global_recall": n_aware / n_gt, "d_rate": n_spatial / n_gt,
            "regime_recall": {k: {"gt": v[1], "matched": v[0],
                                  "recall": v[0] / v[1] if v[1] else None}
                              for k, v in regime.items()},
            "density_correlation": {"pearson_r": r, "t": t, "n_images": n},
            "n_truncated_images": sum(p["truncated"] for p in per_image),
            "per_image": per_image}


def summarize(label, res):
    a = res["aggregate_metrics"]
    print(f"\n[{label}] max_det={res['max_det']}")
    print(f"   val():  P={a['precision']:.4f} R={a['recall']:.4f} "
          f"mAP50={a['map50']:.4f} mAP50-95={a['map50_95']:.4f}")
    print(f"   recall global={res['global_recall']:.4f}  taxa D={res['d_rate']:.4f}  "
          f"truncadas={res['n_truncated_images']}")
    rg = res["regime_recall"]
    print(f"   nano={rg['NANOPLASTIC']['recall']*100:.2f}% ({rg['NANOPLASTIC']['gt']})  "
          f"micro={rg['MICROPLASTIC']['recall']*100:.2f}% ({rg['MICROPLASTIC']['gt']})")
    d = res["density_correlation"]
    print(f"   densidade x recall: r={d['pearson_r']:+.4f} t={d['t']:+.3f} n={d['n_images']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-det", type=int, default=2000)
    ap.add_argument("--skip-sanity", action="store_true",
                    help="pula a checagem com max_det=300 (nao recomendado)")
    ap.add_argument("--skip-seeds", action="store_true",
                    help="avalia so o single run, sem as 10 repeticoes")
    args = ap.parse_args()

    yaml_path = test_yaml()
    out = {}

    if not args.skip_sanity:
        base = evaluate(SINGLE_RUN, 300, yaml_path)
        summarize("single run, SANIDADE", base)
        out["single_run_maxdet300"] = base
        drift = {k: abs(base.get(k, base["aggregate_metrics"].get(k, 0)) - v)
                 if k in ("global_recall", "d_rate")
                 else abs(base["aggregate_metrics"][k] - v)
                 for k, v in PUBLISHED_300.items()}
        ok = all(d < 1e-3 for d in drift.values())
        print(f"\n   reproduz o publicado? {'SIM' if ok else 'NAO'}")
        for k, v in PUBLISHED_300.items():
            print(f"      {k:14s} publicado={v:.5f}  dif={drift[k]:.5f}")
        if not ok:
            print("\n   !! divergencia -- abortando, o resultado com max_det novo nao valeria.")
            return

    single = evaluate(SINGLE_RUN, args.max_det, yaml_path)
    summarize("single run", single)
    out[f"single_run_maxdet{args.max_det}"] = single

    seeds_res = {}
    if not args.skip_seeds:
        for s in SEEDS:
            ck = CTRL_RUNS / f"rep_{s:02d}" / "weights" / "best.pt"
            if not ck.exists():
                LOGGER.warning("checkpoint ausente: %s", ck)
                continue
            r = evaluate(ck, args.max_det, yaml_path)
            seeds_res[s] = r
            print(f"   seed {s}: P={r['aggregate_metrics']['precision']:.4f} "
                  f"R={r['aggregate_metrics']['recall']:.4f} "
                  f"mAP50={r['aggregate_metrics']['map50']:.4f} D={r['d_rate']:.4f}")
        if len(seeds_res) > 1:
            print(f"\n=== {len(seeds_res)} repeticoes, max_det={args.max_det} "
                  f"(formato da Tabela 1) ===")
            for key in ("precision", "recall", "map50", "map50_95"):
                v = np.array([r["aggregate_metrics"][key] for r in seeds_res.values()]) * 100
                h = 2.2621571627409915 * v.std(ddof=1) / math.sqrt(len(v))
                print(f"   {key:10s} {v.mean():6.2f} +/- {v.std(ddof=1):5.2f}  "
                      f"[{v.mean()-h:6.2f}, {v.mean()+h:6.2f}]")
        out[f"controlled_seeds_maxdet{args.max_det}"] = {str(k): v for k, v in seeds_res.items()}

    save_experiment_summary(EXP_NAME, RUNTIME_METADATA, {
        "note": "reavaliacao do detector com max_det explicito. Os scripts originais de "
                "avaliacao nao passam max_det e rodaram com o padrao 300 do Ultralytics, "
                "apesar de max_det=2000 estar nos scripts de treino e gravado no checkpoint "
                "(descartado por predictor.py:426). Nao cobre a cadeia fim-a-fim D.P.S.",
        "conf_threshold": CONF_THRESHOLD,
        "val_nms_iou": VAL_IOU,
        "match_iou": MATCH_IOU,
        "published_with_max_det_300": PUBLISHED_300,
        **out,
    })
    print(f"\nLog completo: {LOG_PATH}")


if __name__ == "__main__":
    main()
