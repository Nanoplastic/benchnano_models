"""Recalcula os intervalos de confianca da Tabela A3 (detector, teste travado)
com max_det explicito, lado a lado com max_det=300 (o que de fato rodou).

Os intervalos publicados vem de dois scripts:
    14_yolo_test_eval.py   -> recall casado: bootstrap por imagem sobre as
                              contagens (casados, referencias), 10.000 replicas
    25_bootstrap_yolo_detection_metrics.py
                           -> precisao, mAP@0.5, mAP@0.5:0.95: UMA passada de
                              model.val() por replica sobre a lista de imagens
                              reamostrada, 2.000 replicas
Ambos com seed 42 e unidade de reamostragem = micrografia. Nenhum dos dois
passava max_det, entao valia o padrao 300 do Ultralytics.

Este script NAO altera nenhum resultado existente:
  - grava so em resultados/64_bootstrap_ci_maxdet/
  - nao le nem escreve em datasets_yolo26_v2/ alem de leitura: as listas de
    reamostragem e o cache do Ultralytics ficam num diretorio temporario
    proprio (symlinks para as imagens e rotulos reais), removido ao final
  - 14_yolo_test_eval, 25_bootstrap_yolo_detection_metrics e 61_reeval_* ficam
    como estao

As duas condicoes (300 e 2000) rodam sobre as MESMAS reamostragens, replica a
replica, entao os intervalos sao diretamente comparaveis. O braco de 300 tambem
serve de SANIDADE: deve reproduzir os intervalos publicados.

Gates (o script para se falharem):
  1. estimativas pontuais com max_det=300 batem 0.6069/0.4668/0.2802/0.0861
  2. recall com max_det=300 reproduz o IC publicado [0.2473, 0.5182]
  3. ao final, os ICs de precisao/mAP a 300 batem os de 25 (tolerancia 0.005)

Uso:
    python3 -u 64_bootstrap_ci_maxdet.py            # completo, ~45 min de GPU
    python3 -u 64_bootstrap_ci_maxdet.py --so-recall   # so a parte de CPU
    python3 -u 64_bootstrap_ci_maxdet.py --teste 20    # 20 replicas, nao grava
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import save_experiment_summary, setup_experiment_logging

from ultralytics import YOLO

EXP_NAME = "64_bootstrap_ci_maxdet"
EXP_DIR, LOG_PATH, LOGGER, RUNTIME_METADATA = setup_experiment_logging(EXP_NAME, __file__)

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "datasets_yolo26_v2"
MODEL_PATH = ROOT / "runs_v2" / "microplastic_yolo26_v2" / "weights" / "best.pt"
RESULTS = ROOT / "resultados"
SUMMARY_61 = RESULTS / "61_reeval_detector_maxdet" / "summary.json"
SUMMARY_14 = RESULTS / "14_yolo_test_eval" / "summary.json"
SUMMARY_25 = RESULTS / "25_bootstrap_yolo_detection_metrics" / "summary.json"
CLASS_NAMES = ["PE", "PET", "PP", "PS"]

IMG_SIZE = 1024
CONF_THRESHOLD = 0.15
SEED = 42
N_BOOT_RECALL = 10_000
N_BOOT_VAL = 2_000
MAX_DETS = (300, 2000)
SANITY_POINT_TOL = 1e-3
SANITY_CI_TOL = 0.005

POINT_300 = {"precision": 0.6069422606876318, "recall": 0.46682537100157057,
             "map50": 0.28024896067965005, "map50_95": 0.08606379406840536}
POINT_2000 = {"precision": 0.5675057080896679, "recall": 0.5121347565230392,
              "map50": 0.31568136979134026, "map50_95": 0.09535511046969802}
METRICS = ("precision", "map50", "map50_95")


# ---------------------------------------------------------------- recall (CPU)

def recall_ci(per_image, n_boot=N_BOOT_RECALL, seed=SEED):
    """Mesma logica de 14_yolo_test_eval.py: bootstrap por imagem das contagens."""
    rng = np.random.default_rng(seed)
    n = len(per_image)
    matched = np.array([p["matched_class_aware"] for p in per_image])
    gt = np.array([p["n_gt"] for p in per_image])
    rec = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        g = gt[idx].sum()
        rec[b] = matched[idx].sum() / g if g else np.nan
    lo, hi = np.nanpercentile(rec, [2.5, 97.5])
    return matched.sum() / gt.sum(), float(lo), float(hi)


def parte_recall():
    m = json.loads(SUMMARY_61.read_text())["metrics"]
    pub = json.loads(SUMMARY_14.read_text())["metrics"]["global_recall_bootstrap_ci95"]
    out = {}
    for md, key in ((300, "single_run_maxdet300"), (2000, "single_run_maxdet2000")):
        per_image = sorted(m[key]["per_image"], key=lambda p: p["image"])
        per_image = [p for p in per_image if p["n_gt"] > 0]
        pt, lo, hi = recall_ci(per_image)
        out[md] = {"point": pt, "low": lo, "high": hi, "n_images": len(per_image)}

    r = out[300]
    ok = (abs(r["point"] - pub["point"]) < 1e-4 and abs(r["low"] - pub["low"]) < 1e-4
          and abs(r["high"] - pub["high"]) < 1e-4)
    print("\n" + "=" * 72)
    print("RECALL CASADO (casamento por IoU>=0.5, mesma classe) -- 10.000 replicas")
    print("=" * 72)
    print(f"SANIDADE -- max_det=300 reproduz o publicado? {'SIM' if ok else 'NAO'}")
    print(f"   publicado  {pub['point']*100:5.2f}%  [{pub['low']*100:5.2f}, {pub['high']*100:5.2f}]")
    print(f"   obtido     {r['point']*100:5.2f}%  [{r['low']*100:5.2f}, {r['high']*100:5.2f}]")
    if not ok:
        print("\n   !! nao reproduz -- o intervalo de max_det=2000 NAO deve ser usado.")
        return None
    r2 = out[2000]
    print(f"   max_det=2000  {r2['point']*100:5.2f}%  [{r2['low']*100:5.2f}, {r2['high']*100:5.2f}]")
    return out


# ------------------------------------------------------------ precisao/mAP (GPU)

def monta_scratch(base: Path):
    """Espelho do conjunto de teste com symlinks, para o cache e as listas de
    reamostragem nunca tocarem datasets_yolo26_v2/."""
    img_dir, lbl_dir = base / "images" / "test", base / "labels" / "test"
    img_dir.mkdir(parents=True)
    lbl_dir.mkdir(parents=True)
    imgs = sorted((TEST_DIR / "images" / "test").glob("*.png"))
    for p in imgs:
        (img_dir / p.name).symlink_to(p.resolve())
        lab = TEST_DIR / "labels" / "test" / f"{p.stem}.txt"
        if lab.exists():
            (lbl_dir / lab.name).symlink_to(lab.resolve())
    # caminhos SEM resolve(): o Ultralytics deriva o rotulo trocando /images/ por /labels/
    return [img_dir / p.name for p in imgs]


def val(model, yaml_path, max_det):
    r = model.val(data=str(yaml_path), split="val", imgsz=IMG_SIZE, conf=CONF_THRESHOLD,
                  iou=0.7, max_det=max_det, plots=False, verbose=False)
    return {"precision": float(r.box.mp), "recall": float(r.box.mr),
            "map50": float(r.box.map50), "map50_95": float(r.box.map)}


def parte_val(n_boot):
    scratch = EXP_DIR / "_scratch_bootstrap"
    shutil.rmtree(scratch, ignore_errors=True)
    try:
        imgs = monta_scratch(scratch)
        n = len(imgs)
        list_path = scratch / "resample.txt"
        yaml_path = scratch / "resample.yaml"
        full_path = scratch / "full.txt"
        cache = scratch / "labels" / "test.cache"
        model = YOLO(str(MODEL_PATH))

        def escreve_yaml(lst):
            yaml_path.write_text(
                f"train: {lst}\nval: {lst}\nnc: {len(CLASS_NAMES)}\nnames:\n"
                + "\n".join(f"  - {c}" for c in CLASS_NAMES) + "\n")

        # gate 1: estimativas pontuais (lista completa, sem reamostrar)
        full_path.write_text("\n".join(str(p) for p in imgs))
        escreve_yaml(full_path)
        print("\n" + "=" * 72)
        print("PRECISAO / mAP -- gate 1: estimativas pontuais")
        print("=" * 72)
        pts = {}
        for md, ref in ((300, POINT_300), (2000, POINT_2000)):
            if cache.exists():
                cache.unlink()
            pts[md] = val(model, yaml_path, md)
            drift = max(abs(pts[md][k] - ref[k]) for k in ref)
            print(f"   max_det={md:4d}  P={pts[md]['precision']:.4f} R={pts[md]['recall']:.4f} "
                  f"mAP50={pts[md]['map50']:.4f} mAP50-95={pts[md]['map50_95']:.4f}  "
                  f"desvio max vs referencia={drift:.5f}")
            if drift > SANITY_POINT_TOL:
                print("\n   !! estimativa pontual nao reproduz -- abortando antes das replicas.")
                return None

        # replicas: mesmas reamostragens (mesma seed e mesma ordem) para os dois bracos
        escreve_yaml(list_path)
        rng = np.random.default_rng(SEED)
        arr = {md: {k: np.empty(n_boot) for k in METRICS} for md in MAX_DETS}
        print(f"\nreplicas: {n_boot} x {len(MAX_DETS)} bracos")
        t0 = time.time()
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            list_path.write_text("\n".join(str(imgs[i]) for i in idx))
            for md in MAX_DETS:
                if cache.exists():
                    cache.unlink()
                m = val(model, yaml_path, md)
                for k in METRICS:
                    arr[md][k][b] = m[k]
            if (b + 1) % 50 == 0:
                el = time.time() - t0
                LOGGER.info("replica %d/%d (%.0fs, %.2fs/replica, ~%.0f min restantes)",
                            b + 1, n_boot, el, el / (b + 1), el / (b + 1) * (n_boot - b - 1) / 60)
        return {"point": pts, "arrays": arr}
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def resume_val(res, n_boot):
    pub = json.loads(SUMMARY_25.read_text())["metrics"]["bootstrap_ci95"]
    ci = {md: {} for md in MAX_DETS}
    for md in MAX_DETS:
        for k in METRICS:
            lo, hi = np.percentile(res["arrays"][md][k], [2.5, 97.5])
            ci[md][k] = {"mean": float(res["arrays"][md][k].mean()),
                         "low": float(lo), "high": float(hi)}

    print("\n" + "=" * 72)
    print(f"PRECISAO / mAP -- ICs bootstrap por imagem ({n_boot} replicas)")
    print("=" * 72)
    ok = True
    if n_boot == N_BOOT_VAL:
        print("SANIDADE -- max_det=300 reproduz os ICs publicados (script 25)?")
        for k in METRICS:
            dl = abs(ci[300][k]["low"] - pub[k]["ci95_low"])
            dh = abs(ci[300][k]["high"] - pub[k]["ci95_high"])
            flag = "" if max(dl, dh) <= SANITY_CI_TOL else "  <-- DIVERGE"
            if max(dl, dh) > SANITY_CI_TOL:
                ok = False
            print(f"   {k:9s} publicado [{pub[k]['ci95_low']*100:5.2f}, {pub[k]['ci95_high']*100:5.2f}]"
                  f"  obtido [{ci[300][k]['low']*100:5.2f}, {ci[300][k]['high']*100:5.2f}]{flag}")
        print("   ->", "SIM" if ok else "NAO")
    else:
        print(f"(modo teste, {n_boot} replicas: sanidade contra o publicado nao se aplica)")
    print()
    print(f"{'metrica':10s} {'max_det=300':>22s} {'max_det=2000':>22s}")
    for k in METRICS:
        a, b = ci[300][k], ci[2000][k]
        print(f"{k:10s} [{a['low']*100:5.1f}, {a['high']*100:5.1f}] "
              f"     [{b['low']*100:5.1f}, {b['high']*100:5.1f}]")
    return ci, ok


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so-recall", action="store_true")
    ap.add_argument("--teste", type=int, default=0, help="N replicas; nao grava summary")
    args = ap.parse_args()

    for p in (MODEL_PATH, SUMMARY_61, SUMMARY_14, SUMMARY_25):
        if not p.exists():
            raise SystemExit(f"nao existe: {p}")

    rec = parte_recall()
    if rec is None:
        return
    if args.so_recall:
        return

    n_boot = args.teste or N_BOOT_VAL
    res = parte_val(n_boot)
    if res is None:
        return
    ci, ok = resume_val(res, n_boot)

    if args.teste:
        print("\nmodo teste: nada gravado.")
        return
    if not ok:
        print("\n!! sanidade falhou -- resultado NAO gravado como valido.")
        return

    print("\n" + "=" * 72)
    print("PARA A LEGENDA DA TABELA A3 (max_det=2000)")
    print("=" * 72)
    c = ci[2000]
    r = rec[2000]
    print(f"precision [{c['precision']['low']*100:.1f}%, {c['precision']['high']*100:.1f}%], "
          f"mAP@0.5 [{c['map50']['low']*100:.1f}%, {c['map50']['high']*100:.1f}%], "
          f"mAP@0.5:0.95 [{c['map50_95']['low']*100:.1f}%, {c['map50_95']['high']*100:.1f}%]; "
          f"matched recall {r['point']*100:.1f}% [{r['low']*100:.1f}%, {r['high']*100:.1f}%]")

    save_experiment_summary(EXP_NAME, RUNTIME_METADATA, {
        "note": "ICs da Tabela A3 com max_det explicito. Os publicados (scripts 14 e 25) "
                "rodaram com o padrao 300 do Ultralytics. Mesmas reamostragens nos dois "
                "bracos; o braco de 300 reproduz o publicado (sanidade).",
        "model_path": str(MODEL_PATH),
        "seed": SEED, "n_bootstrap_val": N_BOOT_VAL, "n_bootstrap_recall": N_BOOT_RECALL,
        "conf_threshold": CONF_THRESHOLD,
        "sanity_reproduces_published": ok,
        "point_estimates": {str(k): v for k, v in res["point"].items()},
        "recall_ci": {str(k): v for k, v in rec.items()},
        "val_ci": {str(k): v for k, v in ci.items()},
        "replicas": {str(md): {k: v.tolist() for k, v in d.items()}
                     for md, d in res["arrays"].items()},
    })
    print(f"\nLog completo: {LOG_PATH}")


if __name__ == "__main__":
    main()
