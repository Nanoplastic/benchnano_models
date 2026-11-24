"""Intervalos de confiança do detector com max_det / Detector confidence intervals with max_det.

PT: Recalcula os intervalos de confiança das métricas do detector no conjunto de teste com
    max_det explícito, lado a lado com max_det=300 (o valor efetivamente usado nas
    avaliações originais).
    Os intervalos originais vêm de dois scripts:
    14_yolo_test_eval.py -> recall pareado: bootstrap por imagem sobre as contagens
    (pareados, referências), 10.000 réplicas
    25_bootstrap_yolo_detection_metrics.py -> precisão, mAP@0.5, mAP@0.5:0.95: uma passada
    de model.val() por réplica sobre a lista de imagens reamostrada, 2.000 réplicas
    Ambos usam semente 42 e micrografia como unidade de reamostragem, sem passar max_det
    (padrão 300 do Ultralytics).
    Não altera resultados existentes: grava só em resultados/64_bootstrap_ci_maxdet/;
    datasets_yolo26_v2/ é só lido (listas de reamostragem e cache do Ultralytics ficam num
    diretório temporário próprio, com symlinks para imagens e rótulos, removido ao final);
    14, 25 e 61 não são modificados.
    As duas condições (300 e 2000) usam as MESMAS reamostragens, réplica a réplica, então os
    intervalos são diretamente comparáveis. O braço de 300 também serve de checagem.
    Checagens (o script para se falharem):
    1. estimativas pontuais com max_det=300 iguais a 0,6069/0,4668/0,2802/0,0861;
    2. recall com max_det=300 reproduz o IC de referência [0,2473; 0,5182];
    3. ao final, os ICs de precisão/mAP com 300 coincidem com os de 25 (tolerância 0,005).
    Uso:
    python3 -u 64_bootstrap_ci_maxdet.py -- completo, ~45 min de GPU
    python3 -u 64_bootstrap_ci_maxdet.py --so-recall -- só a parte de CPU
    python3 -u 64_bootstrap_ci_maxdet.py --teste 20 -- 20 réplicas, não grava
EN: Recomputes the confidence intervals of the detector metrics on the test set with
    explicit max_det, side by side with max_det=300 (the value actually used in the original
    evaluations).
    The original intervals come from two scripts:
    14_yolo_test_eval.py -> matched recall: per-image bootstrap over the counts (matched,
    references), 10,000 replicates
    25_bootstrap_yolo_detection_metrics.py -> precision, mAP@0.5, mAP@0.5:0.95: one
    model.val() pass per replicate over the resampled image list, 2,000 replicates
    Both use seed 42 and the micrograph as resampling unit, without passing max_det
    (Ultralytics default 300).
    Does not alter existing results: writes only to resultados/64_bootstrap_ci_maxdet/;
    datasets_yolo26_v2/ is read only (resampling lists and the Ultralytics cache live in a
    dedicated temporary directory, with symlinks to the images and labels, removed at the
    end); 14, 25 and 61 are not modified.
    Both conditions (300 and 2000) use the SAME resamples, replicate by replicate, so the
    intervals are directly comparable. The 300 arm also works as a check.
    Checks (the script stops if they fail):
    1. point estimates with max_det=300 equal 0.6069/0.4668/0.2802/0.0861;
    2. recall with max_det=300 reproduces the reference CI [0.2473, 0.5182];
    3. at the end, the precision/mAP CIs at 300 match those of 25 (tolerance 0.005).
    Usage:
    python3 -u 64_bootstrap_ci_maxdet.py -- full, ~45 min of GPU
    python3 -u 64_bootstrap_ci_maxdet.py --so-recall -- CPU part only
    python3 -u 64_bootstrap_ci_maxdet.py --teste 20 -- 20 replicates, writes nothing
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import salvar_resumo_experimento, configurar_log_experimento

from ultralytics import YOLO

NOME_EXP = "64_bootstrap_ci_maxdet"
DIR_EXP, CAMINHO_LOG, REGISTRADOR, METADADOS_EXECUCAO = configurar_log_experimento(NOME_EXP, __file__)

RAIZ = Path(__file__).resolve().parent.parent
DIR_TESTE = RAIZ / "datasets_yolo26_v2"
CAMINHO_MODELO = RAIZ / "runs_v2" / "microplastic_yolo26_v2" / "weights" / "best.pt"
RESULTADOS_EXEC = RAIZ / "resultados"
RESUMO_61 = RESULTADOS_EXEC / "61_reeval_detector_maxdet" / "summary.json"
RESUMO_14 = RESULTADOS_EXEC / "14_yolo_test_eval" / "summary.json"
RESUMO_25 = RESULTADOS_EXEC / "25_bootstrap_yolo_detection_metrics" / "summary.json"
NOMES_CLASSES = ["PE", "PET", "PP", "PS"]

TAMANHO_IMG = 1024
LIMIAR_CONF = 0.15
SEMENTE = 42
N_BOOT_RECALL = 10_000
N_BOOT_VAL = 2_000
MAX_DETS_TESTADOS = (300, 2000)
TOL_CHECAGEM_PONTO = 1e-3
TOL_CHECAGEM_IC = 0.005

PONTO_300 = {"precision": 0.6069422606876318, "recall": 0.46682537100157057,
             "map50": 0.28024896067965005, "map50_95": 0.08606379406840536}
PONTO_2000 = {"precision": 0.5675057080896679, "recall": 0.5121347565230392,
              "map50": 0.31568136979134026, "map50_95": 0.09535511046969802}
METRICAS = ("precision", "map50", "map50_95")


# PT: --- recall (CPU) ---
# EN: --- recall (CPU) ---

def ic_recall(por_imagem, n_boot=N_BOOT_RECALL, semente=SEMENTE):
    """PT: Mesma lógica de 14_yolo_test_eval.py: bootstrap por imagem das contagens.
    EN: Same logic as 14_yolo_test_eval.py: per-image bootstrap of the counts.
    """
    gerador = np.random.default_rng(semente)
    n = len(por_imagem)
    pareados = np.array([p["matched_class_aware"] for p in por_imagem])
    gt = np.array([p["n_gt"] for p in por_imagem])
    rec = np.empty(n_boot)
    for b in range(n_boot):
        indice = gerador.integers(0, n, size=n)
        g = gt[indice].sum()
        rec[b] = pareados[indice].sum() / g if g else np.nan
    lo, hi = np.nanpercentile(rec, [2.5, 97.5])
    return pareados.sum() / gt.sum(), float(lo), float(hi)


def parte_recall():
    m = json.loads(RESUMO_61.read_text())["metrics"]
    pub = json.loads(RESUMO_14.read_text())["metrics"]["global_recall_bootstrap_ci95"]
    saida = {}
    for md, chave in ((300, "single_run_maxdet300"), (2000, "single_run_maxdet2000")):
        por_imagem = sorted(m[chave]["per_image"], key=lambda p: p["image"])
        por_imagem = [p for p in por_imagem if p["n_gt"] > 0]
        pt, lo, hi = ic_recall(por_imagem)
        saida[md] = {"point": pt, "low": lo, "high": hi, "n_images": len(por_imagem)}

    r = saida[300]
    ok = (abs(r["point"] - pub["point"]) < 1e-4 and abs(r["low"] - pub["low"]) < 1e-4
          and abs(r["high"] - pub["high"]) < 1e-4)
    print("\n" + "=" * 72)
    print("RECALL CASADO (casamento por IoU>=0.5, mesma classe) -- 10.000 replicas")
    print("=" * 72)
    print(f"SANIDADE -- max_det=300 reproduz a referência? {'SIM' if ok else 'NAO'}")
    print(f"   referência {pub['point']*100:5.2f}%  [{pub['low']*100:5.2f}, {pub['high']*100:5.2f}]")
    print(f"   obtido     {r['point']*100:5.2f}%  [{r['low']*100:5.2f}, {r['high']*100:5.2f}]")
    if not ok:
        print("\n   !! nao reproduz -- o intervalo de max_det=2000 NAO deve ser usado.")
        return None
    r2 = saida[2000]
    print(f"   max_det=2000  {r2['point']*100:5.2f}%  [{r2['low']*100:5.2f}, {r2['high']*100:5.2f}]")
    return saida


# PT: --- precisão/mAP (GPU) ---
# EN: --- precision/mAP (GPU) ---

def monta_scratch(base: Path):
    """PT: Espelho do conjunto de teste com symlinks, para que o cache e as listas de
        reamostragem nunca toquem datasets_yolo26_v2/.
    EN: Mirror of the test set with symlinks, so the cache and resampling lists never touch
        datasets_yolo26_v2/.
    """
    dir_img, dir_rot = base / "images" / "test", base / "labels" / "test"
    dir_img.mkdir(parents=True)
    dir_rot.mkdir(parents=True)
    imgs = sorted((DIR_TESTE / "images" / "test").glob("*.png"))
    for p in imgs:
        (dir_img / p.name).symlink_to(p.resolve())
        rot_lab = DIR_TESTE / "labels" / "test" / f"{p.stem}.txt"
        if rot_lab.exists():
            (dir_rot / rot_lab.name).symlink_to(rot_lab.resolve())
    # PT: caminhos SEM resolve(): o Ultralytics deriva o rótulo trocando /images/ por
    #     /labels/
    # EN: paths WITHOUT resolve(): Ultralytics derives the label by replacing /images/ with
    #     /labels/
    return [dir_img / p.name for p in imgs]


def val(modelo, caminho_yaml, max_det):
    r = modelo.val(data=str(caminho_yaml), split="val", imgsz=TAMANHO_IMG, conf=LIMIAR_CONF,
                  iou=0.7, max_det=max_det, plots=False, verbose=False)
    return {"precision": float(r.box.mp), "recall": float(r.box.mr),
            "map50": float(r.box.map50), "map50_95": float(r.box.map)}


def parte_val(n_boot):
    rascunho = DIR_EXP / "_scratch_bootstrap"
    shutil.rmtree(rascunho, ignore_errors=True)
    try:
        imgs = monta_scratch(rascunho)
        n = len(imgs)
        caminho_lista = rascunho / "resample.txt"
        caminho_yaml = rascunho / "resample.yaml"
        caminho_completo = rascunho / "full.txt"
        cache = rascunho / "labels" / "test.cache"
        modelo = YOLO(str(CAMINHO_MODELO))

        def escreve_yaml(lista):
            caminho_yaml.write_text(
                f"train: {lista}\nval: {lista}\nnc: {len(NOMES_CLASSES)}\nnames:\n"
                + "\n".join(f"  - {c}" for c in NOMES_CLASSES) + "\n")

        # PT: checagem 1: estimativas pontuais (lista completa, sem reamostrar)
        # EN: check 1: point estimates (full list, no resampling)
        caminho_completo.write_text("\n".join(str(p) for p in imgs))
        escreve_yaml(caminho_completo)
        print("\n" + "=" * 72)
        print("PRECISAO / mAP -- gate 1: estimativas pontuais")
        print("=" * 72)
        pts = {}
        for md, ref in ((300, PONTO_300), (2000, PONTO_2000)):
            if cache.exists():
                cache.unlink()
            pts[md] = val(modelo, caminho_yaml, md)
            desvio = max(abs(pts[md][k] - ref[k]) for k in ref)
            print(f"   max_det={md:4d}  P={pts[md]['precision']:.4f} R={pts[md]['recall']:.4f} "
                  f"mAP50={pts[md]['map50']:.4f} mAP50-95={pts[md]['map50_95']:.4f}  "
                  f"desvio max vs referencia={desvio:.5f}")
            if desvio > TOL_CHECAGEM_PONTO:
                print("\n   !! estimativa pontual nao reproduz -- abortando antes das replicas.")
                return None

        # PT: réplicas: mesmas reamostragens (mesma semente e mesma ordem) para os dois
        #     braços
        # EN: replicates: same resamples (same seed and order) for both arms
        escreve_yaml(caminho_lista)
        gerador = np.random.default_rng(SEMENTE)
        matriz = {md: {k: np.empty(n_boot) for k in METRICAS} for md in MAX_DETS_TESTADOS}
        print(f"\nreplicas: {n_boot} x {len(MAX_DETS_TESTADOS)} bracos")
        t0 = time.time()
        for b in range(n_boot):
            indice = gerador.integers(0, n, size=n)
            caminho_lista.write_text("\n".join(str(imgs[i]) for i in indice))
            for md in MAX_DETS_TESTADOS:
                if cache.exists():
                    cache.unlink()
                m = val(modelo, caminho_yaml, md)
                for k in METRICAS:
                    matriz[md][k][b] = m[k]
            if (b + 1) % 50 == 0:
                el = time.time() - t0
                REGISTRADOR.info("replica %d/%d (%.0fs, %.2fs/replica, ~%.0f min restantes)",
                            b + 1, n_boot, el, el / (b + 1), el / (b + 1) * (n_boot - b - 1) / 60)
        return {"point": pts, "arrays": matriz}
    finally:
        shutil.rmtree(rascunho, ignore_errors=True)


def retomar_val(res, n_boot):
    pub = json.loads(RESUMO_25.read_text())["metrics"]["bootstrap_ci95"]
    ci = {md: {} for md in MAX_DETS_TESTADOS}
    for md in MAX_DETS_TESTADOS:
        for k in METRICAS:
            lo, hi = np.percentile(res["arrays"][md][k], [2.5, 97.5])
            ci[md][k] = {"mean": float(res["arrays"][md][k].mean()),
                         "low": float(lo), "high": float(hi)}

    print("\n" + "=" * 72)
    print(f"PRECISAO / mAP -- ICs bootstrap por imagem ({n_boot} replicas)")
    print("=" * 72)
    ok = True
    if n_boot == N_BOOT_VAL:
        print("SANIDADE -- max_det=300 reproduz os ICs de referência (script 25)?")
        for k in METRICAS:
            dl = abs(ci[300][k]["low"] - pub[k]["ci95_low"])
            dh = abs(ci[300][k]["high"] - pub[k]["ci95_high"])
            sinalizador = "" if max(dl, dh) <= TOL_CHECAGEM_IC else "  <-- DIVERGE"
            if max(dl, dh) > TOL_CHECAGEM_IC:
                ok = False
            print(f"   {k:9s} referência [{pub[k]['ci95_low']*100:5.2f}, {pub[k]['ci95_high']*100:5.2f}]"
                  f"  obtido [{ci[300][k]['low']*100:5.2f}, {ci[300][k]['high']*100:5.2f}]{sinalizador}")
        print("   ->", "SIM" if ok else "NAO")
    else:
        print(f"(modo teste, {n_boot} replicas: checagem contra a referência nao se aplica)")
    print()
    print(f"{'metrica':10s} {'max_det=300':>22s} {'max_det=2000':>22s}")
    for k in METRICAS:
        a, b = ci[300][k], ci[2000][k]
        print(f"{k:10s} [{a['low']*100:5.1f}, {a['high']*100:5.1f}] "
              f"     [{b['low']*100:5.1f}, {b['high']*100:5.1f}]")
    return ci, ok


# PT: --- principal ---
# EN: --- main ---

def principal():
    ap = argparse.ArgumentParser()
    ap.add_argument("--so-recall", action="store_true")
    ap.add_argument("--teste", type=int, default=0, help="N replicas; nao grava summary")
    args = ap.parse_args()

    for p in (CAMINHO_MODELO, RESUMO_61, RESUMO_14, RESUMO_25):
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
    ci, ok = retomar_val(res, n_boot)

    if args.teste:
        print("\nmodo teste: nada gravado.")
        return
    if not ok:
        print("\n!! sanidade falhou -- resultado NAO gravado como valido.")
        return

    print("\n" + "=" * 72)
    print("ICs DO DETECTOR (max_det=2000)")
    print("=" * 72)
    c = ci[2000]
    r = rec[2000]
    print(f"precision [{c['precision']['low']*100:.1f}%, {c['precision']['high']*100:.1f}%], "
          f"mAP@0.5 [{c['map50']['low']*100:.1f}%, {c['map50']['high']*100:.1f}%], "
          f"mAP@0.5:0.95 [{c['map50_95']['low']*100:.1f}%, {c['map50_95']['high']*100:.1f}%]; "
          f"matched recall {r['point']*100:.1f}% [{r['low']*100:.1f}%, {r['high']*100:.1f}%]")

    salvar_resumo_experimento(NOME_EXP, METADADOS_EXECUCAO, {
        "note": "ICs das metricas do detector com max_det explicito. Os de referencia (scripts 14 e 25) "
                "rodaram com o padrao 300 do Ultralytics. Mesmas reamostragens nos dois "
                "bracos; o braco de 300 reproduz a referencia (sanidade).",
        "model_path": str(CAMINHO_MODELO),
        "seed": SEMENTE, "n_bootstrap_val": N_BOOT_VAL, "n_bootstrap_recall": N_BOOT_RECALL,
        "conf_threshold": LIMIAR_CONF,
        "sanity_reproduces_published": ok,
        "point_estimates": {str(k): v for k, v in res["point"].items()},
        "recall_ci": {str(k): v for k, v in rec.items()},
        "val_ci": {str(k): v for k, v in ci.items()},
        "replicas": {str(md): {k: v.tolist() for k, v in d.items()}
                     for md, d in res["arrays"].items()},
    })
    print(f"\nLog completo: {CAMINHO_LOG}")


if __name__ == "__main__":
    principal()
