"""Recall por faixa e regime com max_det=2000 / Recall by band and regime with max_det=2000.

PT: Variante de 14_yolo_test_eval.py (faixa por área normalizada) e
    23_recall_by_size_regime.py (faixa por regime físico px->nm) com max_det=2000. Ambos
    usam pareamento guloso, mesma classe, IoU >= 0,5, conf = 0,15, e originalmente não
    passavam max_det (padrão 300). Gera o recall por faixa de tamanho e por regime físico da
    execução única.
    Checagem (max_det=300 deve reproduzir):
    14_yolo_test_eval: small=33,47% medium=51,61% large=20,00%
    23_recall_by_size_regime: NANO=31,10% [15,55; 60,40] MICRO=36,87% [25,48; 61,36]
    Grava em resultados/71_reeval_bucket_regime_recall_maxdet2000/, sem sobrescrever
    14_yolo_test_eval nem 23_recall_by_size_regime.
    Uso:
    python3 -u 71_reeval_bucket_regime_recall_maxdet2000.py
EN: Variant of 14_yolo_test_eval.py (bands by normalized area) and
    23_recall_by_size_regime.py (bands by physical px->nm regime) with max_det=2000. Both
    use greedy, same-class matching, IoU >= 0.5, conf = 0.15, and originally did not pass
    max_det (default 300). Produces the single-run recall per size band and per physical
    regime.
    Check (max_det=300 must reproduce):
    14_yolo_test_eval: small=33.47% medium=51.61% large=20.00%
    23_recall_by_size_regime: NANO=31.10% [15.55, 60.40] MICRO=36.87% [25.48, 61.36]
    Writes to resultados/71_reeval_bucket_regime_recall_maxdet2000/, without overwriting
    14_yolo_test_eval or 23_recall_by_size_regime.
    Usage:
    python3 -u 71_reeval_bucket_regime_recall_maxdet2000.py
"""
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import salvar_resumo_experimento, configurar_log_experimento

from ultralytics import YOLO

NOME_EXP = "71_reeval_bucket_regime_recall_maxdet2000"
DIR_EXP, CAMINHO_LOG, REGISTRADOR, METADADOS_EXECUCAO = configurar_log_experimento(NOME_EXP, __file__)

RAIZ = Path(__file__).resolve().parent.parent
DIR_TESTE = RAIZ / "datasets_yolo26_v2"
CAMINHO_MODELO = RAIZ / "runs_v2" / "microplastic_yolo26_v2" / "weights" / "best.pt"

TAMANHO_IMG = 1024
LIMIAR_CONF = 0.15
LIMIAR_IOU = 0.5
N_BOOTSTRAP = 10_000
SEMENTE = 42
NANO_MAX_NM = 1000.0
MAX_DETS_TESTADOS = (300, 2000)

REFERENCIA_FAIXA = {"small": 0.3347179711412331, "medium": 0.5161290322580645, "large": 0.2}
REFERENCIA_REGIME = {"NANOPLASTIC": {"recall": 0.31102661596958175, "low": 0.1555494405859077, "high": 0.6040493480251318},
                     "MICROPLASTIC": {"recall": 0.3686868686868687, "low": 0.25483942469339704, "high": 0.613550456316928}}
TOL_CHECAGEM = 0.002

AMPL_X = re.compile(r"(\d+)\s*[xX]")
AMPL_TRACO = re.compile(r"^\D*(\d+)-\d+")


def interpretar_ampliacao(nome: str):
    m = AMPL_X.findall(nome)
    if m:
        return int(m[-1])
    m = AMPL_TRACO.match(nome)
    if m:
        return int(m.group(1))
    return None


def yolo_para_xyxy(cx, cy, w, h):
    return (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    intersecao = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    uniao = area_a + area_b - intersecao
    return intersecao / uniao if uniao > 0 else 0.0


def classificar_tamanho(area_normalizada):
    if area_normalizada < 0.01:
        return "small"
    if area_normalizada < 0.05:
        return "medium"
    return "large"


def carregar_gt_ambos(caminho_rotulo: Path, W: int, H: int, um_por_px):
    """PT: Retorna (cls_id, xyxy, area_normalizada, regime) por partícula.
    EN: Returns (cls_id, xyxy, area_norm, regime) per particle.
    """
    saida = []
    if not caminho_rotulo.exists():
        return saida
    for linha in open(caminho_rotulo):
        partes = linha.split()
        if len(partes) < 5:
            continue
        id_cls = int(partes[0])
        cx, cy, w, h = map(float, partes[1:5])
        area_normalizada = w * h
        regime = None
        if um_por_px is not None:
            tamanho_px = max(w * W, h * H)
            tamanho_nm = tamanho_px * um_por_px * 1000.0
            regime = "NANOPLASTIC" if tamanho_nm < NANO_MAX_NM else "MICROPLASTIC"
        saida.append((id_cls, yolo_para_xyxy(cx, cy, w, h), area_normalizada, regime))
    return saida


def parear_predicoes(modelo, caminho_img: Path, gts, max_det):
    pred = modelo.predict(source=str(caminho_img), imgsz=TAMANHO_IMG, conf=LIMIAR_CONF,
                         max_det=max_det, verbose=False)[0]
    preds = []
    if pred.boxes is not None and len(pred.boxes):
        xyxy_norm = pred.boxes.xyxyn.cpu().numpy()
        ids_cls = pred.boxes.cls.cpu().numpy().astype(int)
        for caixa, c in zip(xyxy_norm, ids_cls):
            preds.append((int(c), tuple(caixa)))

    candidatas = []
    for gi, (cls_g, caixa_g, _area_caixa, _regime_tamanho) in enumerate(gts):
        for pi, (cls_p, caixa_p) in enumerate(preds):
            if cls_p != cls_g:
                continue
            iou = iou_xyxy(caixa_g, caixa_p)
            if iou >= LIMIAR_IOU:
                candidatas.append((iou, gi, pi))
    candidatas.sort(reverse=True)

    gt_pareado, pred_pareada = set(), set()
    for iou, gi, pi in candidatas:
        if gi in gt_pareado or pi in pred_pareada:
            continue
        gt_pareado.add(gi)
        pred_pareada.add(pi)
    return gt_pareado


def executar(modelo, max_det):
    faixa_gt, faixa_pareada = defaultdict(int), defaultdict(int)
    regime_gt, regime_pareado = defaultdict(int), defaultdict(int)
    por_imagem_faixa = []
    por_imagem_regime = []

    caminhos_img = sorted((DIR_TESTE / "images/test").glob("*.png"))
    for caminho_img in caminhos_img:
        ampl = interpretar_ampliacao(caminho_img.stem)
        um_por_px = 100.0 / ampl if ampl else None
        with Image.open(caminho_img) as im:
            W, H = im.size
        caminho_rotulo = DIR_TESTE / "labels/test" / f"{caminho_img.stem}.txt"
        gts = carregar_gt_ambos(caminho_rotulo, W, H, um_por_px)
        if not gts:
            continue

        gt_pareado = parear_predicoes(modelo, caminho_img, gts, max_det)

        pb = defaultdict(lambda: [0, 0])
        pr = defaultdict(lambda: [0, 0])
        for gi, (cls_g, caixa_g, area, regime) in enumerate(gts):
            faixa_tamanho = classificar_tamanho(area)
            faixa_gt[faixa_tamanho] += 1
            pb[faixa_tamanho][1] += 1
            if gi in gt_pareado:
                faixa_pareada[faixa_tamanho] += 1
                pb[faixa_tamanho][0] += 1
            if regime:
                regime_gt[regime] += 1
                pr[regime][1] += 1
                if gi in gt_pareado:
                    regime_pareado[regime] += 1
                    pr[regime][0] += 1
        por_imagem_faixa.append((caminho_img.name, dict(pb)))
        por_imagem_regime.append((caminho_img.name, dict(pr)))
        REGISTRADOR.info("max_det=%d imagem=%s gt=%d casados=%d", max_det, caminho_img.name, len(gts), len(gt_pareado))

    resultados_faixa = {k: {"gt": faixa_gt[k], "matched": faixa_pareada[k],
                          "recall": faixa_pareada[k] / faixa_gt[k] if faixa_gt[k] else None}
                      for k in ("small", "medium", "large")}

    gerador = np.random.default_rng(SEMENTE)
    n_imagens = len(por_imagem_regime)
    resultados_regime = {}
    for regime in ("NANOPLASTIC", "MICROPLASTIC"):
        gt, m = regime_gt[regime], regime_pareado[regime]
        recalls = np.empty(N_BOOTSTRAP)
        for b in range(N_BOOTSTRAP):
            indice = gerador.integers(0, n_imagens, size=n_imagens)
            tm = sum(por_imagem_regime[i][1].get(regime, [0, 0])[0] for i in indice)
            tg = sum(por_imagem_regime[i][1].get(regime, [0, 0])[1] for i in indice)
            recalls[b] = tm / tg if tg else float("nan")
        lo, hi = np.nanpercentile(recalls, [2.5, 97.5])
        resultados_regime[regime] = {"gt": gt, "matched": m, "recall": m / gt if gt else None,
                                  "ci95_low": float(lo), "ci95_high": float(hi)}

    return resultados_faixa, resultados_regime


def principal():
    if not CAMINHO_MODELO.exists():
        raise SystemExit(f"checkpoint nao existe: {CAMINHO_MODELO}")
    modelo = YOLO(str(CAMINHO_MODELO))

    resultados = {}
    for md in MAX_DETS_TESTADOS:
        print(f"\n{'='*72}\nmax_det={md}\n{'='*72}")
        bucket, regime = executar(modelo, md)
        resultados[md] = {"bucket": bucket, "regime": regime}
        print("bucket:", {k: round(v["recall"] * 100, 2) if v["recall"] is not None else None
                          for k, v in bucket.items()})
        print("regime:", {k: (round(v["recall"] * 100, 2), round(v["ci95_low"] * 100, 2), round(v["ci95_high"] * 100, 2))
                          for k, v in regime.items()})

    b300 = resultados[300]["bucket"]
    r300 = resultados[300]["regime"]
    print(f"\n{'='*72}\nSANIDADE -- max_det=300 reproduz 14 e 23?\n{'='*72}")
    ok = True
    for k, pub in REFERENCIA_FAIXA.items():
        d = abs(b300[k]["recall"] - pub)
        if d > TOL_CHECAGEM:
            ok = False
        print(f"   bucket {k:7s} referência={pub*100:.2f}%  obtido={b300[k]['recall']*100:.2f}%  "
              f"dif={d*100:.2f}pp" + ("" if d <= TOL_CHECAGEM else "  <-- DIVERGE"))
    for k, pub in REFERENCIA_REGIME.items():
        d_ponto = abs(r300[k]["recall"] - pub["recall"])
        d_inf = abs(r300[k]["ci95_low"] - pub["low"])
        d_sup = abs(r300[k]["ci95_high"] - pub["high"])
        if max(d_ponto, d_inf, d_sup) > TOL_CHECAGEM:
            ok = False
        print(f"   regime {k:13s} referência={pub['recall']*100:.2f}% [{pub['low']*100:.2f},{pub['high']*100:.2f}]  "
              f"obtido={r300[k]['recall']*100:.2f}% [{r300[k]['ci95_low']*100:.2f},{r300[k]['ci95_high']*100:.2f}]"
              + ("" if max(d_ponto, d_inf, d_sup) <= TOL_CHECAGEM else "  <-- DIVERGE"))
    print("   ->", "SIM" if ok else "NAO")
    if not ok:
        print("\n   !! nao reproduz -- o resultado de max_det=2000 NAO deve ser usado.")
        return

    print(f"\n{'='*72}\nmax_det=300 (sanidade)  vs  max_det=2000\n{'='*72}")
    b2000, r2000 = resultados[2000]["bucket"], resultados[2000]["regime"]
    for k in ("small", "medium", "large"):
        print(f"   bucket {k:7s}  {b300[k]['recall']*100:.2f}%  ->  {b2000[k]['recall']*100:.2f}%")
    for k in ("NANOPLASTIC", "MICROPLASTIC"):
        print(f"   regime {k:13s}  {r300[k]['recall']*100:.2f}% [{r300[k]['ci95_low']*100:.2f},{r300[k]['ci95_high']*100:.2f}]"
              f"  ->  {r2000[k]['recall']*100:.2f}% [{r2000[k]['ci95_low']*100:.2f},{r2000[k]['ci95_high']*100:.2f}]")

    salvar_resumo_experimento(NOME_EXP, METADADOS_EXECUCAO, {
        "note": "reavaliacao do recall por bucket de tamanho (area_norm, 14_yolo_test_eval) "
                "e por regime fisico (px->nm, 23_recall_by_size_regime) com max_det explicito",
        "model_path": str(CAMINHO_MODELO),
        "conf_threshold": LIMIAR_CONF, "iou_threshold": LIMIAR_IOU,
        "sanity_reproduces_published": ok,
        "results": {str(k): v for k, v in resultados.items()},
    })
    print(f"\nLog completo: {CAMINHO_LOG}")


if __name__ == "__main__":
    principal()
