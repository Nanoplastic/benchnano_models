"""Recall do detector por regime de tamanho / Detector recall by size regime.

PT: Recall do YOLO26L v2 estratificado pelo regime de tamanho físico
    (NANOPLASTIC/MICROPLASTIC, limiar de 1 µm) em vez da área normalizada da caixa de
    14_yolo_test_eval.py. Permite cruzar o recall de localização por tamanho com a acurácia
    do SmolVLM no subconjunto nano.
    Usa o mesmo pareamento por IoU (guloso, mesma classe, LIMIAR_IOU=0,5, conf=0,15) e o
    mesmo bootstrap por imagem (10000 réplicas, semente 42) de 14_yolo_test_eval.py; muda só
    o critério de faixa: calibração px->nm (100/ampliação) de
    15_build_size_regime_manifest.py, que reproduz as contagens 18329/2791/4610 (total) e
    5198/1571/2630 (NANOPLASTIC) por partição.
EN: YOLO26L v2 recall stratified by physical size regime (NANOPLASTIC/MICROPLASTIC, 1 µm
    threshold) instead of the normalized box area used in 14_yolo_test_eval.py. Allows
    crossing localization recall by size with SmolVLM accuracy on the nano subset.
    Uses the same IoU matching (greedy, same class, LIMIAR_IOU=0.5, conf=0.15) and the same
    per-image bootstrap (10000 replicates, seed 42) as 14_yolo_test_eval.py; only the band
    criterion changes: px->nm calibration (100/magnification) from
    15_build_size_regime_manifest.py, which reproduces the counts 18329/2791/4610 (total)
    and 5198/1571/2630 (NANOPLASTIC) per split.
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

NOME_EXP = "23_recall_by_size_regime"
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


def carregar_gt(caminho_rotulo: Path, W: int, H: int, um_por_px: float):
    if not caminho_rotulo.exists():
        return []
    saida = []
    for linha in open(caminho_rotulo):
        partes = linha.split()
        if len(partes) < 5:
            continue
        id_cls = int(partes[0])
        cx, cy, w, h = map(float, partes[1:5])
        tamanho_px = max(w * W, h * H)
        tamanho_nm = tamanho_px * um_por_px * 1000.0
        regime = "NANOPLASTIC" if tamanho_nm < NANO_MAX_NM else "MICROPLASTIC"
        saida.append((id_cls, yolo_para_xyxy(cx, cy, w, h), regime))
    return saida


def parear_predicoes(modelo, caminho_img: Path, gts):
    pred = modelo.predict(source=str(caminho_img), imgsz=TAMANHO_IMG, conf=LIMIAR_CONF, verbose=False)[0]

    preds = []
    if pred.boxes is not None and len(pred.boxes):
        xyxy_norm = pred.boxes.xyxyn.cpu().numpy()
        ids_cls = pred.boxes.cls.cpu().numpy().astype(int)
        for caixa, c in zip(xyxy_norm, ids_cls):
            preds.append((int(c), tuple(caixa)))

    candidatas = []
    for gi, (cls_g, caixa_g, _regime_tamanho) in enumerate(gts):
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


def principal():
    if not CAMINHO_MODELO.exists():
        print(f"Checkpoint nao existe em {CAMINHO_MODELO}")
        sys.exit(1)

    REGISTRADOR.info("=== recall do YOLO26L v2 por regime de tamanho fisico (test set travado) ===")
    modelo = YOLO(str(CAMINHO_MODELO))

    faixa_gt = defaultdict(int)
    faixa_pareada = defaultdict(int)
    por_imagem = []  # PT: (nome_imagem, regime -> (pareados, gt)) / EN: (image_name, regime -> (matched, gt))

    caminhos_img = sorted((DIR_TESTE / "images/test").glob("*.png"))
    n_ampl_nao_resolvida = 0
    for caminho_img in caminhos_img:
        ampl = interpretar_ampliacao(caminho_img.stem)
        if ampl is None:
            n_ampl_nao_resolvida += 1
            continue
        with Image.open(caminho_img) as im:
            W, H = im.size
        um_por_px = 100.0 / ampl

        caminho_rotulo = DIR_TESTE / "labels/test" / f"{caminho_img.stem}.txt"
        gts = carregar_gt(caminho_rotulo, W, H, um_por_px)
        if not gts:
            continue

        gt_pareado = parear_predicoes(modelo, caminho_img, gts)

        contagens_por_imagem = defaultdict(lambda: [0, 0])  # PT: regime -> [pareados, gt] / EN: regime -> [matched, gt]
        for gi, (cls_g, caixa_g, regime) in enumerate(gts):
            faixa_gt[regime] += 1
            contagens_por_imagem[regime][1] += 1
            if gi in gt_pareado:
                faixa_pareada[regime] += 1
                contagens_por_imagem[regime][0] += 1

        por_imagem.append((caminho_img.name, dict(contagens_por_imagem)))
        REGISTRADOR.info("imagem=%s gt=%d casados=%d", caminho_img.name, len(gts), len(gt_pareado))

    print(f"{'regime':13s}{'GT':>6s}{'casados':>9s}{'recall':>9s}")
    resultados_regime = {}
    for regime in ("NANOPLASTIC", "MICROPLASTIC"):
        gt = faixa_gt[regime]
        m = faixa_pareada[regime]
        recall = m / gt if gt else float("nan")
        resultados_regime[regime] = {"gt": gt, "matched": m, "recall": recall}
        print(f"{regime:13s}{gt:6d}{m:9d}{recall:9.3f}")

    # PT: bootstrap por imagem, separado por regime (mesma lógica de 14_yolo_test_eval.py)
    # EN: per-image bootstrap, split by regime (same logic as 14_yolo_test_eval.py)
    gerador = np.random.default_rng(SEMENTE)
    n_imagens = len(por_imagem)
    resultados_ic = {}
    for regime in ("NANOPLASTIC", "MICROPLASTIC"):
        recalls = np.empty(N_BOOTSTRAP)
        for b in range(N_BOOTSTRAP):
            indice = gerador.integers(0, n_imagens, size=n_imagens)
            total_pareados = sum(por_imagem[i][1].get(regime, [0, 0])[0] for i in indice)
            total_gt = sum(por_imagem[i][1].get(regime, [0, 0])[1] for i in indice)
            recalls[b] = total_pareados / total_gt if total_gt else float("nan")
        ic_inf, ic_sup = np.nanpercentile(recalls, [2.5, 97.5])
        ponto = resultados_regime[regime]["recall"]
        resultados_ic[regime] = {"point": ponto, "low": float(ic_inf), "high": float(ic_sup)}
        print(f"IC 95% {regime}: [{ic_inf:.3f}, {ic_sup:.3f}]")

    salvar_resumo_experimento(NOME_EXP, METADADOS_EXECUCAO, {
        "model_path": str(CAMINHO_MODELO),
        "n_test_images": n_imagens,
        "n_unresolved_magnification": n_ampl_nao_resolvida,
        "size_regime_recall": resultados_regime,
        "size_regime_recall_bootstrap_ci95": resultados_ic,
        "iou_threshold": LIMIAR_IOU,
        "conf_threshold": LIMIAR_CONF,
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEMENTE,
    })
    print(f"\nLog completo: {CAMINHO_LOG}")


if __name__ == "__main__":
    principal()
