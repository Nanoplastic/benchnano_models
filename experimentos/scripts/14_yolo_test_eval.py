"""Avaliação do YOLO26L no conjunto de teste / YOLO26L test set evaluation.

PT: Avalia o YOLO26L v2 (13_train_yolo26_corrected.py) nas 15 imagens de
    datasets_yolo26_v2/images/test, nunca vistas em treino/val. Requer
    runs_v2/microplastic_yolo26_v2/weights/best.pt.
    Duas partes:
    1. Métricas padrão via `model.val()` do Ultralytics (precisão/recall/mAP por classe e
    agregadas), com bootstrap por imagem (não por instância) para IC 95% de
    precisão/recall/mAP50, na mesma lógica de 11_compare_experiments_statistical.py.
    2. Recall por faixa de tamanho com pareamento por IoU: cada caixa prevista é pareada
    1-para-1 com um GT da mesma classe com IoU >= LIMIAR_IOU (guloso, maior IoU primeiro), e
    o recall de cada faixa é calculado a partir desse pareamento.
    Com n=15 imagens a incerteza é grande ; os números devem ser
    lidos junto com o IC.
EN: Evaluates YOLO26L v2 (13_train_yolo26_corrected.py) on the 15 images in
    datasets_yolo26_v2/images/test, never seen in train/val. Requires
    runs_v2/microplastic_yolo26_v2/weights/best.pt.
    Two parts:
    1. Standard metrics via Ultralytics `model.val()` (per-class and aggregate
    precision/recall/mAP), with a per-image (not per-instance) bootstrap for 95% CIs of
    precision/recall/mAP50, following the same logic as
    11_compare_experiments_statistical.py.
    2. Recall per size band with IoU matching: each predicted box is matched 1-to-1 to a
    same-class GT with IoU >= LIMIAR_IOU (greedy, highest IoU first), and each band's recall
    is computed from that matching.
    With n=15 images the uncertainty is large ; the numbers should
    be read together with the CI.
"""
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import salvar_resumo_experimento, configurar_log_experimento

from ultralytics import YOLO

NOME_EXP = "14_yolo_test_eval"
DIR_EXP, CAMINHO_LOG, REGISTRADOR, METADADOS_EXECUCAO = configurar_log_experimento(NOME_EXP, __file__)

RAIZ = Path(__file__).resolve().parent.parent
DIR_TESTE = RAIZ / "datasets_yolo26_v2"
CAMINHO_MODELO = RAIZ / "runs_v2" / "microplastic_yolo26_v2" / "weights" / "best.pt"
NOMES_CLASSES = ["PE", "PET", "PP", "PS"]

TAMANHO_IMG = 1024
LIMIAR_CONF = 0.15
LIMIAR_IOU = 0.5
N_BOOTSTRAP = 10_000
SEMENTE = 42

# PT: limiares de área normalizada (área da bbox / área da imagem) das faixas
#     pequena/média/grande
# EN: normalized area thresholds (bbox area / image area) for the small/medium/large bands
LIMIAR_PEQUENO = 0.01
LIMIAR_MEDIO = 0.05


def classificar_tamanho(area_normalizada: float) -> str:
    if area_normalizada < LIMIAR_PEQUENO:
        return "small"
    if area_normalizada < LIMIAR_MEDIO:
        return "medium"
    return "large"


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


def carregar_gt(caminho_rotulo: Path):
    if not caminho_rotulo.exists():
        return []
    saida = []
    for linha in open(caminho_rotulo):
        partes = linha.split()
        if len(partes) < 5:
            continue
        id_cls = int(partes[0])
        cx, cy, w, h = map(float, partes[1:5])
        saida.append((id_cls, yolo_para_xyxy(cx, cy, w, h), w * h))
    return saida


def parear_predicoes(modelo, caminho_img: Path, gts):
    """PT: Roda o modelo numa imagem e pareia as predições com o GT por IoU (mesma classe).
    EN: Runs the model on one image and matches predictions to GT by IoU (same class).
    """
    pred = modelo.predict(source=str(caminho_img), imgsz=TAMANHO_IMG, conf=LIMIAR_CONF, verbose=False)[0]

    preds = []
    if pred.boxes is not None and len(pred.boxes):
        xyxy_norm = pred.boxes.xyxyn.cpu().numpy()
        ids_cls = pred.boxes.cls.cpu().numpy().astype(int)
        for caixa, c in zip(xyxy_norm, ids_cls):
            preds.append((int(c), tuple(caixa)))

    candidatas = []
    for gi, (cls_g, caixa_g, _) in enumerate(gts):
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

    n_fp = len(preds) - len(pred_pareada)
    return gt_pareado, n_fp, len(preds)


def principal():
    if not CAMINHO_MODELO.exists():
        msg = f"Checkpoint nao existe em {CAMINHO_MODELO} -- rode 13_train_yolo26_corrected.py primeiro."
        REGISTRADOR.error(msg)
        print(msg)
        sys.exit(1)

    REGISTRADOR.info("=== avaliacao final YOLO26L v2 no test set travado (15 imagens) ===")
    REGISTRADOR.info("model_path=%s", CAMINHO_MODELO)

    modelo = YOLO(str(CAMINHO_MODELO))

    # PT: --- Parte 1: métricas padrão via model.val() ---
    # EN: --- Part 1: standard metrics via model.val() ---
    yaml_dados_teste = DIR_TESTE / "data_test.yaml"
    yaml_dados_teste.write_text(
        f"path: {DIR_TESTE}\n"
        f"train: images/test\n"
        f"val: images/test\n"
        f"nc: {len(NOMES_CLASSES)}\n"
        f"names:\n" + "\n".join(f"  - {c}" for c in NOMES_CLASSES) + "\n"
    )
    print("=== Metricas padrao (model.val) no test set travado (15 imagens) ===")
    metricas = modelo.val(data=str(yaml_dados_teste), split="val", imgsz=TAMANHO_IMG,
                         conf=LIMIAR_CONF, iou=0.7, plots=True)
    agregado = {
        "precision": float(metricas.box.mp),
        "recall": float(metricas.box.mr),
        "map50": float(metricas.box.map50),
        "map50_95": float(metricas.box.map),
    }
    dados_por_classe = {}
    for i, nome in metricas.names.items():
        dados_por_classe[nome] = {
            "precision": float(metricas.box.p[i]),
            "recall": float(metricas.box.r[i]),
            "ap50": float(metricas.box.ap50[i]),
            "ap50_95": float(metricas.box.ap[i]),
        }
        print(f"  {nome:6s} P={metricas.box.p[i]:.3f}  R={metricas.box.r[i]:.3f}  "
              f"AP50={metricas.box.ap50[i]:.3f}  AP50-95={metricas.box.ap[i]:.3f}")
    print(f"Agregado: {agregado}")
    REGISTRADOR.info("metricas_agregadas=%s", agregado)
    REGISTRADOR.info("metricas_por_classe=%s", dados_por_classe)

    # PT: --- Parte 2: recall por tamanho com pareamento por IoU + bootstrap ---
    # EN: --- Part 2: recall by size with IoU matching + bootstrap ---
    print(f"\n=== Recall por tamanho de particula (IoU >= {LIMIAR_IOU}, casamento real) ===")
    REGISTRADOR.info("iniciando recall por tamanho com casamento por IoU (threshold=%.2f)", LIMIAR_IOU)

    faixa_gt = defaultdict(int)
    faixa_pareada = defaultdict(int)
    recall_por_imagem = []  # PT: (nome_imagem, pareados, total_gt) para o bootstrap por imagem / EN: (image_name, matched, gt_total) for the per-image bootstrap

    caminhos_img = sorted((DIR_TESTE / "images/test").glob("*.png"))
    for caminho_img in caminhos_img:
        caminho_rotulo = DIR_TESTE / "labels/test" / f"{caminho_img.stem}.txt"
        gts = carregar_gt(caminho_rotulo)
        if not gts:
            continue

        gt_pareado, n_fp, n_pred = parear_predicoes(modelo, caminho_img, gts)

        for gi, (cls_g, caixa_g, area) in enumerate(gts):
            tamanho = classificar_tamanho(area)
            faixa_gt[tamanho] += 1
            if gi in gt_pareado:
                faixa_pareada[tamanho] += 1

        recall_por_imagem.append((caminho_img.name, len(gt_pareado), len(gts)))
        REGISTRADOR.info("imagem=%s gt=%d casados=%d falsos_positivos=%d predicoes=%d",
                    caminho_img.name, len(gts), len(gt_pareado), n_fp, n_pred)

    print(f"{'tamanho':8s}{'GT':>6s}{'casados':>9s}{'recall':>9s}")
    resultados_tamanho = {}
    for tamanho in ("small", "medium", "large"):
        gt = faixa_gt[tamanho]
        m = faixa_pareada[tamanho]
        recall = m / gt if gt else float("nan")
        resultados_tamanho[tamanho] = {"gt": gt, "matched": m, "recall": recall}
        print(f"{tamanho:8s}{gt:6d}{m:9d}{recall:9.3f}")

    # PT: bootstrap por imagem (não por instância) para o IC 95% do recall global
    # EN: per-image (not per-instance) bootstrap for the 95% CI of the overall recall
    gerador = np.random.default_rng(SEMENTE)
    n_imagens = len(recall_por_imagem)
    recalls = np.empty(N_BOOTSTRAP)
    for b in range(N_BOOTSTRAP):
        indice = gerador.integers(0, n_imagens, size=n_imagens)
        amostrados = [recall_por_imagem[i] for i in indice]
        total_pareados = sum(s[1] for s in amostrados)
        total_gt = sum(s[2] for s in amostrados)
        recalls[b] = total_pareados / total_gt if total_gt else float("nan")
    ic_inf, ic_sup = np.nanpercentile(recalls, [2.5, 97.5])
    recall_pontual = sum(s[1] for s in recall_por_imagem) / sum(s[2] for s in recall_por_imagem)
    print(f"\nRecall global (casamento por IoU): {recall_pontual:.3f}  "
          f"IC 95% (bootstrap por imagem, n_img={n_imagens}, {N_BOOTSTRAP} replicas): "
          f"[{ic_inf:.3f}, {ic_sup:.3f}]")
    REGISTRADOR.info("recall_global=%.4f ic95_low=%.4f ic95_high=%.4f n_imagens=%d",
                recall_pontual, ic_inf, ic_sup, n_imagens)

    salvar_resumo_experimento(NOME_EXP, METADADOS_EXECUCAO, {
        "model_path": str(CAMINHO_MODELO),
        "n_test_images": n_imagens,
        "aggregate_metrics": agregado,
        "per_class_metrics": dados_por_classe,
        "size_stratified_recall": resultados_tamanho,
        "global_recall_bootstrap_ci95": {"point": recall_pontual, "low": float(ic_inf), "high": float(ic_sup)},
        "iou_threshold": LIMIAR_IOU,
        "conf_threshold": LIMIAR_CONF,
    })
    print(f"\nLog completo: {CAMINHO_LOG}")


if __name__ == "__main__":
    principal()
