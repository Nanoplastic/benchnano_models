"""Avaliação das dobras de validação cruzada / Cross-validation fold evaluation.

PT: Avalia os 5 checkpoints da validação cruzada, cada um só nas imagens held-out da própria
    dobra (nunca vistas em treino nem na seleção de checkpoint dessa dobra), e agrega os 5
    resultados numa estimativa de recall por tamanho sobre as 102 imagens de origem (155
    partículas médias, 30 grandes; ver 43_build_cv_folds.py), bem maior que as 31/5 da
    partição de teste fixa.
    Pareamento por IoU: guloso, mesma classe, IoU >= 0,5 (mesma aproximação de
    14_yolo_test_eval.py / 25_bootstrap_yolo_detection_metrics.py; não é idêntico bit a bit
    ao validador interno do Ultralytics).
    Análise complementar: verifica se um conjunto de avaliação maior estreita o intervalo de
    confiança do recall obtido no teste fixo de 15 imagens.
EN: Evaluates the 5 cross-validation checkpoints, each only on its own fold's held-out
    images (never seen in training or in that fold's checkpoint selection), and aggregates
    the 5 results into a recall-by-size estimate over the 102 source images (155 medium, 30
    large particles; see 43_build_cv_folds.py), much larger than the 31/5 of the fixed test
    split.
    IoU matching: greedy, same class, IoU >= 0.5 (same approximation as 14_yolo_test_eval.py
    / 25_bootstrap_yolo_detection_metrics.py; not bit-identical to the internal Ultralytics
    validator).
    Complementary analysis: checks whether a larger evaluation set narrows the confidence
    interval of the recall obtained on the fixed 15-image test set.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import salvar_resumo_experimento, configurar_log_experimento

from ultralytics import YOLO

NOME_EXP = "45_eval_cv_folds"
DIR_EXP, CAMINHO_LOG, REGISTRADOR, METADADOS_EXECUCAO = configurar_log_experimento(NOME_EXP, __file__)

RAIZ = Path(__file__).resolve().parent.parent
DIR_DATASETS_CV = RAIZ / "datasets_yolo26_v2_cv"
DIR_EXECUCOES_CV = RAIZ / "runs_v2_cv"
NOMES_CLASSES = ["PE", "PET", "PP", "PS"]
K_DOBRAS = 5

TAMANHO_IMG = 1024
LIMIAR_CONF = 0.15
LIMIAR_IOU = 0.5
N_BOOTSTRAP = 10_000
SEMENTE = 42

LIMIAR_PEQUENO = 0.01
LIMIAR_MEDIO = 0.05

# PT: valores de referência da partição de teste fixa (15 imagens, 14_yolo_test_eval.py),
#     para comparação lado a lado
# EN: reference values from the fixed test split (15 images, 14_yolo_test_eval.py), for
#     side-by-side comparison
IC95_RECALL_REFERENCIA = [0.247, 0.518]
N_MEDIO_REFERENCIA = 31
N_GRANDE_REFERENCIA = 5


def classificar_tamanho(area_normalizada: float) -> str:
    if area_normalizada < LIMIAR_PEQUENO:
        return "small"
    if area_normalizada < LIMIAR_MEDIO:
        return "medium"
    return "large"


def yolo_para_xyxy_px(cx, cy, w, h, W, H):
    return ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)


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


def carregar_gt(caminho_rotulo: Path, W: int, H: int):
    saida = []
    if not caminho_rotulo.exists():
        return saida
    for linha in open(caminho_rotulo):
        partes = linha.split()
        if len(partes) < 5:
            continue
        id_cls = int(partes[0])
        cx, cy, w, h = map(float, partes[1:5])
        xyxy = yolo_para_xyxy_px(cx, cy, w, h, W, H)
        saida.append({"cls_id": id_cls, "xyxy": xyxy, "size": classificar_tamanho(w * h)})
    return saida


def parear_mesma_classe(gts, preds):
    candidatas = []
    for gi, g in enumerate(gts):
        for pi, p in enumerate(preds):
            if g["cls_id"] != p["cls_id"]:
                continue
            iou = iou_xyxy(g["xyxy"], p["xyxy"])
            if iou >= LIMIAR_IOU:
                candidatas.append((iou, gi, pi))
    candidatas.sort(reverse=True)
    gt_pareado = set()
    pred_pareada = set()
    for iou, gi, pi in candidatas:
        if gi in gt_pareado or pi in pred_pareada:
            continue
        gt_pareado.add(gi)
        pred_pareada.add(pi)
    return gt_pareado


def ic_recall_bootstrap(registros, filtro_tamanho, semente=SEMENTE, n_bootstrap=N_BOOTSTRAP):
    """PT: registros: lista de dict(source_image, size, matched). Bootstrap por imagem de
        origem.
    EN: registros: list of dict(source_image, size, matched). Source-image bootstrap.
    """
    sub = [r for r in registros if filtro_tamanho(r["size"])]
    if not sub:
        return None
    df = pd.DataFrame(sub)
    imgs = df["source_image"].unique()
    grupos = {s: g for s, g in df.groupby("source_image")}
    gerador = np.random.default_rng(semente)
    recalls = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        amostrados = gerador.choice(imgs, size=len(imgs), replace=True)
        rep = pd.concat([grupos[s] for s in amostrados], ignore_index=True)
        recalls[i] = rep["matched"].mean()
    ci = np.nanpercentile(recalls, [2.5, 97.5])
    return {
        "n": len(sub),
        "n_images": len(imgs),
        "recall_point": float(df["matched"].mean()),
        "ci95": [float(ci[0]), float(ci[1])],
    }


def principal():
    ausentes = []
    for k in range(K_DOBRAS):
        ckpt = DIR_EXECUCOES_CV / f"fold{k}" / "weights" / "best.pt"
        if not ckpt.exists():
            ausentes.append(str(ckpt))
    if ausentes:
        REGISTRADOR.error("checkpoints faltando: %s -- rode 44_train_cv_fold.py --fold K pra cada fold antes", ausentes)
        for m in ausentes:
            print(f"FALTANDO: {m}")
        sys.exit(1)

    REGISTRADOR.info("=== avaliando %d folds de cross-validation (held-out only) ===", K_DOBRAS)

    todos_registros = []  # PT: source_image, size, matched, fold / EN: source_image, size, matched, fold
    resumo_por_dobra = []

    for k in range(K_DOBRAS):
        caminho_ckpt = DIR_EXECUCOES_CV / f"fold{k}" / "weights" / "best.pt"
        dir_img = DIR_DATASETS_CV / f"fold{k}" / "images" / "test"
        dir_rotulo = DIR_DATASETS_CV / f"fold{k}" / "labels" / "test"
        modelo = YOLO(str(caminho_ckpt))

        caminhos_img = sorted(dir_img.glob("*.png"))
        n_gt_fold = 0
        n_pareados_dobra = 0
        for caminho_img in caminhos_img:
            imagem = Image.open(caminho_img).convert("RGB")
            W, H = imagem.size
            gts = carregar_gt(dir_rotulo / f"{caminho_img.stem}.txt", W, H)

            pred = modelo.predict(source=str(caminho_img), imgsz=TAMANHO_IMG, conf=LIMIAR_CONF, verbose=False)[0]
            preds = []
            if pred.boxes is not None and len(pred.boxes):
                xyxy_px = pred.boxes.xyxy.cpu().numpy()
                ids_cls = pred.boxes.cls.cpu().numpy().astype(int)
                for caixa, c in zip(xyxy_px, ids_cls):
                    preds.append({"cls_id": int(c), "xyxy": tuple(caixa)})

            gt_pareado = parear_mesma_classe(gts, preds)
            for gi, g in enumerate(gts):
                pareado = gi in gt_pareado
                todos_registros.append({
                    "source_image": caminho_img.name, "fold": k,
                    "size": g["size"], "matched": pareado,
                })
                n_gt_fold += 1
                n_pareados_dobra += int(pareado)

            REGISTRADOR.info("fold=%d imagem=%s gt=%d matched=%d", k, caminho_img.name, len(gts), len(gt_pareado))

        recall_dobra = n_pareados_dobra / n_gt_fold if n_gt_fold else float("nan")
        resumo_por_dobra.append({"fold": k, "n_images": len(caminhos_img), "n_gt": n_gt_fold, "recall": recall_dobra})
        print(f"fold {k}: {len(caminhos_img)} imagens held-out, n_gt={n_gt_fold}, recall={recall_dobra:.3f}")

    df_todos = pd.DataFrame(todos_registros)
    csv_saida = DIR_EXP / "reports" / "cv_pooled_predictions.csv"
    csv_saida.parent.mkdir(parents=True, exist_ok=True)
    df_todos.to_csv(csv_saida, index=False)

    print(f"\n=== Recall pooled por tamanho (5 folds, cada imagem avaliada 1x fora do treino) ===")
    resultados_faixa = {}
    for tamanho in ("small", "medium", "large"):
        res = ic_recall_bootstrap(todos_registros, lambda s, sz=tamanho: s == sz)
        resultados_faixa[tamanho] = res
        if res:
            print(f"{tamanho}: n={res['n']} ({res['n_images']} imagens) recall={res['recall_point']:.3f} "
                  f"IC95%=[{res['ci95'][0]:.3f}, {res['ci95'][1]:.3f}]")

    geral = ic_recall_bootstrap(todos_registros, lambda s: True)
    print(f"\ngeral: n={geral['n']} recall={geral['recall_point']:.3f} IC95%={geral['ci95']}")

    print(f"\n=== Comparação com a referência (partição de teste travada, n={N_MEDIO_REFERENCIA} médio / {N_GRANDE_REFERENCIA} grande) ===")
    print(f"referência (recall geral casado por tamanho): IC95%={IC95_RECALL_REFERENCIA}")
    if resultados_faixa.get("medium"):
        print(f"CV pooled médio: n={resultados_faixa['medium']['n']} (vs {N_MEDIO_REFERENCIA} da referência) IC95%={resultados_faixa['medium']['ci95']}")
    if resultados_faixa.get("large"):
        print(f"CV pooled grande: n={resultados_faixa['large']['n']} (vs {N_GRANDE_REFERENCIA} da referência) IC95%={resultados_faixa['large']['ci95']}")

    salvar_resumo_experimento(NOME_EXP, METADADOS_EXECUCAO, {
        "note": "resultado exploratorio/complementar -- o teste travado de 15 imagens continua sendo o resultado principal",
        "k_folds": K_DOBRAS,
        "per_fold_summary": resumo_por_dobra,
        "bucket_results": resultados_faixa,
        "overall": geral,
        "published_comparison": {
            "published_recall_ci95_overall": IC95_RECALL_REFERENCIA,
            "published_n_medium": N_MEDIO_REFERENCIA,
            "published_n_large": N_GRANDE_REFERENCIA,
        },
        "iou_threshold": LIMIAR_IOU,
        "conf_threshold": LIMIAR_CONF,
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEMENTE,
    })
    print(f"\nCSV pooled: {csv_saida}")
    print(f"Log completo: {CAMINHO_LOG}")


if __name__ == "__main__":
    principal()
