"""Avaliação das dobras nano da validação cruzada / Nano cross-validation fold evaluation.

PT: Avalia os 5 checkpoints da validação cruzada nano (49/50), cada um só nas imagens
    held-out da própria dobra. Diferente de 45_eval_cv_folds.py (faixas por área normalizada
    da caixa), aqui a faixa é o regime físico nano/micro (calibração px->nm = 100/ampliação,
    limiar NANO_MAX_NM = 1000 nm de 23_recall_by_size_regime.py), para medir o recall NANO
    do detector nano.
    A partição "test" de cada dobra nano (de 49_build_nano_cv_folds.py) tem o GT completo,
    sem filtro; o filtro nano é aplicado aqui, na avaliação, como em 23/45.
    Mesmo pareamento por IoU (guloso, mesma classe, IoU >= 0,5) e mesmo bootstrap por imagem
    (10000 réplicas, semente 42) de 45_eval_cv_folds.py. Análise complementar, como a do 45.
EN: Evaluates the 5 nano cross-validation checkpoints (49/50), each only on its own fold's
    held-out images. Unlike 45_eval_cv_folds.py (bands by normalized box area), here the
    band is the physical nano/micro regime (px->nm = 100/magnification calibration,
    NANO_MAX_NM = 1000 nm threshold from 23_recall_by_size_regime.py), to measure the nano
    detector's NANO recall.
    Each nano fold's "test" split (from 49_build_nano_cv_folds.py) holds the complete,
    unfiltered GT; the nano filter is applied here, at evaluation, as in 23/45.
    Same IoU matching (greedy, same class, IoU >= 0.5) and per-image bootstrap (10000
    replicates, seed 42) as 45_eval_cv_folds.py. Complementary analysis, like 45.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import salvar_resumo_experimento, configurar_log_experimento

from ultralytics import YOLO

NOME_EXP = "51_eval_cv_folds_nano"
DIR_EXP, CAMINHO_LOG, REGISTRADOR, METADADOS_EXECUCAO = configurar_log_experimento(NOME_EXP, __file__)

RAIZ = Path(__file__).resolve().parent.parent
DIR_DATASETS_CV = RAIZ / "datasets_yolo26_v2_cv_nano"
DIR_EXECUCOES_CV = RAIZ / "runs_v2_cv_nano"
NOMES_CLASSES = ["PE", "PET", "PP", "PS"]
K_DOBRAS = 5

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


def carregar_gt(caminho_rotulo: Path, W: int, H: int, um_por_px: float):
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
        tamanho_px = max(w * W, h * H)
        tamanho_nm = tamanho_px * um_por_px * 1000.0
        regime = "NANOPLASTIC" if tamanho_nm < NANO_MAX_NM else "MICROPLASTIC"
        saida.append({"cls_id": id_cls, "xyxy": xyxy, "size": regime})
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
        REGISTRADOR.error("checkpoints faltando: %s -- rode 50_train_cv_fold_nano.py --fold K pra cada fold antes", ausentes)
        for m in ausentes:
            print(f"FALTANDO: {m}")
        sys.exit(1)

    REGISTRADOR.info("=== avaliando %d folds de cross-validation nano-only (held-out only) ===", K_DOBRAS)

    todos_registros = []
    resumo_por_dobra = []
    n_ampl_nao_resolvida = 0

    for k in range(K_DOBRAS):
        caminho_ckpt = DIR_EXECUCOES_CV / f"fold{k}" / "weights" / "best.pt"
        dir_img = DIR_DATASETS_CV / f"fold{k}" / "images" / "test"
        dir_rotulo = DIR_DATASETS_CV / f"fold{k}" / "labels" / "test"
        modelo = YOLO(str(caminho_ckpt))

        caminhos_img = sorted(dir_img.glob("*.png"))
        n_gt_fold = 0
        n_pareados_dobra = 0
        for caminho_img in caminhos_img:
            ampl = interpretar_ampliacao(caminho_img.stem)
            if ampl is None:
                n_ampl_nao_resolvida += 1
                continue
            um_por_px = 100.0 / ampl

            imagem = Image.open(caminho_img).convert("RGB")
            W, H = imagem.size
            gts = carregar_gt(dir_rotulo / f"{caminho_img.stem}.txt", W, H, um_por_px)

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
    csv_saida = DIR_EXP / "reports" / "cv_pooled_predictions_nano.csv"
    csv_saida.parent.mkdir(parents=True, exist_ok=True)
    df_todos.to_csv(csv_saida, index=False)

    print(f"\n=== Recall pooled por regime fisico (5 folds nano-only, cada imagem avaliada 1x fora do treino) ===")
    resultados_faixa = {}
    for regime in ("NANOPLASTIC", "MICROPLASTIC"):
        res = ic_recall_bootstrap(todos_registros, lambda s, r=regime: s == r)
        resultados_faixa[regime] = res
        if res:
            print(f"{regime}: n={res['n']} ({res['n_images']} imagens) recall={res['recall_point']:.3f} "
                  f"IC95%=[{res['ci95'][0]:.3f}, {res['ci95'][1]:.3f}]")

    geral = ic_recall_bootstrap(todos_registros, lambda s: True)
    print(f"\ngeral: n={geral['n']} recall={geral['recall_point']:.3f} IC95%={geral['ci95']}")

    salvar_resumo_experimento(NOME_EXP, METADADOS_EXECUCAO, {
        "note": "resultado exploratorio/complementar, detector nano-only -- compara com 45_eval_cv_folds.py (dataset misto) e com o SAHI fine-tunado (40/41)",
        "nano_max_nm": NANO_MAX_NM,
        "k_folds": K_DOBRAS,
        "n_unresolved_magnification": n_ampl_nao_resolvida,
        "per_fold_summary": resumo_por_dobra,
        "bucket_results": resultados_faixa,
        "overall": geral,
        "iou_threshold": LIMIAR_IOU,
        "conf_threshold": LIMIAR_CONF,
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEMENTE,
    })
    print(f"\nCSV pooled: {csv_saida}")
    print(f"Log completo: {CAMINHO_LOG}")


if __name__ == "__main__":
    principal()
