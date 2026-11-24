"""Análise de erro por classe do detector nano / Nano detector per-class error analysis.

PT: Mesma análise de 42_class_error_analysis.py (baixa precisão de PE, baixo recall de PS),
    rodada no detector nano (runs_v2_nano/microplastic_yolo26_nano_ft/weights/best.pt, de
    48_finetune_yolo_nano_only.py) com o GT restrito a nanopartículas (< 1000 nm; calibração
    px->nm = 100/ampliação de 23_recall_by_size_regime.py). A partição de teste é lida do
    dataset original (datasets_yolo26_v2/), sem cópia nem filtro em disco; o filtro é
    aplicado ao montar o GT usado no pareamento.
    Mesmo pareamento (IoU guloso, mesma classe, IoU >= 0,5, conf = 0,15) e mesmo bootstrap
    pareado por imagem (10000 réplicas, semente 42) de 42_class_error_analysis.py.
    Análise exploratória: gera hipóteses sobre o comportamento do detector nano, não é
    resultado confirmatório.
EN: Same analysis as 42_class_error_analysis.py (low PE precision, low PS recall), run on
    the nano detector (runs_v2_nano/microplastic_yolo26_nano_ft/weights/best.pt, from
    48_finetune_yolo_nano_only.py) with GT restricted to nanoparticles (< 1000 nm; px->nm =
    100/magnification calibration from 23_recall_by_size_regime.py). The test split is read
    from the original dataset (datasets_yolo26_v2/), with no on-disk copy or filter; the
    filter is applied when building the GT used for matching.
    Same matching (greedy IoU, same class, IoU >= 0.5, conf = 0.15) and per-image paired
    bootstrap (10000 replicates, seed 42) as 42_class_error_analysis.py.
    Exploratory analysis: generates hypotheses about the nano detector's behaviour, not a
    confirmatory result.
"""
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fig_style import aplicar_estilo
from vlm_common import salvar_resumo_experimento, configurar_log_experimento

from ultralytics import YOLO

NOME_EXP = "52_class_error_analysis_nano"
DIR_EXP, CAMINHO_LOG, REGISTRADOR, METADADOS_EXECUCAO = configurar_log_experimento(NOME_EXP, __file__)

RAIZ = Path(__file__).resolve().parent.parent
DIR_TESTE = RAIZ / "datasets_yolo26_v2"
CAMINHO_MODELO = RAIZ / "runs_v2_nano" / "microplastic_yolo26_nano_ft" / "weights" / "best.pt"
NOMES_CLASSES = ["PE", "PET", "PP", "PS"]

TAMANHO_IMG = 1024
LIMIAR_CONF = 0.15
LIMIAR_IOU = 0.5
MARGEM_RECORTE_PX = 10
N_GALERIA = 6
N_BOOTSTRAP = 10_000
SEMENTE = 42
NANO_MAX_NM = 1000.0

AMPL_X = re.compile(r"(\d+)\s*[xX]")
AMPL_TRACO = re.compile(r"^\D*(\d+)-\d+")

aplicar_estilo()


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


def carregar_gt_nano(caminho_rotulo: Path, W: int, H: int, um_por_px: float):
    """PT: Como 42.carregar_gt, mas retorna só caixas NANO (< 1000 nm); caixas micro ficam fora
        do escopo (não contam como GT no pareamento nem entram nos totais).
    EN: Like 42.carregar_gt, but returns only NANO boxes (< 1000 nm); micro boxes are out of
        scope (not counted as GT for matching nor included in totals).
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
        tamanho_px = max(w * W, h * H)
        tamanho_nm = tamanho_px * um_por_px * 1000.0
        if tamanho_nm >= NANO_MAX_NM:
            continue
        xyxy = yolo_para_xyxy_px(cx, cy, w, h, W, H)
        area_normalizada = w * h
        saida.append({"cls_id": id_cls, "xyxy": xyxy, "area_norm": area_normalizada})
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
    gt_pareado, pred_pareada = {}, {}
    for iou, gi, pi in candidatas:
        if gi in gt_pareado or pi in pred_pareada:
            continue
        gt_pareado[gi] = pi
        pred_pareada[pi] = gi
    return gt_pareado, pred_pareada


def recortar_com_margem(imagem, xyxy, W, H, margem=MARGEM_RECORTE_PX):
    x1, y1, x2, y2 = xyxy
    x1 = max(0, x1 - margem)
    y1 = max(0, y1 - margem)
    x2 = min(W, x2 + margem)
    y2 = min(H, y2 + margem)
    return imagem.crop((x1, y1, x2, y2))


def bootstrap_dif_mediana_por_imagem(registros_a, registros_b, semente=SEMENTE, n_bootstrap=N_BOOTSTRAP):
    df_a = pd.DataFrame(registros_a, columns=["source_image", "value"])
    df_b = pd.DataFrame(registros_b, columns=["source_image", "value"])
    imgs = sorted(set(df_a["source_image"]) | set(df_b["source_image"]))
    grupos_a = {s: g["value"].values for s, g in df_a.groupby("source_image")}
    grupos_b = {s: g["value"].values for s, g in df_b.groupby("source_image")}
    gerador = np.random.default_rng(semente)
    diferencas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        amostrados = gerador.choice(imgs, size=len(imgs), replace=True)
        vals_a = np.concatenate([grupos_a[s] for s in amostrados if s in grupos_a]) if any(s in grupos_a for s in amostrados) else np.array([])
        vals_b = np.concatenate([grupos_b[s] for s in amostrados if s in grupos_b]) if any(s in grupos_b for s in amostrados) else np.array([])
        diferencas[i] = (np.median(vals_b) if len(vals_b) else np.nan) - (np.median(vals_a) if len(vals_a) else np.nan)
    ci = np.nanpercentile(diferencas, [2.5, 97.5])
    ponto = (np.median(df_b["value"]) if len(df_b) else float("nan")) - (np.median(df_a["value"]) if len(df_a) else float("nan"))
    return {
        "diff_median_point": float(ponto),
        "ci95": [float(ci[0]), float(ci[1])],
        "excludes_zero": bool(ci[0] > 0 or ci[1] < 0),
    }


def principal():
    if not CAMINHO_MODELO.exists():
        REGISTRADOR.error("checkpoint nao existe: %s -- rode 48_finetune_yolo_nano_only.py primeiro", CAMINHO_MODELO)
        print(f"ERRO: {CAMINHO_MODELO} nao existe -- rode 48_finetune_yolo_nano_only.py primeiro.")
        sys.exit(1)

    REGISTRADOR.info("=== analise de erro por classe (nano-only): PE falso positivo, PS falso negativo ===")
    modelo = YOLO(str(CAMINHO_MODELO))

    caminhos_img = sorted((DIR_TESTE / "images/test").glob("*.png"))

    linhas_pe_fp = []
    linhas_ps = []
    candidatos_galeria_pe_fp = []
    candidatos_galeria_ps_fn = []
    n_ampl_nao_resolvida = 0

    for caminho_img in caminhos_img:
        ampl = interpretar_ampliacao(caminho_img.stem)
        if ampl is None:
            n_ampl_nao_resolvida += 1
            continue
        um_por_px = 100.0 / ampl

        imagem = Image.open(caminho_img).convert("RGB")
        W, H = imagem.size
        caminho_rotulo = DIR_TESTE / "labels/test" / f"{caminho_img.stem}.txt"
        gts = carregar_gt_nano(caminho_rotulo, W, H, um_por_px)

        pred = modelo.predict(source=str(caminho_img), imgsz=TAMANHO_IMG, conf=LIMIAR_CONF, verbose=False)[0]
        preds = []
        if pred.boxes is not None and len(pred.boxes):
            xyxy_px = pred.boxes.xyxy.cpu().numpy()
            ids_cls = pred.boxes.cls.cpu().numpy().astype(int)
            confs = pred.boxes.conf.cpu().numpy()
            for caixa, c, conf in zip(xyxy_px, ids_cls, confs):
                x1, y1, x2, y2 = caixa
                area_normalizada = ((x2 - x1) / W) * ((y2 - y1) / H)
                preds.append({"cls_id": int(c), "xyxy": tuple(caixa), "conf": float(conf), "area_norm": area_normalizada})

        gt_pareado, pred_pareada = parear_mesma_classe(gts, preds)

        for pi, p in enumerate(preds):
            if p["cls_id"] != NOMES_CLASSES.index("PE"):
                continue
            if pi in pred_pareada:
                continue
            melhor_iou, melhor_cls = 0.0, None
            for g in gts:
                if g["cls_id"] == p["cls_id"]:
                    continue
                iou = iou_xyxy(p["xyxy"], g["xyxy"])
                if iou > melhor_iou:
                    melhor_iou, melhor_cls = iou, NOMES_CLASSES[g["cls_id"]]
            linhas_pe_fp.append({
                "source_image": caminho_img.name, "confidence": p["conf"],
                "box_area_norm": p["area_norm"], "best_iou_other_class_gt": melhor_iou,
                "overlapping_class": melhor_cls,
            })
            candidatos_galeria_pe_fp.append((p["conf"], caminho_img, p["xyxy"], W, H))

        for gi, g in enumerate(gts):
            if g["cls_id"] != NOMES_CLASSES.index("PS"):
                continue
            pareado = gi in gt_pareado
            alguma_sobreposicao_pred = False
            if not pareado:
                for p in preds:
                    if iou_xyxy(g["xyxy"], p["xyxy"]) > 0:
                        alguma_sobreposicao_pred = True
                        break
            linhas_ps.append({
                "source_image": caminho_img.name, "status": "TP" if pareado else "FN",
                "box_area_norm": g["area_norm"],
                "any_pred_overlap": alguma_sobreposicao_pred if not pareado else None,
            })
            if not pareado:
                candidatos_galeria_ps_fn.append((g["area_norm"], caminho_img, g["xyxy"], W, H))

        REGISTRADOR.info("imagem=%s n_gt_nano=%d n_pred=%d", caminho_img.name, len(gts), len(preds))

    df_pe_fp = pd.DataFrame(linhas_pe_fp)
    df_ps = pd.DataFrame(linhas_ps)

    dir_relatorios = DIR_EXP / "reports"
    dir_relatorios.mkdir(parents=True, exist_ok=True)
    csv_pe_fp = dir_relatorios / "pe_false_positives_nano.csv"
    csv_ps = dir_relatorios / "ps_false_negatives_vs_matched_nano.csv"
    df_pe_fp.to_csv(csv_pe_fp, index=False)
    df_ps.to_csv(csv_ps, index=False)

    n_pe_fp = len(df_pe_fp)
    frac_pe_fp_sobrepoe_outra = float((df_pe_fp["best_iou_other_class_gt"] > 0).mean()) if n_pe_fp else float("nan")
    print(f"\n=== PE falsos positivos, nano-only (n={n_pe_fp}) ===")
    print(f"fracao que sobrepoe (IoU>0) GT nano de outra classe: {frac_pe_fp_sobrepoe_outra:.3f}")
    if n_pe_fp:
        print("classe mais frequente sobreposta:")
        print(df_pe_fp.loc[df_pe_fp['best_iou_other_class_gt'] > 0, 'overlapping_class'].value_counts())

    n_ps_fn = int((df_ps["status"] == "FN").sum())
    n_ps_tp = int((df_ps["status"] == "TP").sum())
    frac_ps_fn_sem_proposta = float((df_ps.loc[df_ps["status"] == "FN", "any_pred_overlap"] == False).mean()) if n_ps_fn else float("nan")
    print(f"\n=== PS falsos negativos, nano-only (n={n_ps_fn}) vs. verdadeiros positivos (n={n_ps_tp}) ===")
    print(f"fracao de FN sem NENHUMA proposta sobrepondo (qualquer classe): {frac_ps_fn_sem_proposta:.3f}")
    if n_ps_fn and n_ps_tp:
        print(f"area_norm mediana -- FN: {df_ps.loc[df_ps['status']=='FN','box_area_norm'].median():.6f}  "
              f"TP: {df_ps.loc[df_ps['status']=='TP','box_area_norm'].median():.6f}")

    registros_ps_fn = list(df_ps.loc[df_ps["status"] == "FN", ["source_image", "box_area_norm"]].itertuples(index=False, name=None))
    registros_ps_tp = list(df_ps.loc[df_ps["status"] == "TP", ["source_image", "box_area_norm"]].itertuples(index=False, name=None))
    bootstrap_area_ps = bootstrap_dif_mediana_por_imagem(registros_ps_tp, registros_ps_fn) if (registros_ps_fn and registros_ps_tp) else None
    if bootstrap_area_ps:
        print(f"\nbootstrap (exploratorio) diff mediana area_norm FN-TP: {bootstrap_area_ps['diff_median_point']:+.6f} "
              f"IC95%={bootstrap_area_ps['ci95']} exclui_zero={'sim' if bootstrap_area_ps['excludes_zero'] else 'nao'}")

    candidatos_galeria_pe_fp.sort(key=lambda x: x[0], reverse=True)
    candidatos_galeria_ps_fn.sort(key=lambda x: x[0], reverse=True)
    pe_topo = candidatos_galeria_pe_fp[:N_GALERIA]
    ps_topo = candidatos_galeria_ps_fn[:N_GALERIA]

    fig, axes = plt.subplots(2, N_GALERIA, figsize=(2.2 * N_GALERIA, 4.6))
    for coluna in range(N_GALERIA):
        ax = axes[0, coluna]
        if coluna < len(pe_topo):
            conf, caminho_img, xyxy, W, H = pe_topo[coluna]
            imagem = Image.open(caminho_img).convert("RGB")
            recorte = recortar_com_margem(imagem, xyxy, W, H)
            ax.imshow(np.asarray(recorte))
            ax.set_title(f"conf={conf:.2f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        if coluna == 0:
            ax.set_ylabel("PE falso positivo\n(maior confiança)", fontsize=9)

        ax = axes[1, coluna]
        if coluna < len(ps_topo):
            area, caminho_img, xyxy, W, H = ps_topo[coluna]
            imagem = Image.open(caminho_img).convert("RGB")
            recorte = recortar_com_margem(imagem, xyxy, W, H)
            ax.imshow(np.asarray(recorte))
            ax.set_title(f"area={area:.4f}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        if coluna == 0:
            ax.set_ylabel("PS falso negativo\n(maior área)", fontsize=9)

    fig.suptitle("Galeria de inspeção -- nano-only: PE falso positivo / PS falso negativo (exploratório)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    caminho_galeria = dir_relatorios / "pe_fp_ps_fn_gallery_nano.png"
    fig.savefig(caminho_galeria, bbox_inches="tight")
    plt.close(fig)

    salvar_resumo_experimento(NOME_EXP, METADADOS_EXECUCAO, {
        "note": "analise exploratoria, detector nano-only -- gera hipotese, nao e resultado confirmatorio",
        "model_path": str(CAMINHO_MODELO),
        "nano_max_nm": NANO_MAX_NM,
        "n_unresolved_magnification": n_ampl_nao_resolvida,
        "n_pe_false_positives": n_pe_fp,
        "frac_pe_fp_overlapping_other_class_gt": frac_pe_fp_sobrepoe_outra,
        "n_ps_false_negatives": n_ps_fn,
        "n_ps_true_positives": n_ps_tp,
        "frac_ps_fn_with_no_prediction_overlap": frac_ps_fn_sem_proposta,
        "ps_area_norm_median_fn": float(df_ps.loc[df_ps["status"] == "FN", "box_area_norm"].median()) if n_ps_fn else None,
        "ps_area_norm_median_tp": float(df_ps.loc[df_ps["status"] == "TP", "box_area_norm"].median()) if n_ps_tp else None,
        "bootstrap_ps_area_norm_median_diff_fn_vs_tp": bootstrap_area_ps,
        "iou_threshold": LIMIAR_IOU,
        "conf_threshold": LIMIAR_CONF,
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEMENTE,
    })
    print(f"\nCSVs: {csv_pe_fp}, {csv_ps}")
    print(f"Galeria: {caminho_galeria}")
    print(f"Log completo: {CAMINHO_LOG}")


if __name__ == "__main__":
    principal()
