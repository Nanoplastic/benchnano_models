"""Reavaliação das dobras de validação cruzada com max_det=2000 / Cross-validation fold re-evaluation with max_det=2000.

PT: Reavalia os 10 checkpoints de validação cruzada (5 dobras MISTAS + 5 dobras só NANO) com
    max_det explícito. Os scripts originais (45/51/58) não passavam max_det, então valia o
    padrão 300 do Ultralytics, embora o checkpoint de treino use 2.000.
    A validação cruzada cobre as 102 micrografias utilizáveis, então há micrografias densas
    além de PP_PP-1000x1 e PS_PS-100x truncando em 300: 15 das ~102 imagens held-out (soma
    das 5 dobras mistas) e 2 das ~100 (soma das 5 dobras nano) atingem o teto, algumas com
    quase 3x mais predições a max_det=2000.
    Gera o recall agregado e por regime da ablação só com nanopartículas contra o treino
    misto, por polímero.
    Checagem (max_det=300 deve reproduzir):
    só nano: overall=6,19% NANO=15,01% (n=9399) MICRO=1,11% (n=16331)
    misto: overall=23,78% NANO=21,03% (n=9399) MICRO=25,36% (n=16331)
    Grava em resultados/77_reeval_cv_folds_maxdet2000/, sem sobrescrever
    51_eval_cv_folds_nano nem 58_eval_cv_folds_mixed_by_regime.
    Uso:
    python3 -u 77_reeval_cv_folds_maxdet2000.py
EN: Re-evaluates the 10 cross-validation checkpoints (5 MIXED folds + 5 NANO-only folds)
    with explicit max_det. The original scripts (45/51/58) did not pass max_det, so the
    Ultralytics default 300 applied, although the training checkpoint uses 2,000.
    Cross-validation covers the 102 usable micrographs, so dense micrographs other than
    PP_PP-1000x1 and PS_PS-100x also truncate at 300: 15 of the ~102 held-out images (sum
    over the 5 mixed folds) and 2 of the ~100 (sum over the 5 nano folds) hit the cap, some
    with almost 3x more predictions at max_det=2000.
    Produces the overall and per-regime recall of the nano-only ablation against mixed
    training, per polymer.
    Check (max_det=300 must reproduce):
    nano only: overall=6.19% NANO=15.01% (n=9399) MICRO=1.11% (n=16331)
    mixed: overall=23.78% NANO=21.03% (n=9399) MICRO=25.36% (n=16331)
    Writes to resultados/77_reeval_cv_folds_maxdet2000/, without overwriting
    51_eval_cv_folds_nano or 58_eval_cv_folds_mixed_by_regime.
    Usage:
    python3 -u 77_reeval_cv_folds_maxdet2000.py
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

NOME_EXP = "77_reeval_cv_folds_maxdet2000"
DIR_EXP, CAMINHO_LOG, REGISTRADOR, METADADOS_EXECUCAO = configurar_log_experimento(NOME_EXP, __file__)

RAIZ = Path(__file__).resolve().parent.parent
K_DOBRAS = 5
TAMANHO_IMG = 1024
LIMIAR_CONF = 0.15
LIMIAR_IOU = 0.5
N_BOOTSTRAP = 10_000
SEMENTE = 42
NANO_MAX_NM = 1000.0
MAX_DETS_TESTADOS = (300, 2000)

CONDICOES = {
    "nano_only": {
        "runs_dir": RAIZ / "runs_v2_cv_nano",
        "data_dir": RAIZ / "datasets_yolo26_v2_cv_nano",
        "published": {"overall": 0.061912164788184995,
                      "NANOPLASTIC": (0.1501223534418555, 9399),
                      "MICROPLASTIC": (0.011144449207029576, 16331)},
    },
    "mixed": {
        "runs_dir": RAIZ / "runs_v2_cv",
        "data_dir": RAIZ / "datasets_yolo26_v2_cv",
        "published": {"overall": 0.23777691410804508,
                      "NANOPLASTIC": (0.2103415256942228, 9399),
                      "MICROPLASTIC": (0.25356683607862346, 16331)},
    },
}
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


def yolo_para_xyxy(cx, cy, w, h, W, H):
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


def carregar_gt(caminho_rotulo: Path, W: int, H: int, um_por_px):
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
        regime = "NANOPLASTIC" if tamanho_nm < NANO_MAX_NM else "MICROPLASTIC"
        saida.append({"cls_id": id_cls, "xyxy": yolo_para_xyxy(cx, cy, w, h, W, H), "size": regime})
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
    gt_pareado, pred_pareada = set(), set()
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
    return {"n": len(sub), "n_images": len(imgs), "recall_point": float(df["matched"].mean()),
            "ci95": [float(ci[0]), float(ci[1])]}


def executar_condicao(chave_cond, max_det):
    cfg = CONDICOES[chave_cond]
    dir_execucoes, dir_dados = cfg["runs_dir"], cfg["data_dir"]

    todos_registros = []
    por_dobra = []
    for k in range(K_DOBRAS):
        ckpt = dir_execucoes / f"fold{k}" / "weights" / "best.pt"
        dir_img = dir_dados / f"fold{k}" / "images" / "test"
        dir_rotulo = dir_dados / f"fold{k}" / "labels" / "test"
        if not ckpt.exists():
            raise SystemExit(f"checkpoint nao existe: {ckpt}")
        modelo = YOLO(str(ckpt))

        caminhos_img = sorted(dir_img.glob("*.png"))
        n_gt_fold = n_pareados_dobra = 0
        for caminho_img in caminhos_img:
            ampl = interpretar_ampliacao(caminho_img.stem)
            if ampl is None:
                continue
            um_por_px = 100.0 / ampl
            with Image.open(caminho_img) as im:
                im = im.convert("RGB")
                W, H = im.size
            gts = carregar_gt(dir_rotulo / f"{caminho_img.stem}.txt", W, H, um_por_px)

            pred = modelo.predict(source=str(caminho_img), imgsz=TAMANHO_IMG, conf=LIMIAR_CONF,
                                 max_det=max_det, verbose=False)[0]
            preds = []
            if pred.boxes is not None and len(pred.boxes):
                xyxy_px = pred.boxes.xyxy.cpu().numpy()
                ids_cls = pred.boxes.cls.cpu().numpy().astype(int)
                for caixa, c in zip(xyxy_px, ids_cls):
                    preds.append({"cls_id": int(c), "xyxy": tuple(caixa)})

            gt_pareado = parear_mesma_classe(gts, preds)
            for gi, g in enumerate(gts):
                pareado = gi in gt_pareado
                todos_registros.append({"source_image": caminho_img.name, "fold": k,
                                    "size": g["size"], "matched": pareado})
                n_gt_fold += 1
                n_pareados_dobra += int(pareado)
            REGISTRADOR.info("cond=%s max_det=%d fold=%d imagem=%s gt=%d matched=%d n_pred=%d",
                        chave_cond, max_det, k, caminho_img.name, len(gts), len(gt_pareado), len(preds))

        por_dobra.append({"fold": k, "n_images": len(caminhos_img), "n_gt": n_gt_fold,
                         "recall": n_pareados_dobra / n_gt_fold if n_gt_fold else None})

    bucket = {}
    for regime in ("NANOPLASTIC", "MICROPLASTIC"):
        bucket[regime] = ic_recall_bootstrap(todos_registros, lambda s, r=regime: s == r)
    geral = ic_recall_bootstrap(todos_registros, lambda s: True)

    # PT: salva os registros por partícula, usados por
    #     78_compare_nano_ablation_maxdet2000.py (mesmo formato de 51/58: source_image,
    #     size, matched)
    # EN: saves the per-particle records, used by 78_compare_nano_ablation_maxdet2000.py
    #     (same format as 51/58: source_image, size, matched)
    csv_saida = DIR_EXP / "reports" / f"cv_pooled_predictions_{chave_cond}_maxdet{max_det}.csv"
    csv_saida.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(todos_registros).to_csv(csv_saida, index=False)

    return {"max_det": max_det, "per_fold": por_dobra, "bucket_results": bucket, "overall": geral,
            "csv_path": str(csv_saida)}


def principal():
    resultados = {}
    for chave_cond in CONDICOES:
        resultados[chave_cond] = {}
        for md in MAX_DETS_TESTADOS:
            print(f"\n{'='*72}\n{chave_cond} -- max_det={md}\n{'='*72}")
            REGISTRADOR.info("iniciando %s max_det=%d", chave_cond, md)
            r = executar_condicao(chave_cond, md)
            resultados[chave_cond][md] = r
            print(f"  overall recall={r['overall']['recall_point']*100:.2f}%  "
                  f"NANO={r['bucket_results']['NANOPLASTIC']['recall_point']*100:.2f}%  "
                  f"MICRO={r['bucket_results']['MICROPLASTIC']['recall_point']*100:.2f}%")

    print(f"\n{'='*72}\nSANIDADE -- max_det=300 reproduz 51 e 58?\n{'='*72}")
    ok = True
    for chave_cond, cfg in CONDICOES.items():
        pub = cfg["published"]
        r300 = resultados[chave_cond][300]
        d_geral = abs(r300["overall"]["recall_point"] - pub["overall"])
        sinalizador = "" if d_geral <= TOL_CHECAGEM else "  <-- DIVERGE"
        if d_geral > TOL_CHECAGEM:
            ok = False
        print(f"  {chave_cond:10s} overall  referência={pub['overall']*100:.2f}%  "
              f"obtido={r300['overall']['recall_point']*100:.2f}%  dif={d_geral*100:.3f}pp{sinalizador}")
        for regime in ("NANOPLASTIC", "MICROPLASTIC"):
            ref_r, ref_n = pub[regime]
            obt = r300["bucket_results"][regime]
            d = abs(obt["recall_point"] - ref_r)
            n_ok = obt["n"] == ref_n
            sinalizador = "" if (d <= TOL_CHECAGEM and n_ok) else "  <-- DIVERGE"
            if d > TOL_CHECAGEM or not n_ok:
                ok = False
            print(f"  {chave_cond:10s} {regime:13s} referência={ref_r*100:.2f}% (n={ref_n})  "
                  f"obtido={obt['recall_point']*100:.2f}% (n={obt['n']})  dif={d*100:.3f}pp{sinalizador}")
    print("\n  ->", "SIM" if ok else "NAO")
    if not ok:
        print("\n  !! nao reproduz -- o resultado de max_det=2000 NAO deve ser usado.")
        return

    print(f"\n{'='*72}\nmax_det=300 (sanidade)  vs  max_det=2000\n{'='*72}")
    for chave_cond in CONDICOES:
        r300, r2000 = resultados[chave_cond][300], resultados[chave_cond][2000]
        print(f"\n  {chave_cond}")
        print(f"    overall   {r300['overall']['recall_point']*100:6.2f}%  ->  {r2000['overall']['recall_point']*100:6.2f}%")
        for regime in ("NANOPLASTIC", "MICROPLASTIC"):
            a, b = r300["bucket_results"][regime], r2000["bucket_results"][regime]
            print(f"    {regime:13s} {a['recall_point']*100:6.2f}% [{a['ci95'][0]*100:.1f},{a['ci95'][1]*100:.1f}]  ->  "
                  f"{b['recall_point']*100:6.2f}% [{b['ci95'][0]*100:.1f},{b['ci95'][1]*100:.1f}]")

    # PT: diferença pareada entre misto e só nano a max_det=2000, por micrografia de origem
    #     (bootstrap sobre imagens, como em 59_compare_nano_ablation_by_label.py)
    # EN: paired difference between mixed and nano-only at max_det=2000, per source
    #     micrograph (bootstrap over images, as in 59_compare_nano_ablation_by_label.py)
    nano_2000 = resultados["nano_only"][2000]["bucket_results"]["NANOPLASTIC"]
    misto_2000 = resultados["mixed"][2000]["bucket_results"]["NANOPLASTIC"]
    print(f"\n  diferenca NANO (misto - nano_only) a max_det=2000: "
          f"{(misto_2000['recall_point']-nano_2000['recall_point'])*100:+.2f} pp")

    salvar_resumo_experimento(NOME_EXP, METADADOS_EXECUCAO, {
        "note": "reavaliacao dos 10 folds de CV (5 mistos + 5 nano-only) com max_det "
                "explicito -- alimenta achado C (ablacao nano-only) na Discussao. "
                "Varias micrografias held-out truncavam em 300 predicoes.",
        "conf_threshold": LIMIAR_CONF, "iou_threshold": LIMIAR_IOU,
        "sanity_reproduces_published": ok,
        "results": {chave_cond: {str(md): r for md, r in by_md.items()} for chave_cond, by_md in resultados.items()},
    })
    print(f"\nLog completo: {CAMINHO_LOG}")


if __name__ == "__main__":
    principal()
