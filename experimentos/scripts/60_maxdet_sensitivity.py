"""Sensibilidade ao max_det / max_det sensitivity.

PT: Mede o efeito do limite de detecções por imagem (max_det) sobre o recall do detector no
    conjunto de teste.
    O treino usa max_det=2000, mas os scripts de avaliação (14, 20, 23, 42) não passam
    `max_det` para `model.predict()`, então rodam com o padrão do Ultralytics, 300
    (ultralytics/cfg/default.yaml).
    Duas micrografias do teste param exatamente em 300 predições: PP_PP-1000x1 (1.268
    referências) e PS_PS-100x (957), as duas mais densas, que somam 2.225 das 4.610
    partículas do teste. Nelas o recall máximo alcançável é 23,7% e 31,3%, independentemente
    do modelo.
    O limite pode ser alterado: em modelos end2end o preditor propaga o valor para a cabeça
    (ultralytics/engine/predictor.py:426, `set_head_attr(max_det=max(args.max_det, 300))`) e
    o top-k usa k = min(max_det, n_anchors); 300 é o padrão, não um teto da arquitetura.
    Usa o mesmo checkpoint e o mesmo pareamento de 14_yolo_test_eval.py (IoU >= 0,5, guloso,
    mesma classe) e de 20_predicted_box_and_end_to_end.py (pareamento espacial, sem classe),
    variando só o max_det.
    Checagem: com max_det=300 deve reproduzir o recall global de 0,3358 e a taxa D de 0,3544
    da avaliação original; se não reproduzir, o restante não é válido.
    Uso:
    python3 60_maxdet_sensitivity.py
EN: Measures the effect of the per-image detection limit (max_det) on detector recall on the
    test set.
    Training uses max_det=2000, but the evaluation scripts (14, 20, 23, 42) do not pass
    `max_det` to `model.predict()`, so they run with the Ultralytics default, 300
    (ultralytics/cfg/default.yaml).
    Two test micrographs stop at exactly 300 predictions: PP_PP-1000x1 (1,268 references)
    and PS_PS-100x (957), the two densest, which together hold 2,225 of the 4,610 test
    particles. On them the maximum attainable recall is 23.7% and 31.3%, regardless of the
    model.
    The limit can be changed: for end2end models the predictor propagates the value to the
    head (ultralytics/engine/predictor.py:426, `set_head_attr(max_det=max(args.max_det,
    300))`) and top-k uses k = min(max_det, n_anchors); 300 is the default, not an
    architectural cap.
    Uses the same checkpoint and matching as 14_yolo_test_eval.py (IoU >= 0.5, greedy, same
    class) and 20_predicted_box_and_end_to_end.py (spatial, class-agnostic matching),
    varying only max_det.
    Check: with max_det=300 it must reproduce the overall recall of 0.3358 and the D rate of
    0.3544 from the original evaluation; otherwise the rest is not valid.
    Usage:
    python3 60_maxdet_sensitivity.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import salvar_resumo_experimento, configurar_log_experimento

from ultralytics import YOLO

NOME_EXP = "60_maxdet_sensitivity"
DIR_EXP, CAMINHO_LOG, REGISTRADOR, METADADOS_EXECUCAO = configurar_log_experimento(NOME_EXP, __file__)

RAIZ = Path(__file__).resolve().parent.parent
DIR_TESTE = RAIZ / "datasets_yolo26_v2"
CAMINHO_MODELO = RAIZ / "runs_v2" / "microplastic_yolo26_v2" / "weights" / "best.pt"
NOMES_CLASSES = ["PE", "PET", "PP", "PS"]

TAMANHO_IMG = 1024
LIMIAR_CONF = 0.15
LIMIAR_IOU = 0.5
VALORES_MAX_DET = [300, 2000]

# PT: valores da avaliação original, com o max_det padrão de 300
# EN: values from the original evaluation, with the default max_det of 300
REFERENCIA = {"global_recall": 0.33579, "d_rate": 0.35445}


def iou_xyxy(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    intersecao = iw * ih
    if intersecao <= 0:
        return 0.0
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - intersecao
    return intersecao / ua if ua > 0 else 0.0


def carregar_gt(caminho_rotulo: Path):
    gts = []
    if not caminho_rotulo.exists():
        return gts
    for linha in caminho_rotulo.read_text().splitlines():
        p = linha.split()
        if len(p) < 5:
            continue
        c = int(p[0])
        cx, cy, w, h = (float(v) for v in p[1:5])
        gts.append((c, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
    return gts


def par(gts, preds, com_classe: bool):
    """PT: Pareamento guloso por IoU decrescente; cada referência é usada uma vez.
    EN: Greedy matching by decreasing IoU; each reference is used once.
    """
    cands = []
    for gi, (cls_g, caixa_g) in enumerate(gts):
        for pi, (cls_p, caixa_p) in enumerate(preds):
            if com_classe and cls_p != cls_g:
                continue
            iou = iou_xyxy(caixa_g, caixa_p)
            if iou >= LIMIAR_IOU:
                cands.append((iou, gi, pi))
    cands.sort(reverse=True)
    mg, mp = set(), set()
    for _, gi, pi in cands:
        if gi in mg or pi in mp:
            continue
        mg.add(gi)
        mp.add(pi)
    return mg


def executar(max_det: int):
    modelo = YOLO(str(CAMINHO_MODELO))
    dir_img = DIR_TESTE / "images" / "test"
    dir_rot = DIR_TESTE / "labels" / "test"

    por_imagem, total_gt_aux = [], 0
    n_com_classe = n_sem_classe = 0
    for img in sorted(dir_img.iterdir()):
        if img.suffix.lower() not in (".png", ".jpg", ".jpeg", ".tif"):
            continue
        gts = carregar_gt(dir_rot / f"{img.stem}.txt")
        if not gts:
            continue
        pred = modelo.predict(source=str(img), imgsz=TAMANHO_IMG, conf=LIMIAR_CONF,
                             max_det=max_det, verbose=False)[0]
        preds = []
        if pred.boxes is not None and len(pred.boxes):
            for caixa, c in zip(pred.boxes.xyxyn.cpu().numpy(),
                              pred.boxes.cls.cpu().numpy().astype(int)):
                preds.append((int(c), tuple(caixa)))
        ma = par(gts, preds, com_classe=True)
        mg = par(gts, preds, com_classe=False)
        n_com_classe += len(ma)
        n_sem_classe += len(mg)
        total_gt_aux += len(gts)
        por_imagem.append({"image": img.name, "n_gt": len(gts), "n_pred": len(preds),
                          "matched_class_aware": len(ma), "matched_spatial": len(mg),
                          "truncated": len(preds) >= max_det})
    return {"max_det": max_det, "n_gt": total_gt_aux,
            "global_recall": n_com_classe / total_gt_aux, "d_rate": n_sem_classe / total_gt_aux,
            "per_image": por_imagem}


def principal():
    resultados = {}
    for md in VALORES_MAX_DET:
        REGISTRADOR.info("avaliando com max_det=%d", md)
        r = executar(md)
        resultados[md] = r
        truncado = [p["image"] for p in r["per_image"] if p["truncated"]]
        print(f"\nmax_det={md:5d}  recall global={r['global_recall']:.4f}  "
              f"taxa D={r['d_rate']:.4f}  predicoes totais="
              f"{sum(p['n_pred'] for p in r['per_image']):5d}  truncadas={len(truncado)}")
        for t in truncado:
            print(f"               truncada: {t}")

    base = resultados[300]
    ok = (abs(base["global_recall"] - REFERENCIA["global_recall"]) < 1e-3 and
          abs(base["d_rate"] - REFERENCIA["d_rate"]) < 1e-3)
    print("\n" + "=" * 70)
    print("SANIDADE -- max_det=300 reproduz a referência?", "SIM" if ok else "NAO")
    print(f"   recall global  referência={REFERENCIA['global_recall']:.4f}  obtido={base['global_recall']:.4f}")
    print(f"   taxa D         referência={REFERENCIA['d_rate']:.4f}  obtido={base['d_rate']:.4f}")
    if not ok:
        print("\n   !! nao reproduz -- o resultado com max_det=2000 NAO deve ser usado.")
        return

    hi = resultados[2000]
    print("\n" + "=" * 70)
    print("EFEITO DE max_det=2000")
    print(f"   recall global  {base['global_recall']*100:5.2f}%  ->  {hi['global_recall']*100:5.2f}%"
          f"   ({(hi['global_recall']-base['global_recall'])*100:+.2f} pp)")
    print(f"   taxa D         {base['d_rate']*100:5.2f}%  ->  {hi['d_rate']*100:5.2f}%"
          f"   ({(hi['d_rate']-base['d_rate'])*100:+.2f} pp)")
    print("\n   por micrografia truncada em 300:")
    bi = {p["image"]: p for p in base["per_image"]}
    for p in hi["per_image"]:
        b = bi[p["image"]]
        if not b["truncated"]:
            continue
        print(f"   {p['image']:24s} gt={p['n_gt']:5d}  pred {b['n_pred']:4d}->{p['n_pred']:5d}"
              f"  recall {b['matched_spatial']/p['n_gt']*100:5.2f}% -> {p['matched_spatial']/p['n_gt']*100:5.2f}%")

    salvar_resumo_experimento(NOME_EXP, METADADOS_EXECUCAO, {
        "note": "sensibilidade ao max_det. O treino usa max_det=2.000 deteccoes por "
                "imagem, mas os scripts de avaliacao nao passam max_det e rodaram com o padrao "
                "300 do Ultralytics. Duas micrografias do teste travado foram truncadas.",
        "model_path": str(CAMINHO_MODELO),
        "published_with_max_det_300": REFERENCIA,
        "sanity_reproduces_published": ok,
        "conf_threshold": LIMIAR_CONF,
        "iou_threshold": LIMIAR_IOU,
        "results": {str(k): v for k, v in resultados.items()},
    })
    print(f"\nLog completo: {CAMINHO_LOG}")


if __name__ == "__main__":
    principal()
