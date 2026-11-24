"""Reavaliação do detector com max_det explícito / Detector re-evaluation with explicit max_det.

PT: Reavalia o detector no conjunto de teste com um limite explícito de detecções por imagem
    (`max_det`), para a execução única e para as 10 repetições controladas.
    `max_det=2000` está nos 5 scripts de treino (13, 44, 48, 50 e 01_yolo26_main_10x) e fica
    gravado no checkpoint, mas nenhum script de avaliação passa o parâmetro, e o preditor do
    Ultralytics sobrescreve o atributo da cabeça a cada chamada:
    ultralytics/engine/predictor.py:426
    model.set_head_attr(max_det=max(self.args.max_det, 300), ...)
    Sem o parâmetro, `args.max_det` vale o padrão 300 e o 2000 do checkpoint é descartado.
    Duas micrografias do teste param exatamente em 300 predições (PP_PP-1000x1, 1.268
    referências, e PS_PS-100x, 957), as duas mais densas, que somam 2.225 das 4.610
    partículas do teste.
    Reproduz a lógica de 14_yolo_test_eval.py (val() + pareamento guloso por IoU) e de
    23_recall_by_size_regime.py (regime físico nano/micro), variando só o max_det. Também
    calcula a correlação densidade x recall.
    Não cobre a cadeia fim a fim (D.P.S) nem a avaliação pareada, que dependem do SmolVLM e
    do adaptador de tamanho escolhido (regra eq1 ou eq2); ver eval_and_pair.py.
    Checagem: com --max-det 300 a execução única deve reproduzir os valores da avaliação
    original (P=0,6069 R=0,4668 mAP50=0,2802; recall global 0,3358; taxa D 0,3544). O script
    aborta se divergir.
    Uso:
    python3 61_reeval_detector_maxdet.py -- checagem com 300 + 2000
    python3 61_reeval_detector_maxdet.py --max-det 2000 -- só 2000
    python3 61_reeval_detector_maxdet.py --skip-seeds -- só a execução única
EN: Re-evaluates the detector on the test set with an explicit per-image detection limit
    (`max_det`), for the single run and for the 10 controlled repetitions.
    `max_det=2000` is set in the 5 training scripts (13, 44, 48, 50 and 01_yolo26_main_10x)
    and stored in the checkpoint, but no evaluation script passes the parameter, and the
    Ultralytics predictor overwrites the head attribute on every call:
    ultralytics/engine/predictor.py:426
    model.set_head_attr(max_det=max(self.args.max_det, 300), ...)
    Without the parameter, `args.max_det` takes the default 300 and the checkpoint's 2000 is
    discarded. Two test micrographs stop at exactly 300 predictions (PP_PP-1000x1, 1,268
    references, and PS_PS-100x, 957), the two densest, holding 2,225 of the 4,610 test
    particles.
    Reproduces the logic of 14_yolo_test_eval.py (val() + greedy IoU matching) and
    23_recall_by_size_regime.py (physical nano/micro regime), varying only max_det. Also
    computes the density x recall correlation.
    Does not cover the end-to-end chain (D.P.S) or the paired evaluation, which depend on
    SmolVLM and on the chosen size adapter (eq1 or eq2 rule); see eval_and_pair.py.
    Check: with --max-det 300 the single run must reproduce the original evaluation values
    (P=0.6069 R=0.4668 mAP50=0.2802; overall recall 0.3358; D rate 0.3544). The script
    aborts if they diverge.
    Usage:
    python3 61_reeval_detector_maxdet.py -- check with 300 + 2000
    python3 61_reeval_detector_maxdet.py --max-det 2000 -- 2000 only
    python3 61_reeval_detector_maxdet.py --skip-seeds -- single run only
"""
import argparse
import math
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import salvar_resumo_experimento, configurar_log_experimento

from ultralytics import YOLO

NOME_EXP = "61_reeval_detector_maxdet"
DIR_EXP, CAMINHO_LOG, REGISTRADOR, METADADOS_EXECUCAO = configurar_log_experimento(NOME_EXP, __file__)

RAIZ = Path(__file__).resolve().parent.parent
DIR_TESTE = RAIZ / "datasets_yolo26_v2"
EXECUCAO_UNICA = RAIZ / "runs_v2" / "microplastic_yolo26_v2" / "weights" / "best.pt"
EXECUCOES_CTRL = RAIZ.parent / "experimentos_controlados" / "runs" / "01_yolo26_main"
NOMES_CLASSES = ["PE", "PET", "PP", "PS"]

TAMANHO_IMG = 1024
LIMIAR_CONF = 0.15
IOU_VAL = 0.7          # PT: NMS do val(), como em 14_yolo_test_eval.py / EN: val() NMS, as in 14_yolo_test_eval.py
IOU_PAR = 0.5        # PT: pareamento predição x referência / EN: prediction x reference matching
NANO_MAX_NM = 1000.0
SEMENTES = range(10)

REFERENCIA_300 = {"precision": 0.60694, "recall": 0.46683, "map50": 0.28025,
                 "global_recall": 0.33579, "d_rate": 0.35445}

AMPL_X = re.compile(r"(\d+)\s*[xX]")
AMPL_TRACO = re.compile(r"^\D*(\d+)-\d+")


def interpretar_ampliacao(nome: str):
    m = AMPL_X.findall(nome)
    if m:
        return int(m[-1])
    m = AMPL_TRACO.match(nome)
    return int(m.group(1)) if m else None


def yaml_teste() -> Path:
    """PT: data.yaml com caminho absoluto correto (o do repositório aponta para outro local).
    EN: data.yaml with the correct absolute path (the repository one points elsewhere).
    """
    p = Path(tempfile.mkdtemp()) / "data_test.yaml"
    p.write_text(
        f"path: {DIR_TESTE}\ntrain: images/test\nval: images/test\n"
        f"nc: {len(NOMES_CLASSES)}\nnames:\n" + "\n".join(f"  - {c}" for c in NOMES_CLASSES) + "\n"
    )
    return p


def iou_xyxy(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    intersecao = iw * ih
    if intersecao <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersecao
    return intersecao / ua if ua > 0 else 0.0


def carregar_gt(caminho_rotulo: Path, ampl):
    """PT: Referências com classe, caixa normalizada e regime físico.
    EN: References with class, normalized box and physical regime.
    """
    saida = []
    if not caminho_rotulo.exists():
        return saida
    for linha in caminho_rotulo.read_text().splitlines():
        p = linha.split()
        if len(p) < 5:
            continue
        id_cls = int(p[0])
        cx, cy, w, h = (float(v) for v in p[1:5])
        caixa = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        regime = None
        if ampl:
            tamanho_nm = max(w * 1280, h * 960) * (100.0 / ampl) * 1000.0
            regime = "NANOPLASTIC" if tamanho_nm < NANO_MAX_NM else "MICROPLASTIC"
        saida.append({"cls_id": id_cls, "box": caixa, "regime": regime})
    return saida


def pareamento_guloso(gts, preds, com_classe: bool):
    cands = []
    for gi, g in enumerate(gts):
        for pi, p in enumerate(preds):
            if com_classe and p["cls_id"] != g["cls_id"]:
                continue
            iou = iou_xyxy(g["box"], p["box"])
            if iou >= IOU_PAR:
                cands.append((iou, gi, pi))
    cands.sort(reverse=True)
    mg, mp = set(), set()
    for _, gi, pi in cands:
        if gi in mg or pi in mp:
            continue
        mg.add(gi)
        mp.add(pi)
    return mg


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    r = sxy / math.sqrt(sxx * syy)
    t = r * math.sqrt((n - 2) / (1 - r * r))
    return r, t, n


def avaliar(ckpt: Path, max_det: int, caminho_yaml: Path):
    """PT: val() + pareamento por partícula, com max_det explícito.
    EN: val() + per-particle matching, with explicit max_det.
    """
    modelo = YOLO(str(ckpt))
    m = modelo.val(data=str(caminho_yaml), split="val", imgsz=TAMANHO_IMG,
                  conf=LIMIAR_CONF, iou=IOU_VAL, max_det=max_det,
                  plots=False, verbose=False)
    agregado = {"precision": float(m.box.mp), "recall": float(m.box.mr),
           "map50": float(m.box.map50), "map50_95": float(m.box.map)}
    dados_por_classe = {}
    for i, nome in enumerate(NOMES_CLASSES):
        try:
            p, r, ap50, ap = m.box.class_result(i)
            dados_por_classe[nome] = {"precision": float(p), "recall": float(r),
                               "ap50": float(ap50), "ap50_95": float(ap)}
        except Exception:
            pass

    dir_img, dir_rot = DIR_TESTE / "images" / "test", DIR_TESTE / "labels" / "test"
    por_imagem, regime = [], {"NANOPLASTIC": [0, 0], "MICROPLASTIC": [0, 0]}
    n_com_classe = n_espacial = n_gt = 0
    for img in sorted(dir_img.iterdir()):
        if img.suffix.lower() not in (".png", ".jpg", ".jpeg", ".tif"):
            continue
        ampl = interpretar_ampliacao(img.name)
        gts = carregar_gt(dir_rot / f"{img.stem}.txt", ampl)
        if not gts:
            continue
        pred = modelo.predict(source=str(img), imgsz=TAMANHO_IMG, conf=LIMIAR_CONF,
                             max_det=max_det, verbose=False)[0]
        preds = []
        if pred.boxes is not None and len(pred.boxes):
            for caixa, c in zip(pred.boxes.xyxyn.cpu().numpy(),
                              pred.boxes.cls.cpu().numpy().astype(int)):
                preds.append({"cls_id": int(c), "box": tuple(caixa)})
        ma = pareamento_guloso(gts, preds, com_classe=True)
        ms = pareamento_guloso(gts, preds, com_classe=False)
        for gi, g in enumerate(gts):
            if g["regime"]:
                regime[g["regime"]][1] += 1
                if gi in ms:
                    regime[g["regime"]][0] += 1
        n_com_classe += len(ma); n_espacial += len(ms); n_gt += len(gts)
        por_imagem.append({"image": img.name, "n_gt": len(gts), "n_pred": len(preds),
                          "matched_class_aware": len(ma), "matched_spatial": len(ms),
                          "truncated": len(preds) >= max_det})

    contagens_gt = [p["n_gt"] for p in por_imagem]
    recalls = [p["matched_spatial"] / p["n_gt"] for p in por_imagem]
    r, t, n = pearson(contagens_gt, recalls)
    return {"max_det": max_det, "aggregate_metrics": agregado, "per_class_metrics": dados_por_classe,
            "n_gt": n_gt, "global_recall": n_com_classe / n_gt, "d_rate": n_espacial / n_gt,
            "regime_recall": {k: {"gt": v[1], "matched": v[0],
                                  "recall": v[0] / v[1] if v[1] else None}
                              for k, v in regime.items()},
            "density_correlation": {"pearson_r": r, "t": t, "n_images": n},
            "n_truncated_images": sum(p["truncated"] for p in por_imagem),
            "per_image": por_imagem}


def resumir(rotulo, res):
    a = res["aggregate_metrics"]
    print(f"\n[{rotulo}] max_det={res['max_det']}")
    print(f"   val():  P={a['precision']:.4f} R={a['recall']:.4f} "
          f"mAP50={a['map50']:.4f} mAP50-95={a['map50_95']:.4f}")
    print(f"   recall global={res['global_recall']:.4f}  taxa D={res['d_rate']:.4f}  "
          f"truncadas={res['n_truncated_images']}")
    rg = res["regime_recall"]
    print(f"   nano={rg['NANOPLASTIC']['recall']*100:.2f}% ({rg['NANOPLASTIC']['gt']})  "
          f"micro={rg['MICROPLASTIC']['recall']*100:.2f}% ({rg['MICROPLASTIC']['gt']})")
    d = res["density_correlation"]
    print(f"   densidade x recall: r={d['pearson_r']:+.4f} t={d['t']:+.3f} n={d['n_images']}")


def principal():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-det", type=int, default=2000)
    ap.add_argument("--skip-sanity", action="store_true",
                    help="pula a checagem com max_det=300 (nao recomendado)")
    ap.add_argument("--skip-seeds", action="store_true",
                    help="avalia so o single run, sem as 10 repeticoes")
    args = ap.parse_args()

    caminho_yaml = yaml_teste()
    saida = {}

    if not args.skip_sanity:
        base = avaliar(EXECUCAO_UNICA, 300, caminho_yaml)
        resumir("single run, SANIDADE", base)
        saida["single_run_maxdet300"] = base
        desvio = {k: abs(base.get(k, base["aggregate_metrics"].get(k, 0)) - v)
                 if k in ("global_recall", "d_rate")
                 else abs(base["aggregate_metrics"][k] - v)
                 for k, v in REFERENCIA_300.items()}
        ok = all(d < 1e-3 for d in desvio.values())
        print(f"\n   reproduz a referência? {'SIM' if ok else 'NAO'}")
        for k, v in REFERENCIA_300.items():
            print(f"      {k:14s} referência={v:.5f}  dif={desvio[k]:.5f}")
        if not ok:
            print("\n   !! divergencia -- abortando, o resultado com max_det novo nao valeria.")
            return

    unico = avaliar(EXECUCAO_UNICA, args.max_det, caminho_yaml)
    resumir("single run", unico)
    saida[f"single_run_maxdet{args.max_det}"] = unico

    resultados_sementes = {}
    if not args.skip_seeds:
        for s in SEMENTES:
            ck = EXECUCOES_CTRL / f"rep_{s:02d}" / "weights" / "best.pt"
            if not ck.exists():
                REGISTRADOR.warning("checkpoint ausente: %s", ck)
                continue
            r = avaliar(ck, args.max_det, caminho_yaml)
            resultados_sementes[s] = r
            print(f"   seed {s}: P={r['aggregate_metrics']['precision']:.4f} "
                  f"R={r['aggregate_metrics']['recall']:.4f} "
                  f"mAP50={r['aggregate_metrics']['map50']:.4f} D={r['d_rate']:.4f}")
        if len(resultados_sementes) > 1:
            print(f"\n=== {len(resultados_sementes)} repeticoes, max_det={args.max_det} "
                  f"(média, desvio e IC 95%) ===")
            for chave in ("precision", "recall", "map50", "map50_95"):
                v = np.array([r["aggregate_metrics"][chave] for r in resultados_sementes.values()]) * 100
                h = 2.2621571627409915 * v.std(ddof=1) / math.sqrt(len(v))
                print(f"   {chave:10s} {v.mean():6.2f} +/- {v.std(ddof=1):5.2f}  "
                      f"[{v.mean()-h:6.2f}, {v.mean()+h:6.2f}]")
        saida[f"controlled_seeds_maxdet{args.max_det}"] = {str(k): v for k, v in resultados_sementes.items()}

    salvar_resumo_experimento(NOME_EXP, METADADOS_EXECUCAO, {
        "note": "reavaliacao do detector com max_det explicito. Os scripts originais de "
                "avaliacao nao passam max_det e rodaram com o padrao 300 do Ultralytics, "
                "apesar de max_det=2000 estar nos scripts de treino e gravado no checkpoint "
                "(descartado por predictor.py:426). Nao cobre a cadeia fim-a-fim D.P.S.",
        "conf_threshold": LIMIAR_CONF,
        "val_nms_iou": IOU_VAL,
        "match_iou": IOU_PAR,
        "published_with_max_det_300": REFERENCIA_300,
        **saida,
    })
    print(f"\nLog completo: {CAMINHO_LOG}")


if __name__ == "__main__":
    principal()
