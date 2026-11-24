"""Caixa prevista e fim a fim com max_det=2000 / Predicted box and end-to-end with max_det=2000.

PT: Mesma avaliação de 20_predicted_box_and_end_to_end.py (execução única, mesmos
    checkpoints: runs_v2/microplastic_yolo26_v2/best.pt e checkpoints/size_lora) com uma
    mudança de inferência: max_det. Esta execução gera os resultados de execução única do
    detector, a matriz de confusão de tamanho na condição caixa do YOLO (n=1482) e o funil
    fim a fim; a avaliação pareada 07_end_to_end_paired dos experimentos controlados usa os
    checkpoints de cada semente.
    Roda duas vezes: primeiro com max_det=300 (checagem: deve reproduzir D=35,4% P|D=90,7%
    S|D,P=90,8% E=29,2%, n=1482, do summary.json original), depois com max_det=2000. Só
    grava o resultado de 2000 se a checagem passar.
    Grava em resultados/65_predicted_box_and_end_to_end_maxdet2000/, sem sobrescrever
    20_predicted_box_and_end_to_end.
    Uso:
    python3 -u 65_predicted_box_and_end_to_end_maxdet2000.py
EN: Same evaluation as 20_predicted_box_and_end_to_end.py (single run, same checkpoints:
    runs_v2/microplastic_yolo26_v2/best.pt and checkpoints/size_lora) with one inference
    change: max_det. This run produces the single-run detector results, the size confusion
    matrix for the YOLO box condition (n=1482) and the end-to-end funnel; the paired
    evaluation 07_end_to_end_paired of the controlled experiments uses each seed's
    checkpoints.
    Runs twice: first with max_det=300 (check: must reproduce D=35.4% P|D=90.7% S|D,P=90.8%
    E=29.2%, n=1482, from the original summary.json), then with max_det=2000. Only writes
    the 2000 result if the check passes.
    Writes to resultados/65_predicted_box_and_end_to_end_maxdet2000/, without overwriting
    20_predicted_box_and_end_to_end.
    Usage:
    python3 -u 65_predicted_box_and_end_to_end_maxdet2000.py
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import (
    DIR_CHECKPOINTS, DISPOSITIVO, ID_MODELO, PROMPT_CLASSIFICACAO_TAMANHO, ROTULOS_TAMANHO,
    classificar_imagem, extrair_rotulo, carregar_modelo_base, salvar_resumo_experimento,
    configurar_log_experimento,
)

from ultralytics import YOLO

NOME_EXP = "65_predicted_box_and_end_to_end_maxdet2000"
DIR_EXP, CAMINHO_LOG, REGISTRADOR, METADADOS_EXECUCAO = configurar_log_experimento(NOME_EXP, __file__)

RAIZ = Path(__file__).resolve().parent.parent
DIR_TESTE = RAIZ / "datasets_yolo26_v2"
CAMINHO_MODELO_YOLO = RAIZ / "runs_v2" / "microplastic_yolo26_v2" / "weights" / "best.pt"
ADAPTADOR_LORA_TAMANHO = DIR_CHECKPOINTS / "size_lora"
NOMES_CLASSES = ["PE", "PET", "PP", "PS"]

TAMANHO_IMG = 1024
LIMIAR_CONF = 0.15
LIMIAR_IOU = 0.5
MARGEM_RECORTE_PX = 10  # PT: mesma margem usada em 01_extract_real_crops.py / EN: same margin as in 01_extract_real_crops.py
N_BOOTSTRAP = 10_000
SEMENTE = 42

AMPL_X = re.compile(r"(\d+)\s*[xX]")
AMPL_TRACO = re.compile(r"^\D*(\d+)-\d+")

RESUMO_REFERENCIA = (Path(__file__).resolve().parent.parent / "resultados"
                     / "20_predicted_box_and_end_to_end" / "summary.json")
MAX_DETS_TESTADOS = (300, 2000)
TOL_CHECAGEM = 0.002


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


def carregar_gt(caminho_rotulo: Path, W: int, H: int, ampliacao):
    """PT: Retorna lista de dicts: cls_id, xyxy_px, size_nm, regime.
    EN: Returns a list of dicts: cls_id, xyxy_px, size_nm, regime.
    """
    saida = []
    if not caminho_rotulo.exists():
        return saida
    um_por_px = 100.0 / ampliacao if ampliacao else None
    for linha in open(caminho_rotulo):
        partes = linha.split()
        if len(partes) < 5:
            continue
        id_cls = int(partes[0])
        cx, cy, w, h = map(float, partes[1:5])
        xyxy = yolo_para_xyxy_px(cx, cy, w, h, W, H)
        tamanho_px = max(w * W, h * H)
        tamanho_nm = tamanho_px * um_por_px * 1000.0 if um_por_px else None
        regime = None
        if tamanho_nm is not None:
            regime = "NANOPLASTIC" if tamanho_nm < 1000 else "MICROPLASTIC"
        saida.append({"cls_id": id_cls, "xyxy": xyxy, "size_nm": tamanho_nm, "regime": regime})
    return saida


def executar_uma_vez(max_det: int):
    if not CAMINHO_MODELO_YOLO.exists():
        REGISTRADOR.error("checkpoint YOLO nao existe: %s", CAMINHO_MODELO_YOLO)
        sys.exit(1)
    if not ADAPTADOR_LORA_TAMANHO.exists():
        REGISTRADOR.error("adapter LoRA nao existe: %s -- rode 17_size_finetune_lora.py primeiro", ADAPTADOR_LORA_TAMANHO)
        sys.exit(1)

    REGISTRADOR.info("=== condicao predicted-box + avaliacao fim-a-fim ===")
    modelo_yolo = YOLO(str(CAMINHO_MODELO_YOLO))

    from peft import PeftModel
    processador, modelo_base = carregar_modelo_base(ID_MODELO)
    modelo_vlm = PeftModel.from_pretrained(modelo_base, ADAPTADOR_LORA_TAMANHO)
    modelo_vlm.config.use_cache = True

    caminhos_img = sorted((DIR_TESTE / "images/test").glob("*.png"))

    linhas_tab = []
    n_gt_total = 0
    for caminho_img in caminhos_img:
        prefixo_cls = caminho_img.stem.split("_", 1)[0]
        nome_original = caminho_img.stem.split("_", 1)[1] if "_" in caminho_img.stem else caminho_img.stem
        ampliacao = interpretar_ampliacao(nome_original)
        if ampliacao is None:
            REGISTRADOR.warning("magnificacao nao resolvida para %s -- pulando imagem", caminho_img.name)
            continue

        imagem = Image.open(caminho_img).convert("RGB")
        W, H = imagem.size

        caminho_rotulo = DIR_TESTE / "labels/test" / f"{caminho_img.stem}.txt"
        gts = carregar_gt(caminho_rotulo, W, H, ampliacao)
        gts = [g for g in gts if g["regime"] is not None]
        n_gt_total += len(gts)

        pred = modelo_yolo.predict(source=str(caminho_img), imgsz=TAMANHO_IMG, conf=LIMIAR_CONF,
                                  max_det=max_det, verbose=False)[0]
        preds = []
        if pred.boxes is not None and len(pred.boxes):
            xyxy_px = pred.boxes.xyxy.cpu().numpy()
            ids_cls = pred.boxes.cls.cpu().numpy().astype(int)
            for caixa, c in zip(xyxy_px, ids_cls):
                preds.append({"cls_id": int(c), "xyxy": tuple(caixa)})

        # PT: pareamento espacial guloso, sem exigir classe igual (D_i puramente geométrico)
        # EN: greedy spatial matching, without requiring the same class (D_i is purely
        #     geometric)
        candidatas = []
        for gi, g in enumerate(gts):
            for pi, p in enumerate(preds):
                iou = iou_xyxy(g["xyxy"], p["xyxy"])
                if iou >= LIMIAR_IOU:
                    candidatas.append((iou, gi, pi))
        candidatas.sort(reverse=True)
        gt_pareado, pred_pareada = {}, set()
        for iou, gi, pi in candidatas:
            if gi in gt_pareado or pi in pred_pareada:
                continue
            gt_pareado[gi] = pi
            pred_pareada.add(pi)

        for gi, g in enumerate(gts):
            D_i = gi in gt_pareado
            P_i = False
            S_i = None
            regime_pred = None
            if D_i:
                p = preds[gt_pareado[gi]]
                P_i = (p["cls_id"] == g["cls_id"])
                if P_i:
                    x1, y1, x2, y2 = p["xyxy"]
                    x1 = max(0, x1 - MARGEM_RECORTE_PX)
                    y1 = max(0, y1 - MARGEM_RECORTE_PX)
                    x2 = min(W, x2 + MARGEM_RECORTE_PX)
                    y2 = min(H, y2 + MARGEM_RECORTE_PX)
                    recorte = imagem.crop((x1, y1, x2, y2))
                    bruto = classificar_imagem(processador, modelo_vlm, recorte, prompt=PROMPT_CLASSIFICACAO_TAMANHO)
                    regime_pred = extrair_rotulo(bruto, rotulos=ROTULOS_TAMANHO)
                    S_i = (regime_pred == g["regime"])

            E_i = bool(D_i and P_i and (S_i is True))
            linhas_tab.append({
                "source_image": caminho_img.name, "class_prefix": prefixo_cls,
                "gt_regime": g["regime"], "pred_regime": regime_pred,
                "D_i": D_i, "P_i": P_i, "S_i": S_i, "E_i": E_i,
            })

        REGISTRADOR.info("imagem=%s gt=%d casados=%d", caminho_img.name, len(gts), len(gt_pareado))
        print(f"{caminho_img.name}: gt={len(gts)} casados_espacialmente={len(gt_pareado)}")

    df = pd.DataFrame(linhas_tab)
    csv_saida = DIR_EXP / "reports" / f"predicted_box_end_to_end_maxdet{max_det}.csv"

    n = len(df)
    taxa_d = df["D_i"].mean()
    taxa_p_dado_d = df.loc[df["D_i"], "P_i"].mean() if df["D_i"].any() else float("nan")
    taxa_s_dado_dp = df.loc[df["D_i"] & df["P_i"], "S_i"].mean() if (df["D_i"] & df["P_i"]).any() else float("nan")
    taxa_e = df["E_i"].mean()

    print(f"\n=== Resumo (n={n} partículas de referência, {len(caminhos_img)} imagens) ===")
    print(f"D_i (localização, IoU>=0.5): {taxa_d:.3f}")
    print(f"P_i | D_i (polímero correto, dado localizado): {taxa_p_dado_d:.3f}")
    print(f"S_i | D_i,P_i (tamanho correto, dado localizado+polímero certo): {taxa_s_dado_dp:.3f}")
    print(f"E_i (fim-a-fim estrito, D*P*S): {taxa_e:.3f}")

    n_classificados = int((df["D_i"] & df["P_i"]).sum())
    if n_classificados > 0:
        sub = df[df["D_i"] & df["P_i"]].copy()
        acur_caixa_prevista = sub["S_i"].mean()
        print(f"\nAcurácia do SmolVLM SÓ nos casos classificados (condição predicted-box, n={n_classificados}): {acur_caixa_prevista:.3f}")
        print("Matriz de confusão (predicted-box):")
        print(pd.crosstab(sub["gt_regime"], sub["pred_regime"]))

        gerador = np.random.default_rng(SEMENTE)
        imgs = sub["source_image"].unique()
        grupos = {s: g for s, g in sub.groupby("source_image")}
        acurs = np.empty(N_BOOTSTRAP)
        for i in range(N_BOOTSTRAP):
            amostrados = gerador.choice(imgs, size=len(imgs), replace=True)
            rep = pd.concat([grupos[s] for s in amostrados], ignore_index=True)
            acurs[i] = rep["S_i"].mean()
        ic_inf, ic_sup = np.nanpercentile(acurs, [2.5, 97.5])
        print(f"IC 95% (bootstrap por imagem, {N_BOOTSTRAP} réplicas, {len(imgs)} imagens): [{ic_inf:.3f}, {ic_sup:.3f}]")
    else:
        acur_caixa_prevista = float("nan")
        ic_inf = ic_sup = float("nan")

    metricas = {
        "n_gt_particles": n,
        "D_rate": float(taxa_d),
        "P_rate_given_D": float(taxa_p_dado_d),
        "S_rate_given_DP": float(taxa_s_dado_dp) if not pd.isna(taxa_s_dado_dp) else None,
        "E_rate_strict_end_to_end": float(taxa_e),
        "n_classified_by_smolvlm": n_classificados,
        "smolvlm_accuracy_predicted_box": float(acur_caixa_prevista) if not pd.isna(acur_caixa_prevista) else None,
        "smolvlm_accuracy_ci95": [float(ic_inf), float(ic_sup)] if not pd.isna(ic_inf) else None,
        "iou_threshold": LIMIAR_IOU,
        "conf_threshold": LIMIAR_CONF,
        "crop_margin_px": MARGEM_RECORTE_PX,
        "max_det": max_det,
    }
    return metricas, df, csv_saida


def principal():
    if not RESUMO_REFERENCIA.exists():
        raise SystemExit(f"nao existe: {RESUMO_REFERENCIA}")
    referencia = json.loads(RESUMO_REFERENCIA.read_text())["metrics"]

    resultados = {}
    for md in MAX_DETS_TESTADOS:
        print(f"\n{'='*72}\nmax_det={md}\n{'='*72}")
        REGISTRADOR.info("iniciando run_once(max_det=%d)", md)
        metricas, df, csv_saida = executar_uma_vez(md)
        resultados[md] = (metricas, df, csv_saida)

    m300 = resultados[300][0]
    chaves = ["D_rate", "P_rate_given_D", "S_rate_given_DP", "E_rate_strict_end_to_end"]
    print(f"\n{'='*72}\nSANIDADE -- max_det=300 reproduz 20_predicted_box_and_end_to_end?\n{'='*72}")
    ok = True
    for k in chaves:
        d = abs(m300[k] - referencia[k])
        sinalizador = "" if d <= TOL_CHECAGEM else "  <-- DIVERGE"
        if d > TOL_CHECAGEM:
            ok = False
        print(f"   {k:26s} referência={referencia[k]*100:6.2f}%  obtido={m300[k]*100:6.2f}%  "
              f"dif={d*100:5.2f}pp{sinalizador}")
    if m300["n_classified_by_smolvlm"] != referencia["n_classified_by_smolvlm"]:
        print(f"   n_classified_by_smolvlm      referência={referencia['n_classified_by_smolvlm']}  "
              f"obtido={m300['n_classified_by_smolvlm']}  <-- DIVERGE")
        ok = False
    print("   ->", "SIM" if ok else "NAO")
    if not ok:
        print("\n   !! nao reproduz -- o resultado de max_det=2000 NAO deve ser usado.")
        return

    m2000, df2000, csv_saida_2000 = resultados[2000]
    csv_saida_2000.parent.mkdir(parents=True, exist_ok=True)
    df2000.to_csv(csv_saida_2000, index=False)
    print(f"\nCSV detalhado por particula (max_det=2000): {csv_saida_2000}")

    print(f"\n{'='*72}\nmax_det=300 (sanidade)  vs  max_det=2000\n{'='*72}")
    for k in chaves:
        print(f"   {k:26s} {m300[k]*100:6.2f}%  ->  {m2000[k]*100:6.2f}%"
              f"   ({(m2000[k]-m300[k])*100:+.2f} pp)")
    print(f"   n_classified_by_smolvlm      {m300['n_classified_by_smolvlm']:5d}  ->  "
          f"{m2000['n_classified_by_smolvlm']:5d}")

    salvar_resumo_experimento(NOME_EXP, METADADOS_EXECUCAO, {
        "note": "reavaliacao fim-a-fim do single run com max_det explicito -- gera "
                "as metricas do detector, a matriz de tamanho na condicao caixa-YOLO e o funil. A referencia "
                "(20_predicted_box_and_end_to_end) rodou com o padrao 300 do Ultralytics.",
        "published_reference": {k: referencia[k] for k in chaves + ["n_classified_by_smolvlm"]},
        "sanity_reproduces_published": ok,
        "maxdet_300": m300,
        "maxdet_2000": m2000,
        "csv_maxdet_2000": str(csv_saida_2000),
    })
    print(f"\nLog completo: {CAMINHO_LOG}")


if __name__ == "__main__":
    principal()
