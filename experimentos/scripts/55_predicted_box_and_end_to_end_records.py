"""Registros por partícula da avaliação fim a fim / Per-particle end-to-end records.

PT: Camada 3 (representação digital) para a condição "caixa prevista": a mesma avaliação fim
    a fim de `20_predicted_box_and_end_to_end.py`, persistindo um `RegistroParticula` por
    partícula de referência em vez de só as taxas agregadas.
    O script 20 e seu CSV (`reports/predicted_box_end_to_end.csv`) não são alterados. Este
    script recalcula a mesma avaliação (mesmos checkpoints, limiares de confiança/IoU e
    pareamento) e imprime as mesmas quatro taxas (D/P/S/E); rodar os dois deve dar valores
    idênticos, o que confirma que a persistência não muda a análise.
    O que este script acrescenta em relação ao 20:
    - mantém a caixa PREVISTA e a confiança do YOLO por partícula;
    - dá a cada partícula de referência um `particle_index` estável (posição no arquivo de
    rótulos de teste da imagem). Não é o id de anotação original do MiNa: o pareamento por
    IoU contra os JSONs COCO brutos (`MPDataset/Full_Images/COCO Format`) falha em ~80% das
    partículas em imagens densas, porque a contagem de anotações não coincide com o número
    de partículas da partição de teste. O índice é estável entre execuções porque o arquivo
    de rótulos de teste é estático;
    - grava tudo como `RegistroParticula` (ver `particle_record.py`), ACRESCENTANDO (nunca
    sobrescrevendo) a `resultados/particle_records/predicted_box_end_to_end.jsonl`, com um
    `run_id` por execução; reprocessar com outro checkpoint ou prompt não apaga registros
    anteriores.
    Uso:
    python3 55_predicted_box_and_end_to_end_records.py
EN: Layer 3 (digital representation) for the "predicted box" condition: the same end-to-end
    evaluation as `20_predicted_box_and_end_to_end.py`, persisting one `RegistroParticula`
    per reference particle instead of only aggregate rates.
    Script 20 and its CSV (`reports/predicted_box_end_to_end.csv`) are left unchanged. This
    script recomputes the same evaluation (same checkpoints, confidence/IoU thresholds and
    matching) and prints the same four rates (D/P/S/E); running both should give identical
    values, which confirms that persistence does not change the analysis.
    What this script adds relative to 20:
    - keeps the PREDICTED box and the YOLO confidence per particle;
    - gives each reference particle a stable `particle_index` (position in the image's test
    label file). It is not the original MiNa annotation id: IoU matching against the raw
    COCO JSONs (`MPDataset/Full_Images/COCO Format`) fails for ~80% of particles in dense
    images, because the annotation count does not match the number of particles in the test
    split. The index is stable across runs because the test label file is static;
    - writes everything as `RegistroParticula` (see `particle_record.py`), APPENDING (never
    overwriting) to `resultados/particle_records/predicted_box_end_to_end.jsonl`, with one
    `run_id` per run; reprocessing with another checkpoint or prompt does not delete earlier
    records.
    Usage:
    python3 55_predicted_box_and_end_to_end_records.py
"""
import re
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from particle_record import (
    RegistroParticula, DIR_REGISTROS_PARTICULAS, anexar_registros,
    coletar_versoes_modelos, status_avaliacao, gerar_id_execucao,
)
from vlm_common import (
    DIR_CHECKPOINTS, ID_MODELO, PROMPT_CLASSIFICACAO_TAMANHO, ROTULOS_TAMANHO,
    classificar_imagem, extrair_rotulo, carregar_modelo_base, salvar_resumo_experimento,
    configurar_log_experimento,
)

from ultralytics import YOLO

NOME_EXP = "55_predicted_box_and_end_to_end_records"
DIR_EXP, CAMINHO_LOG, REGISTRADOR, METADADOS_EXECUCAO = configurar_log_experimento(NOME_EXP, __file__)

RAIZ = Path(__file__).resolve().parent.parent
DIR_TESTE = RAIZ / "datasets_yolo26_v2"
CAMINHO_MODELO_YOLO = RAIZ / "runs_v2" / "microplastic_yolo26_v2" / "weights" / "best.pt"
ADAPTADOR_LORA_TAMANHO = DIR_CHECKPOINTS / "size_lora"
NOMES_CLASSES = ["PE", "PET", "PP", "PS"]

TAMANHO_IMG = 1024
LIMIAR_CONF = 0.15
LIMIAR_IOU = 0.5
MARGEM_RECORTE_PX = 10

CAMINHO_SAIDA = DIR_REGISTROS_PARTICULAS / "predicted_box_end_to_end.jsonl"

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


def carregar_gt(caminho_rotulo: Path, W: int, H: int, ampliacao):
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


def principal():
    if not CAMINHO_MODELO_YOLO.exists():
        REGISTRADOR.error("checkpoint YOLO nao existe: %s", CAMINHO_MODELO_YOLO)
        sys.exit(1)
    if not ADAPTADOR_LORA_TAMANHO.exists():
        REGISTRADOR.error("adapter LoRA nao existe: %s -- rode 17_size_finetune_lora.py primeiro", ADAPTADOR_LORA_TAMANHO)
        sys.exit(1)

    REGISTRADOR.info("=== condicao predicted-box + avaliacao fim-a-fim, com registros de particula (Camada 3) ===")
    versoes_modelos = coletar_versoes_modelos({
        "yolo_checkpoint": str(CAMINHO_MODELO_YOLO),
        "size_lora_adapter": str(ADAPTADOR_LORA_TAMANHO),
        "model_id": ID_MODELO,
        "conf_threshold": LIMIAR_CONF,
        "iou_threshold": LIMIAR_IOU,
    })
    id_execucao = gerar_id_execucao(versoes_modelos, rotulo_exp=NOME_EXP)
    REGISTRADOR.info("run_id=%s", id_execucao)

    modelo_yolo = YOLO(str(CAMINHO_MODELO_YOLO))

    from peft import PeftModel
    processador, modelo_base = carregar_modelo_base(ID_MODELO)
    modelo_vlm = PeftModel.from_pretrained(modelo_base, ADAPTADOR_LORA_TAMANHO)
    modelo_vlm.config.use_cache = True

    caminhos_img = sorted((DIR_TESTE / "images/test").glob("*.png"))

    linhas_tab = []
    registros = []
    for caminho_img in caminhos_img:
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

        pred = modelo_yolo.predict(source=str(caminho_img), imgsz=TAMANHO_IMG, conf=LIMIAR_CONF, verbose=False)[0]
        preds = []
        if pred.boxes is not None and len(pred.boxes):
            xyxy_px = pred.boxes.xyxy.cpu().numpy()
            ids_cls = pred.boxes.cls.cpu().numpy().astype(int)
            confs = pred.boxes.conf.cpu().numpy()
            for caixa, c, conf in zip(xyxy_px, ids_cls, confs):
                preds.append({"cls_id": int(c), "xyxy": tuple(float(v) for v in caixa), "conf": float(conf)})

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
            P_i, S_i, regime_pred = False, None, None
            caixa_pred, conf_pred, pred_polimero = None, None, None
            if D_i:
                p = preds[gt_pareado[gi]]
                caixa_pred, conf_pred = p["xyxy"], p["conf"]
                pred_polimero = NOMES_CLASSES[p["cls_id"]]
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
            status = status_avaliacao(D_i, P_i, S_i)

            linhas_tab.append({"source_image": caminho_img.name, "gt_regime": g["regime"],
                         "pred_regime": regime_pred, "D_i": D_i, "P_i": P_i, "S_i": S_i, "E_i": E_i})

            registros.append(RegistroParticula(
                source_image=caminho_img.name,
                particle_index=gi,
                magnification=float(ampliacao),
                box_ref_xyxy=tuple(float(v) for v in g["xyxy"]),
                polymer_label_ref=NOMES_CLASSES[g["cls_id"]],
                size_nm_ref=g["size_nm"],
                size_regime_ref=g["regime"],
                detected=D_i,
                box_pred_xyxy=caixa_pred,
                polymer_label_pred=pred_polimero,
                detector_confidence=conf_pred,
                polymer_correct=P_i,
                size_regime_pred=regime_pred,
                size_correct=S_i,
                end_to_end_correct=E_i,
                evaluation_status=status,
                run_id=id_execucao,
                created_at_utc=versoes_modelos["timestamp_utc"],
                model_versions=versoes_modelos,
            ))

        REGISTRADOR.info("imagem=%s gt=%d casados=%d", caminho_img.name, len(gts), len(gt_pareado))
        print(f"{caminho_img.name}: gt={len(gts)} casados_espacialmente={len(gt_pareado)}")

    caminho_registros_saida = anexar_registros(registros, CAMINHO_SAIDA)

    df = pd.DataFrame(linhas_tab)
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
    print("\nEstes quatro números devem bater com os de "
          "20_predicted_box_and_end_to_end.py -- se não baterem, a mudança "
          "introduziu uma diferença analítica e não deve ser usada sem investigar.")

    n_classificados = int((df["D_i"] & df["P_i"]).sum())
    acur_caixa_prevista = df.loc[df["D_i"] & df["P_i"], "S_i"].mean() if n_classificados > 0 else float("nan")

    salvar_resumo_experimento(NOME_EXP, METADADOS_EXECUCAO, {
        "run_id": id_execucao,
        "n_gt_particles": n,
        "D_rate": float(taxa_d),
        "P_rate_given_D": float(taxa_p_dado_d),
        "S_rate_given_DP": float(taxa_s_dado_dp) if not pd.isna(taxa_s_dado_dp) else None,
        "E_rate_strict_end_to_end": float(taxa_e),
        "n_classified_by_smolvlm": n_classificados,
        "smolvlm_accuracy_predicted_box": float(acur_caixa_prevista) if not pd.isna(acur_caixa_prevista) else None,
        "particle_records_path": str(caminho_registros_saida),
    })
    print(f"\nLog completo: {CAMINHO_LOG}")
    print(f"Registros de partícula (Camada 3, acrescentados, nunca sobrescritos): {caminho_registros_saida}")


if __name__ == "__main__":
    principal()
