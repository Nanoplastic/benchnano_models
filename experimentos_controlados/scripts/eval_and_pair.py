"""Avaliação e pareamento das repetições / Repetition evaluation and pairing.

PT: Funções de avaliação usadas pelos 6 scripts `NN_..._10x.py`, mais o modo `--pair`, que
    roda a avaliação fim a fim pareada (YOLO26L repetição i + SmolVLM de regime de tamanho
    repetição i) com a lógica de experimentos/scripts/20_predicted_box_and_end_to_end.py.
    Reimplementado aqui (em vez de importar o script 20) porque aquele script fixa os
    caminhos de checkpoint como constantes de módulo (CAMINHO_MODELO_YOLO,
    ADAPTADOR_LORA_TAMANHO); apontar para o checkpoint de outra repetição exigiria editar o
    original. O pareamento por IoU, o cálculo de D_i/P_i/S_i/E_i e a leitura da ampliação
    são os mesmos do script 20, só parametrizados.
EN: Evaluation functions used by the 6 `NN_..._10x.py` scripts, plus the `--pair` mode,
    which runs the paired end-to-end evaluation (YOLO26L repetition i + size regime SmolVLM
    repetition i) with the logic of experimentos/scripts/20_predicted_box_and_end_to_end.py.
    Reimplemented here (instead of importing script 20) because that script fixes the
    checkpoint paths as module constants (CAMINHO_MODELO_YOLO, ADAPTADOR_LORA_TAMANHO);
    pointing to another repetition's checkpoint would require editing the original. IoU
    matching, the D_i/P_i/S_i/E_i computation and magnification parsing are the same as in
    script 20, just parametrized.
"""
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctrl_common as cc

sys.path.insert(0, str(cc.SCRIPTS_EXPERIMENTOS))
from vlm_common import (
    ID_MODELO, PROMPT_CLASSIFICACAO_TAMANHO, ROTULOS_TAMANHO, classificar_imagem,
    extrair_rotulo, carregar_modelo_base,
)

from ultralytics import YOLO

DIR_TESTE = cc.RAIZ_PROJETO_ORIGEM / "experimentos" / "datasets_yolo26_v2"
NOMES_CLASSES = ["PE", "PET", "PP", "PS"]
TAMANHO_IMG = 1024
LIMIAR_CONF = 0.15
LIMIAR_IOU = 0.5
MARGEM_RECORTE_PX = 10
N_BOOTSTRAP = 10_000
SEMENTE_BOOTSTRAP = 42

AMPL_X = re.compile(r"(\d+)\s*[xX]")
AMPL_TRACO = re.compile(r"^\D*(\d+)-\d+")


# PT: Avaliação padrão do YOLO26L na partição de teste (espelha a Parte 1 de
#     14_yolo_test_eval.py: métricas agregadas e por classe via model.val())
# EN: Standard YOLO26L evaluation on the test split (mirrors Part 1 of 14_yolo_test_eval.py:
#     aggregate and per-class metrics via model.val())

def avaliar_yolo_conjunto_teste(caminho_modelo) -> dict:
    caminho_modelo = Path(caminho_modelo)
    modelo = YOLO(str(caminho_modelo))

    yaml_dados_teste = DIR_TESTE / "data_test.yaml"
    if not yaml_dados_teste.exists():
        yaml_dados_teste.write_text(
            f"path: {DIR_TESTE}\n"
            f"train: images/test\n"
            f"val: images/test\n"
            f"nc: {len(NOMES_CLASSES)}\n"
            f"names:\n" + "\n".join(f"  - {c}" for c in NOMES_CLASSES) + "\n"
        )

    metricas = modelo.val(data=str(yaml_dados_teste), split="val", imgsz=TAMANHO_IMG,
                         conf=LIMIAR_CONF, iou=0.7, plots=False, verbose=False)
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
    return {"model_path": str(caminho_modelo), "aggregate_metrics": agregado, "per_class_metrics": dados_por_classe}


# PT: Avaliação fim a fim pareada (espelha
#     experimentos/scripts/20_predicted_box_and_end_to_end.py, parametrizada por checkpoint)
# EN: Paired end-to-end evaluation (mirrors
#     experimentos/scripts/20_predicted_box_and_end_to_end.py, parametrized by checkpoint)

def interpretar_ampliacao(nome: str):
    m = AMPL_X.findall(nome)
    if m:
        return int(m[-1])
    m = AMPL_TRACO.match(nome)
    if m:
        return int(m.group(1))
    return None


def _yolo_para_xyxy_px(cx, cy, w, h, W, H):
    return ((cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H)


def _iou_xyxy_interno(a, b):
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


def _carregar_gt(caminho_rotulo: Path, W: int, H: int, ampliacao):
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
        xyxy = _yolo_para_xyxy_px(cx, cy, w, h, W, H)
        tamanho_px = max(w * W, h * H)
        tamanho_nm = tamanho_px * um_por_px * 1000.0 if um_por_px else None
        regime = None
        if tamanho_nm is not None:
            regime = "NANOPLASTIC" if tamanho_nm < 1000 else "MICROPLASTIC"
        saida.append({"cls_id": id_cls, "xyxy": xyxy, "size_nm": tamanho_nm, "regime": regime})
    return saida


def avaliar_fim_a_fim(caminho_modelo_yolo, caminho_adaptador_lora_tamanho) -> dict:
    caminho_modelo_yolo = Path(caminho_modelo_yolo)
    caminho_adaptador_lora_tamanho = Path(caminho_adaptador_lora_tamanho)
    if not caminho_modelo_yolo.exists():
        raise FileNotFoundError(f"checkpoint YOLO não existe: {caminho_modelo_yolo}")
    if not caminho_adaptador_lora_tamanho.exists():
        raise FileNotFoundError(f"adapter LoRA não existe: {caminho_adaptador_lora_tamanho}")

    from peft import PeftModel

    modelo_yolo = YOLO(str(caminho_modelo_yolo))
    processador, modelo_base = carregar_modelo_base(ID_MODELO)
    modelo_vlm = PeftModel.from_pretrained(modelo_base, caminho_adaptador_lora_tamanho)
    modelo_vlm.config.use_cache = True

    caminhos_img = sorted((DIR_TESTE / "images/test").glob("*.png"))
    linhas_tab = []

    for caminho_img in caminhos_img:
        nome_original = caminho_img.stem.split("_", 1)[1] if "_" in caminho_img.stem else caminho_img.stem
        ampliacao = interpretar_ampliacao(nome_original)
        if ampliacao is None:
            continue

        imagem = Image.open(caminho_img).convert("RGB")
        W, H = imagem.size

        caminho_rotulo = DIR_TESTE / "labels/test" / f"{caminho_img.stem}.txt"
        gts = [g for g in _carregar_gt(caminho_rotulo, W, H, ampliacao) if g["regime"] is not None]

        pred = modelo_yolo.predict(source=str(caminho_img), imgsz=TAMANHO_IMG, conf=LIMIAR_CONF, verbose=False)[0]
        preds = []
        if pred.boxes is not None and len(pred.boxes):
            xyxy_px = pred.boxes.xyxy.cpu().numpy()
            ids_cls = pred.boxes.cls.cpu().numpy().astype(int)
            for caixa, c in zip(xyxy_px, ids_cls):
                preds.append({"cls_id": int(c), "xyxy": tuple(caixa)})

        candidatas = []
        for gi, g in enumerate(gts):
            for pi, p in enumerate(preds):
                iou = _iou_xyxy_interno(g["xyxy"], p["xyxy"])
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
            linhas_tab.append({"source_image": caminho_img.name, "gt_regime": g["regime"],
                         "pred_regime": regime_pred, "D_i": D_i, "P_i": P_i, "S_i": S_i, "E_i": E_i})

    df = pd.DataFrame(linhas_tab)
    n = len(df)
    taxa_d = float(df["D_i"].mean())
    taxa_p_dado_d = float(df.loc[df["D_i"], "P_i"].mean()) if df["D_i"].any() else float("nan")
    taxa_s_dado_dp = float(df.loc[df["D_i"] & df["P_i"], "S_i"].mean()) if (df["D_i"] & df["P_i"]).any() else float("nan")
    taxa_e = float(df["E_i"].mean())

    n_classificados = int((df["D_i"] & df["P_i"]).sum())
    acur_caixa_prevista = float(df.loc[df["D_i"] & df["P_i"], "S_i"].mean()) if n_classificados > 0 else float("nan")

    return {
        "n_gt_particles": n,
        "D_rate": taxa_d,
        "P_rate_given_D": taxa_p_dado_d,
        "S_rate_given_DP": taxa_s_dado_dp if not pd.isna(taxa_s_dado_dp) else None,
        "E_rate_strict_end_to_end": taxa_e,
        "n_classified_by_smolvlm": n_classificados,
        "smolvlm_accuracy_predicted_box": acur_caixa_prevista if not pd.isna(acur_caixa_prevista) else None,
        "per_particle": df.to_dict(orient="records"),
    }


# PT: Modo --pair: roda o fim a fim pareado (repetição i do YOLO + repetição i do SmolVLM de
#     regime de tamanho) para cada uma das 10 repetições, depois que os dois experimentos de
#     origem tiverem as 10 repetições prontas.
# EN: --pair mode: runs the paired end-to-end evaluation (YOLO repetition i + size regime
#     SmolVLM repetition i) for each of the 10 repetitions, once both source experiments
#     have all 10 repetitions done.

EXPERIMENTO_PAREADO = "07_end_to_end_paired"


def executar_pareamento():
    estado = cc.carregar_estado_fila()
    reps_yolo = cc.carregar_todos_resumos_rep("01_yolo26_main")
    reps_tamanho = cc.carregar_todos_resumos_rep("05_smolvlm_size_lora")

    if len(reps_yolo) < cc.N_REPS or len(reps_tamanho) < cc.N_REPS:
        print(f"[{EXPERIMENTO_PAREADO}] esperando: 01_yolo26_main tem {len(reps_yolo)}/10, "
              f"05_smolvlm_size_lora tem {len(reps_tamanho)}/10 repetições. Rode os dois primeiro.")
        return

    for semente in cc.SEMENTES:
        if cc.rep_concluida(estado, EXPERIMENTO_PAREADO, semente):
            print(f"[{EXPERIMENTO_PAREADO}] rep {semente:02d} já concluída, pulando.")
            continue

        resumo_yolo = next(r for r in reps_yolo if r["repetition_index"] == semente)
        resumo_tamanho = next(r for r in reps_tamanho if r["repetition_index"] == semente)
        ckpt_yolo = Path(resumo_yolo["metadata"]["best_weights"])
        adaptador_tamanho = Path(resumo_tamanho["metadata"]["adapter_dir"])

        print(f"\n=== {EXPERIMENTO_PAREADO} — repetição {semente:02d}/9 (YOLO+SmolVLM pareados) ===")
        cc.definir_tarefa_atual(EXPERIMENTO_PAREADO, semente)

        metricas = avaliar_fim_a_fim(ckpt_yolo, adaptador_tamanho)
        por_particula = metricas.pop("per_particle")
        pd.DataFrame(por_particula).to_csv(
            cc.dir_rep(EXPERIMENTO_PAREADO, semente) / "predicted_box_end_to_end.csv", index=False)

        print(f"[{EXPERIMENTO_PAREADO}] rep {semente:02d}: E_rate={metricas['E_rate_strict_end_to_end']:.3f} "
              f"D_rate={metricas['D_rate']:.3f}")
        cc.salvar_resumo_rep(EXPERIMENTO_PAREADO, semente, semente, metricas, metadados_extras={
            "yolo_checkpoint": str(ckpt_yolo), "size_lora_adapter": str(adaptador_tamanho),
        })
        cc.marcar_rep_concluida(estado, EXPERIMENTO_PAREADO, semente)
        cc.limpar_tarefa_atual()


if __name__ == "__main__":
    executar_pareamento()
