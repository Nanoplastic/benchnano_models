"""Reavaliação macro de Faster/Mask R-CNN / Faster/Mask R-CNN macro re-evaluation.

PT: Reavalia os 4 checkpoints treinados de Faster/Mask R-CNN (11_/12_) com precisão/recall
    macro por classe em `rcnn_common.avaliar()`, a mesma convenção de `metrics.box.mp`/`mr`
    do Ultralytics (usada por YOLOv10 e Benchnano-seg). Só inferência: os pesos em
    runs/11_fasterrcnn|12_maskrcnn/<protocolo>/model_final.pt são reaproveitados sem
    retreino.
    Substitui o campo "metrics" de cada summary.json existente (mantém "environment" e
    "metadata"/train_history) e registra a reavaliação em metadata.reeval_note.
    Uso:
    python3 15_reeval_rcnn_macro_pr.py
EN: Re-evaluates the 4 trained Faster/Mask R-CNN checkpoints (11_/12_) with per-class macro
    precision/recall in `rcnn_common.avaliar()`, the same convention as Ultralytics
    `metrics.box.mp`/`mr` (used by YOLOv10 and Benchnano-seg). Inference only: the weights
    in runs/11_fasterrcnn|12_maskrcnn/<protocolo>/model_final.pt are reused without
    retraining.
    Replaces the "metrics" field of each existing summary.json (keeps "environment" and
    "metadata"/train_history) and records the re-evaluation in metadata.reeval_note.
    Usage:
    python3 15_reeval_rcnn_macro_pr.py
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comp_common as cc
import rcnn_common as rc

NOMES_CLASSES = cc.CLASSES_POLIMERO


def conjunto_teste_fasterrcnn(protocolo: str):
    if protocolo == "full":
        json_coco = cc.RESULTADOS_DIR / "11_train_fasterrcnn" / "coco_det_full" / "test.json"
        dir_imagens = cc.DATASET_DET_YOLO / "images" / "test"
    else:
        json_coco = cc.DIR_RECORTE_CURADO / "coco_det" / "test.json"
        dir_imagens = cc.DIR_RECORTE_CURADO / "det" / "images" / "test"
    return rc.ConjuntoCocoDeteccao(json_coco, dir_imagens, max_instances=None)


def conjunto_teste_maskrcnn(protocolo: str):
    if protocolo == "full":
        json_coco = cc.DIR_DATASET_SEG / "coco" / "test.json"
        dir_imagens = cc.DIR_DATASET_SEG / "images" / "test"
    else:
        json_coco = cc.DIR_RECORTE_CURADO / "coco_seg" / "test.json"
        dir_imagens = cc.DIR_RECORTE_CURADO / "seg" / "images" / "test"
    return rc.ConjuntoCocoSegmentacao(json_coco, dir_imagens, max_instances=None)


TAREFAS = [
    ("11_fasterrcnn__full", "full", conjunto_teste_fasterrcnn,
     lambda: rc.construir_fasterrcnn(num_classes_com_fundo=len(NOMES_CLASSES) + 1), False),
    ("11_fasterrcnn__patches", "patches", conjunto_teste_fasterrcnn,
     lambda: rc.construir_fasterrcnn(num_classes_com_fundo=len(NOMES_CLASSES) + 1), False),
    ("12_maskrcnn__full", "full", conjunto_teste_maskrcnn,
     lambda: rc.construir_maskrcnn(num_classes_com_fundo=2), True),
    ("12_maskrcnn__patches", "patches", conjunto_teste_maskrcnn,
     lambda: rc.construir_maskrcnn(num_classes_com_fundo=2), True),
]


def principal():
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[15_reeval_rcnn_macro_pr] device={dispositivo}")

    for experimento, protocolo, fn_conjunto, fn_construir_modelo, avaliar_mascaras in TAREFAS:
        caminho_resumo = cc.RESULTADOS_DIR / experimento / "summary.json"
        if not caminho_resumo.exists():
            print(f"  [PULAR] {experimento}: summary.json não encontrado")
            continue
        with open(caminho_resumo, encoding="utf-8") as f:
            resumo = json.load(f)

        caminho_modelo = Path(resumo["metadata"]["model_path"])
        if not caminho_modelo.exists():
            print(f"  [PULAR] {experimento}: checkpoint não encontrado em {caminho_modelo}")
            continue

        metricas_antigas = resumo["metrics"]["aggregate_metrics"]
        print(f"\n[15_reeval_rcnn_macro_pr] {experimento} -- carregando {caminho_modelo.name}")
        modelo = fn_construir_modelo()
        modelo.load_state_dict(torch.load(caminho_modelo, map_location=dispositivo))

        conj_teste = fn_conjunto(protocolo)
        novas_metricas = rc.avaliar(modelo, conj_teste, dispositivo, avaliar_mascaras=avaliar_mascaras)
        novo_agregado = novas_metricas["aggregate_metrics"]

        print(f"  precisão: micro-antiga={metricas_antigas['precision']:.4f}  macro-nova={novo_agregado['precision']:.4f}")
        print(f"  recall:   micro-antiga={metricas_antigas['recall']:.4f}  macro-nova={novo_agregado['recall']:.4f}")
        print(f"  mAP50: antigo={metricas_antigas['map50']:.4f}  novo={novo_agregado['map50']:.4f}  (deve ser ~igual)")
        if "per_class" in novas_metricas:
            nomes = {i + 1: c for i, c in enumerate(NOMES_CLASSES)}
            for id_cls, v in novas_metricas["per_class"].items():
                nome = nomes.get(id_cls, f"classe {id_cls}")
                print(f"    {nome:5s} tp={v['tp']:4d} fp={v['fp']:4d} fn={v['fn']:4d} "
                      f"P={v['precision']*100:5.1f}% R={v['recall']*100:5.1f}%")

        resumo["metrics"] = novas_metricas
        resumo["metadata"]["reeval_note"] = (
            "Precisao/Recall recalculados por 15_reeval_rcnn_macro_pr.py em "
            + __import__("datetime").datetime.now().isoformat()
            + " -- media macro por classe (antes: media micro, pool de TP/FP/FN de "
              "todas as classes, nao comparavel com metrics.box.mp/mr do Ultralytics "
              "usado por YOLOv10/Benchnano-seg). mAP50/mAP50:95 inalterados (ja eram "
              "por classe com media, via torchmetrics). Nenhum checkpoint foi "
              "retreinado, so reavaliado."
        )
        with open(caminho_resumo, "w", encoding="utf-8") as f:
            json.dump(resumo, f, indent=2, ensure_ascii=False)
        print(f"  [OK] {caminho_resumo} atualizado")

    print("\n[15_reeval_rcnn_macro_pr] concluído -- rode 13_mina_compare_report.py "
          "de novo pra atualizar relatorio_final/.")


if __name__ == "__main__":
    principal()
