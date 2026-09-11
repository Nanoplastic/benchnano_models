"""Reavalia os 4 checkpoints já treinados de Faster/Mask R-CNN (11_/12_)
com a correção de Precisão/Recall macro por classe em `rcnn_common.evaluate()`
(antes era média micro, pool de TP/FP/FN de todas as classes -- não
comparável com `metrics.box.mp`/`mr` do Ultralytics, usado por YOLOv10 e
Benchnano-seg, que é macro). Só inferência -- nenhum checkpoint é
retreinado, os pesos em runs/11_fasterrcnn|12_maskrcnn/<protocolo>/model_final.pt
são reaproveitados como estão.

Sobrescreve o campo "metrics" de cada summary.json existente (mantém
"environment" e "metadata"/train_history intactos) e registra o motivo em
metadata.reeval_note.

Uso:
    python3 15_reeval_rcnn_macro_pr.py
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comp_common as cc
import rcnn_common as rc

CLASS_NAMES = cc.CLASSES


def fasterrcnn_test_dataset(protocol: str):
    if protocol == "full":
        coco_json = cc.RESULTADOS_DIR / "11_train_fasterrcnn" / "coco_det_full" / "test.json"
        images_dir = cc.YOLO_DET_DATASET / "images" / "test"
    else:
        coco_json = cc.CURATED_CROP_DIR / "coco_det" / "test.json"
        images_dir = cc.CURATED_CROP_DIR / "det" / "images" / "test"
    return rc.CocoDetDataset(coco_json, images_dir, max_instances=None)


def maskrcnn_test_dataset(protocol: str):
    if protocol == "full":
        coco_json = cc.SEG_DATASET_DIR / "coco" / "test.json"
        images_dir = cc.SEG_DATASET_DIR / "images" / "test"
    else:
        coco_json = cc.CURATED_CROP_DIR / "coco_seg" / "test.json"
        images_dir = cc.CURATED_CROP_DIR / "seg" / "images" / "test"
    return rc.CocoSegDataset(coco_json, images_dir, max_instances=None)


JOBS = [
    ("11_fasterrcnn__full", "full", fasterrcnn_test_dataset,
     lambda: rc.build_fasterrcnn(num_classes_with_bg=len(CLASS_NAMES) + 1), False),
    ("11_fasterrcnn__patches", "patches", fasterrcnn_test_dataset,
     lambda: rc.build_fasterrcnn(num_classes_with_bg=len(CLASS_NAMES) + 1), False),
    ("12_maskrcnn__full", "full", maskrcnn_test_dataset,
     lambda: rc.build_maskrcnn(num_classes_with_bg=2), True),
    ("12_maskrcnn__patches", "patches", maskrcnn_test_dataset,
     lambda: rc.build_maskrcnn(num_classes_with_bg=2), True),
]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[15_reeval_rcnn_macro_pr] device={device}")

    for experiment, protocol, dataset_fn, build_model_fn, eval_masks in JOBS:
        summary_path = cc.RESULTADOS_DIR / experiment / "summary.json"
        if not summary_path.exists():
            print(f"  [PULAR] {experiment}: summary.json não encontrado")
            continue
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)

        model_path = Path(summary["metadata"]["model_path"])
        if not model_path.exists():
            print(f"  [PULAR] {experiment}: checkpoint não encontrado em {model_path}")
            continue

        old_metrics = summary["metrics"]["aggregate_metrics"]
        print(f"\n[15_reeval_rcnn_macro_pr] {experiment} -- carregando {model_path.name}")
        model = build_model_fn()
        model.load_state_dict(torch.load(model_path, map_location=device))

        test_ds = dataset_fn(protocol)
        new_metrics = rc.evaluate(model, test_ds, device, eval_masks=eval_masks)
        new_agg = new_metrics["aggregate_metrics"]

        print(f"  precisão: micro-antiga={old_metrics['precision']:.4f}  macro-nova={new_agg['precision']:.4f}")
        print(f"  recall:   micro-antiga={old_metrics['recall']:.4f}  macro-nova={new_agg['recall']:.4f}")
        print(f"  mAP50: antigo={old_metrics['map50']:.4f}  novo={new_agg['map50']:.4f}  (deve ser ~igual)")
        if "per_class" in new_metrics:
            names = {i + 1: c for i, c in enumerate(CLASS_NAMES)}
            for cls_id, v in new_metrics["per_class"].items():
                name = names.get(cls_id, f"classe {cls_id}")
                print(f"    {name:5s} tp={v['tp']:4d} fp={v['fp']:4d} fn={v['fn']:4d} "
                      f"P={v['precision']*100:5.1f}% R={v['recall']*100:5.1f}%")

        summary["metrics"] = new_metrics
        summary["metadata"]["reeval_note"] = (
            "Precisao/Recall recalculados por 15_reeval_rcnn_macro_pr.py em "
            + __import__("datetime").datetime.now().isoformat()
            + " -- media macro por classe (antes: media micro, pool de TP/FP/FN de "
              "todas as classes, nao comparavel com metrics.box.mp/mr do Ultralytics "
              "usado por YOLOv10/Benchnano-seg). mAP50/mAP50:95 inalterados (ja eram "
              "por classe com media, via torchmetrics). Nenhum checkpoint foi "
              "retreinado, so reavaliado."
        )
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"  [OK] {summary_path} atualizado")

    print("\n[15_reeval_rcnn_macro_pr] concluído -- rode 13_mina_compare_report.py e "
          "14_fig_comparacao_mina_barras.py de novo pra atualizar relatorio_final/.")


if __name__ == "__main__":
    main()
