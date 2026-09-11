"""Mask R-CNN (ResNet-50 FPN, torchvision) -- ver `rcnn_common.py` pro
porquê de torchvision em vez do Detectron2 original dos autores do MiNa.

Dois protocolos, ambos já têm COCO-seg pronto (classe única "particle"):
    --protocol full     -- experimentos_comparacao/datasets_baseline_seg/coco/
                            (já existe, construído por 00_build_seg_dataset.py;
                            cap de instâncias ativo, mesma razão do Faster R-CNN)
    --protocol patches   -- experimentos_comparacao/datasets_curated_crop/coco_seg/
                            (sem cap -- patches são pequenos)

Reporta mAP de caixa (tabela principal) e AP de máscara (texto, junto do
Benchnano-seg -- main_pt.tex: "instance mask evaluated separately for
Benchnano-seg and Mask R-CNN"). Execução única (seed=0).

Uso:
    python3 12_train_maskrcnn.py --protocol full [--epochs 20] [--smoke-test]
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comp_common as cc
import rcnn_common as rc

SEED = 0


def dataset_for(protocol: str, split: str, fixed_pool: bool = False):
    # cap de instâncias só se aplica ao TREINO (evita OOM na atribuição de
    # âncoras); no split de teste teria que ser sempre a groundtruth
    # completa, senão a avaliação vira uma amostra aleatória não
    # reprodutível do ground truth nas imagens densas (bug encontrado e
    # corrigido em 15_reeval_rcnn_macro_pr.py -- 2 das 15 imagens de teste
    # têm mais de 500 partículas anotadas).
    if protocol == "full":
        seg_dir = (cc.COMP_ROOT / "datasets_baseline_seg_fixed") if fixed_pool else cc.SEG_DATASET_DIR
        coco_json = seg_dir / "coco" / f"{split}.json"
        images_dir = seg_dir / "images" / split
        max_inst = cc.MAX_INSTANCES_PER_IMAGE if split == "train" else None
    else:
        coco_json = cc.CURATED_CROP_DIR / "coco_seg" / f"{split}.json"
        images_dir = cc.CURATED_CROP_DIR / "seg" / "images" / split
        max_inst = None
    return rc.CocoSegDataset(coco_json, images_dir, max_instances=max_inst)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", required=True, choices=["full", "patches"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--fixed-pool", action="store_true",
                         help="protocolo full só: usa datasets_baseline_seg_fixed/ (162 treino/15 teste, "
                              "com as 10 imagens PET-B recuperadas) em vez de datasets_baseline_seg/ (155/14)")
    args = parser.parse_args()
    if args.fixed_pool and args.protocol != "full":
        parser.error("--fixed-pool só se aplica a --protocol full")

    epochs = args.epochs or (1 if args.smoke_test else (30 if args.protocol == "patches" else 100))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_suffix = f"{args.protocol}_fixed" if args.fixed_pool else args.protocol

    print(f"\n{'='*70}\n[12_train_maskrcnn] protocolo={args.protocol} fixed_pool={args.fixed_pool} epochs={epochs} device={device}\n{'='*70}")
    torch.manual_seed(SEED)

    train_ds = dataset_for(args.protocol, "train", args.fixed_pool)
    test_ds = dataset_for(args.protocol, "test", args.fixed_pool)
    print(f"[12_train_maskrcnn] treino={len(train_ds)} imagens, teste={len(test_ds)} imagens")

    model = rc.build_maskrcnn(num_classes_with_bg=2)  # background + particle
    t0 = time.time()
    history = rc.train_loop(model, train_ds, device, epochs=epochs, batch_size=2 if args.protocol == "full" else 6)
    duration = time.time() - t0

    out_dir = cc.RUNS_DIR / "12_maskrcnn" / exp_suffix
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model_final.pt")

    metrics = rc.evaluate(model, test_ds, device, eval_masks=True)
    print(f"[12_train_maskrcnn] {exp_suffix}: {metrics['aggregate_metrics']}")

    if not args.smoke_test:
        cc.save_summary(f"12_maskrcnn__{exp_suffix}", metrics=metrics, extra_metadata={
            "protocol": args.protocol, "fixed_pool": args.fixed_pool, "epochs": epochs,
            "max_instances_per_image": cc.MAX_INSTANCES_PER_IMAGE if args.protocol == "full" else None,
            "model_path": str(out_dir / "model_final.pt"), "train_history": history,
            "duration_sec": round(duration, 1), "duration_human": cc.human_duration(duration),
        })
    else:
        print(f"[12_train_maskrcnn] SMOKE TEST OK -- {cc.human_duration(duration)} pra {epochs} época(s)")


if __name__ == "__main__":
    main()
