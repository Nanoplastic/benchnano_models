"""Faster R-CNN (ResNet-50 FPN, torchvision) -- ver `rcnn_common.py` pro
porquê de torchvision em vez do Detectron2 original dos autores do MiNa.

Dois protocolos:
    --protocol full     -- experimentos/datasets_yolo26_v2/ (converte os
                            labels YOLO já existentes pra COCO on-the-fly,
                            sem mudar o dataset original; cap de instâncias
                            ativo, micrografias densas estouram VRAM)
    --protocol patches   -- experimentos_comparacao/datasets_curated_crop/det/
                            (COCO já pronto, sem cap -- patches são pequenos)

Execução única (seed=0) -- ver README sobre o prazo de submissão.

Uso:
    python3 11_train_fasterrcnn.py --protocol full [--epochs 20] [--smoke-test]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comp_common as cc
import rcnn_common as rc

CLASS_NAMES = cc.CLASSES
SEED = 0


def build_coco_det_from_yolo(split: str) -> Path:
    """Converte experimentos/datasets_yolo26_v2/{images,labels}/<split> (YOLO
    txt) pra COCO json, sem alterar o dataset original -- cacheado em
    resultados/ pra não refazer a cada rodada."""
    out_dir = cc.RESULTADOS_DIR / "11_train_fasterrcnn" / "coco_det_full"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{split}.json"
    if out_path.exists():
        return out_path

    from PIL import Image
    img_dir = cc.YOLO_DET_DATASET / "images" / split
    lbl_dir = cc.YOLO_DET_DATASET / "labels" / split
    images, annotations = [], []
    ann_id = 1
    for img_id, img_path in enumerate(sorted(img_dir.glob("*.png")), start=1):
        with Image.open(img_path) as im:
            W, H = im.size
        images.append({"id": img_id, "file_name": img_path.name, "width": W, "height": H})
        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        for line in lbl_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            cls_id, cx, cy, w, h = int(parts[0]), *map(float, parts[1:5])
            bw, bh = w * W, h * H
            x, y = cx * W - bw / 2, cy * H - bh / 2
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": cls_id + 1,
                "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0,
            })
            ann_id += 1
    coco = {"images": images, "annotations": annotations,
            "categories": [{"id": i + 1, "name": c} for i, c in enumerate(CLASS_NAMES)]}
    with open(out_path, "w") as f:
        json.dump(coco, f)
    return out_path


def dataset_for(protocol: str, split: str, matched_pool: bool = False):
    # cap de instâncias só se aplica ao TREINO (evita OOM na atribuição de
    # âncoras); no split de teste teria que ser sempre a groundtruth
    # completa, senão a avaliação vira uma amostra aleatória não
    # reprodutível do ground truth nas imagens densas (bug encontrado e
    # corrigido em 15_reeval_rcnn_macro_pr.py -- 3 das 15 imagens de teste
    # têm mais de 500 partículas anotadas).
    if protocol == "full":
        if matched_pool:
            coco_json = cc.RESULTADOS_DIR / "11_train_fasterrcnn" / "coco_det_full_matched" / f"{split}.json"
            if not coco_json.exists():
                raise FileNotFoundError(f"{coco_json} não existe -- rode 16_build_det_full_matched.py primeiro")
        else:
            coco_json = build_coco_det_from_yolo(split)
        images_dir = cc.YOLO_DET_DATASET / "images" / split
        max_inst = cc.MAX_INSTANCES_PER_IMAGE if split == "train" else None
    else:
        coco_json = cc.CURATED_CROP_DIR / "coco_det" / f"{split}.json"
        images_dir = cc.CURATED_CROP_DIR / "det" / "images" / split
        max_inst = None
    return rc.CocoDetDataset(coco_json, images_dir, max_instances=max_inst)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", required=True, choices=["full", "patches"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--matched-pool", action="store_true",
                         help="protocolo full só: usa o mesmo pool de imagens de datasets_baseline_seg/ "
                              "(155 treino/14 teste) em vez do dataset completo (162/15) -- ver "
                              "16_build_det_full_matched.py")
    args = parser.parse_args()
    if args.matched_pool and args.protocol != "full":
        parser.error("--matched-pool só se aplica a --protocol full")

    epochs = args.epochs or (1 if args.smoke_test else (30 if args.protocol == "patches" else 100))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    exp_suffix = f"{args.protocol}_matched" if args.matched_pool else args.protocol

    print(f"\n{'='*70}\n[11_train_fasterrcnn] protocolo={args.protocol} matched_pool={args.matched_pool} epochs={epochs} device={device}\n{'='*70}")
    torch.manual_seed(SEED)

    train_ds = dataset_for(args.protocol, "train", args.matched_pool)
    test_ds = dataset_for(args.protocol, "test", args.matched_pool)
    print(f"[11_train_fasterrcnn] treino={len(train_ds)} imagens, teste={len(test_ds)} imagens")

    model = rc.build_fasterrcnn(num_classes_with_bg=len(CLASS_NAMES) + 1)
    t0 = time.time()
    history = rc.train_loop(model, train_ds, device, epochs=epochs, batch_size=4 if args.protocol == "full" else 8)
    duration = time.time() - t0

    out_dir = cc.RUNS_DIR / "11_fasterrcnn" / exp_suffix
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "model_final.pt")

    metrics = rc.evaluate(model, test_ds, device)
    print(f"[11_train_fasterrcnn] {exp_suffix}: {metrics['aggregate_metrics']}")

    if not args.smoke_test:
        cc.save_summary(f"11_fasterrcnn__{exp_suffix}", metrics=metrics, extra_metadata={
            "protocol": args.protocol, "matched_pool": args.matched_pool, "epochs": epochs,
            "max_instances_per_image": cc.MAX_INSTANCES_PER_IMAGE if args.protocol == "full" else None,
            "model_path": str(out_dir / "model_final.pt"), "train_history": history,
            "duration_sec": round(duration, 1), "duration_human": cc.human_duration(duration),
        })
    else:
        print(f"[11_train_fasterrcnn] SMOKE TEST OK -- {cc.human_duration(duration)} pra {epochs} época(s)")


if __name__ == "__main__":
    main()
