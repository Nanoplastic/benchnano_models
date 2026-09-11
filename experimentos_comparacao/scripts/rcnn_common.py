"""Faster R-CNN / Mask R-CNN via torchvision (não Detectron2 -- ver
main_pt.tex linha 1031: "reimplemented with the torchvision package...
due to a library version incompatibility between Detectron2 and the
PyTorch version already fixed in this project").

Dataset lê direto dos COCO JSONs gerados por `00_build_seg_dataset.py`
(protocolo imagem-inteira, classe única) e `08_build_curated_crop_dataset.py`
(protocolo patches, ambos os formatos) -- nenhum dataset é gerado aqui,
só consumido.

Cap de instâncias por imagem (`MAX_INSTANCES_PER_IMAGE` em comp_common.py):
necessário só no protocolo imagem-inteira -- micrografias com 1000+
partículas estouram VRAM no cálculo de IoU âncora×GT (confirmado: o
Detectron2 que rodou antes de ser interrompido mostrava repetidamente
"Attempting to copy inputs of pairwise_iou to CPU due to CUDA OOM"
nessas imagens). Patches 256x256 não precisam do cap (poucas partículas
por patch, por construção).
"""
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn,
                                           FasterRCNN_ResNet50_FPN_Weights, MaskRCNN_ResNet50_FPN_Weights)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comp_common as cc


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------

class CocoDetDataset(Dataset):
    """bbox multi-classe (category_id 1..N no COCO -> label 1..N no torch;
    0 é sempre background, convenção torchvision)."""

    def __init__(self, coco_json: Path, images_dir: Path, max_instances: int | None = None):
        with open(coco_json, encoding="utf-8") as f:
            d = json.load(f)
        self.images_dir = images_dir
        self.images = {im["id"]: im for im in d["images"]}
        self.by_image = {}
        for ann in d["annotations"]:
            self.by_image.setdefault(ann["image_id"], []).append(ann)
        self.ids = list(self.images.keys())
        self.max_instances = max_instances

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        im_meta = self.images[img_id]
        img = Image.open(self.images_dir / im_meta["file_name"]).convert("RGB")
        anns = self.by_image.get(img_id, [])
        if self.max_instances and len(anns) > self.max_instances:
            anns = random.sample(anns, self.max_instances)
        boxes, labels = [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(a["category_id"])  # já 1-based no COCO gerado por este projeto
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        target = {"boxes": boxes_t, "labels": labels_t, "image_id": torch.tensor([img_id])}
        return torchvision.transforms.functional.to_tensor(img), target


class CocoSegDataset(Dataset):
    """polígono classe única (category_id sempre 1 -> label 1)."""

    def __init__(self, coco_json: Path, images_dir: Path, max_instances: int | None = None):
        with open(coco_json, encoding="utf-8") as f:
            d = json.load(f)
        self.images_dir = images_dir
        self.images = {im["id"]: im for im in d["images"]}
        self.by_image = {}
        for ann in d["annotations"]:
            self.by_image.setdefault(ann["image_id"], []).append(ann)
        self.ids = list(self.images.keys())
        self.max_instances = max_instances

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        im_meta = self.images[img_id]
        img = Image.open(self.images_dir / im_meta["file_name"]).convert("RGB")
        W, H = im_meta["width"], im_meta["height"]
        anns = self.by_image.get(img_id, [])
        if self.max_instances and len(anns) > self.max_instances:
            anns = random.sample(anns, self.max_instances)
        boxes, labels, masks = [], [], []
        for a in anns:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            boxes.append([x, y, x + w, y + h])
            labels.append(1)
            mask_img = Image.new("L", (W, H), 0)
            poly = a["segmentation"][0]
            ImageDraw.Draw(mask_img).polygon(list(zip(poly[0::2], poly[1::2])), outline=1, fill=1)
            masks.append(torch.from_numpy(np.array(mask_img, dtype="uint8")))
        boxes_t = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        labels_t = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        masks_t = torch.stack(masks) if masks else torch.zeros((0, H, W), dtype=torch.uint8)
        target = {"boxes": boxes_t, "labels": labels_t, "masks": masks_t, "image_id": torch.tensor([img_id])}
        return torchvision.transforms.functional.to_tensor(img), target


def collate_fn(batch):
    return tuple(zip(*batch))


# --------------------------------------------------------------------------
# Modelos
# --------------------------------------------------------------------------

def build_fasterrcnn(num_classes_with_bg: int):
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_with_bg)
    return model


def build_maskrcnn(num_classes_with_bg: int):
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes_with_bg)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes_with_bg)
    return model


# --------------------------------------------------------------------------
# Treino (loop manual -- torchvision não tem `.train()` pronto como Ultralytics)
# --------------------------------------------------------------------------

def train_loop(model, dataset, device, epochs: int, batch_size: int = 4, lr: float = 1e-4, log_every: int = 20):
    model.to(device)
    model.train()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    history = []
    t0 = time.time()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for images, targets in loader:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            n_batches += 1
        avg_loss = epoch_loss / max(n_batches, 1)
        history.append({"epoch": epoch, "loss": avg_loss, "elapsed_sec": round(time.time() - t0, 1)})
        if epoch % log_every == 0 or epoch == epochs - 1:
            print(f"  epoch {epoch:4d}/{epochs}  loss={avg_loss:.4f}  elapsed={cc.human_duration(time.time() - t0)}")
    return history


# --------------------------------------------------------------------------
# Avaliação: Precisão/Recall (conf=0.15, IoU=0.5, mesmo operating point do
# resto do projeto) + mAP50/mAP50:95 via torchmetrics.
# --------------------------------------------------------------------------

def _iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


CONF_THRESHOLD = 0.15
IOU_MATCH = 0.5


@torch.no_grad()
def evaluate(model, dataset, device, eval_masks: bool = False) -> dict:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_fn)

    map_metric = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", backend="faster_coco_eval")
    mask_metric = MeanAveragePrecision(iou_type="segm", backend="faster_coco_eval") if eval_masks else None
    tp = fp = fn = 0
    class_tp: dict[int, int] = {}
    class_fp: dict[int, int] = {}
    class_fn: dict[int, int] = {}

    for images, targets in loader:
        images_dev = [img.to(device) for img in images]
        preds = model(images_dev)
        pred = preds[0]
        target = targets[0]

        keep = pred["scores"] >= CONF_THRESHOLD
        pred_boxes = pred["boxes"][keep].cpu()
        pred_labels = pred["labels"][keep].cpu()
        pred_scores = pred["scores"][keep].cpu()

        map_metric.update(
            [{"boxes": pred_boxes, "scores": pred_scores, "labels": pred_labels}],
            [{"boxes": target["boxes"], "labels": target["labels"]}],
        )

        if eval_masks and "masks" in pred:
            pred_masks = (pred["masks"][keep].cpu() > 0.5).squeeze(1)  # (N,1,H,W) soft -> (N,H,W) bool
            gt_masks = target["masks"].cpu().bool()
            mask_metric.update(
                [{"masks": pred_masks, "scores": pred_scores, "labels": pred_labels}],
                [{"masks": gt_masks, "labels": target["labels"]}],
            )

        gt_boxes = target["boxes"].tolist()
        gt_labels = target["labels"].tolist()
        p_boxes = pred_boxes.tolist()
        p_labels = pred_labels.tolist()

        matched_gt = set()
        candidates = []
        for pi, (pb, pl) in enumerate(zip(p_boxes, p_labels)):
            for gi, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                if gl != pl:
                    continue
                iou = _iou_xyxy(pb, gb)
                if iou >= IOU_MATCH:
                    candidates.append((iou, pi, gi))
        candidates.sort(reverse=True)
        matched_pred = set()
        for iou, pi, gi in candidates:
            if pi in matched_pred or gi in matched_gt:
                continue
            matched_pred.add(pi)
            matched_gt.add(gi)
        tp += len(matched_pred)
        fp += len(p_boxes) - len(matched_pred)
        fn += len(gt_boxes) - len(matched_gt)

        for pi in range(len(p_boxes)):
            lbl = p_labels[pi]
            if pi in matched_pred:
                class_tp[lbl] = class_tp.get(lbl, 0) + 1
            else:
                class_fp[lbl] = class_fp.get(lbl, 0) + 1
        for gi in range(len(gt_boxes)):
            if gi not in matched_gt:
                lbl = gt_labels[gi]
                class_fn[lbl] = class_fn.get(lbl, 0) + 1

    # Precisão/Recall macro (média simples por classe, cada classe com peso
    # igual) -- mesma convenção de metrics.box.mp/mr do Ultralytics (usada
    # por YOLOv10 e Benchnano-seg). A média micro anterior (pool de TP/FP/FN
    # de todas as classes) pesava cada *partícula* igual, não cada *classe*;
    # num dataset com PE em 0,6% do teste e PP em 42%, isso produzia números
    # não comparáveis entre famílias de arquitetura -- ver análise da seção
    # "Comparison with the MiNa benchmark study".
    classes_with_gt = sorted(set(class_tp) | set(class_fn))
    per_class = {}
    for c in classes_with_gt:
        ctp, cfp, cfn = class_tp.get(c, 0), class_fp.get(c, 0), class_fn.get(c, 0)
        p = ctp / (ctp + cfp) if (ctp + cfp) > 0 else 0.0
        r = ctp / (ctp + cfn) if (ctp + cfn) > 0 else 0.0
        per_class[c] = {"precision": p, "recall": r, "tp": ctp, "fp": cfp, "fn": cfn}
    precision_macro = sum(v["precision"] for v in per_class.values()) / len(per_class) if per_class else 0.0
    recall_macro = sum(v["recall"] for v in per_class.values()) / len(per_class) if per_class else 0.0

    precision_micro = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_micro = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    map_result = map_metric.compute()

    out = {
        "aggregate_metrics": {
            "precision": precision_macro, "recall": recall_macro,
            "map50": float(map_result["map_50"]), "map50_95": float(map_result["map"]),
            "f1": (2 * precision_macro * recall_macro / (precision_macro + recall_macro))
                  if (precision_macro + recall_macro) > 0 else 0.0,
        },
        "precision_micro": precision_micro, "recall_micro": recall_micro,
        "per_class": per_class,
        "n_tp": tp, "n_fp": fp, "n_fn": fn,
    }
    if eval_masks:
        mask_result = mask_metric.compute()
        out["mask"] = {"ap50": float(mask_result["map_50"]), "ap75": float(mask_result["map_75"]), "ap": float(mask_result["map"])}
    return out
