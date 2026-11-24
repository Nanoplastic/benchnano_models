"""Faster R-CNN / Mask R-CNN com torchvision / Faster R-CNN / Mask R-CNN with torchvision.

PT: Implementação de Faster R-CNN e Mask R-CNN com o torchvision (em vez do Detectron2, por
    incompatibilidade de versão entre o Detectron2 e o PyTorch fixado neste projeto).
    Os datasets são lidos dos JSONs COCO gerados por `00_build_seg_dataset.py` (protocolo de
    imagem inteira, classe única) e `08_build_curated_crop_dataset.py` (protocolo de
    recortes, ambos os formatos); nenhum dataset é gerado aqui.
    Limite de instâncias por imagem (`MAX_INSTANCIAS_POR_IMAGEM` em comp_common.py):
    necessário só no protocolo de imagem inteira, porque micrografias com 1000+ partículas
    estouram a VRAM no cálculo de IoU âncora x GT ("Attempting to copy inputs of
    pairwise_iou to CPU due to CUDA OOM"). Recortes 256x256 não precisam do limite.
EN: Faster R-CNN and Mask R-CNN implemented with torchvision (instead of Detectron2, due to
    a version incompatibility between Detectron2 and the PyTorch version pinned in this
    project).
    Datasets are read from the COCO JSONs produced by `00_build_seg_dataset.py` (full-image
    protocol, single class) and `08_build_curated_crop_dataset.py` (crop protocol, both
    formats); no dataset is generated here.
    Per-image instance cap (`MAX_INSTANCIAS_POR_IMAGEM` in comp_common.py): needed only in
    the full-image protocol, because micrographs with 1000+ particles exhaust VRAM in the
    anchor x GT IoU computation ("Attempting to copy inputs of pairwise_iou to CPU due to
    CUDA OOM"). 256x256 crops do not need the cap.
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


# PT: Datasets
# EN: Datasets

class ConjuntoCocoDeteccao(Dataset):
    """PT: bbox multiclasse (category_id 1..N no COCO -> rótulo 1..N no torch; 0 é sempre
        fundo, convenção do torchvision).
    EN: multi-class bbox (category_id 1..N in COCO -> label 1..N in torch; 0 is always
        background, torchvision convention).
    """

    def __init__(self, coco_json: Path, images_dir: Path, max_instances: int | None = None):
        with open(coco_json, encoding="utf-8") as f:
            d = json.load(f)
        self.images_dir = images_dir
        self.images = {im["id"]: im for im in d["images"]}
        self.by_image = {}
        for anot in d["annotations"]:
            self.by_image.setdefault(anot["image_id"], []).append(anot)
        self.ids = list(self.images.keys())
        self.max_instances = max_instances

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_img = self.ids[idx]
        meta_img = self.images[id_img]
        img = Image.open(self.images_dir / meta_img["file_name"]).convert("RGB")
        anots = self.by_image.get(id_img, [])
        if self.max_instances and len(anots) > self.max_instances:
            anots = random.sample(anots, self.max_instances)
        caixas, rotulos = [], []
        for a in anots:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            caixas.append([x, y, x + w, y + h])
            rotulos.append(a["category_id"])  # PT: já começa em 1 no COCO gerado por este projeto / EN: already 1-based in the COCO produced by this project
        caixas_t = torch.as_tensor(caixas, dtype=torch.float32) if caixas else torch.zeros((0, 4), dtype=torch.float32)
        rotulos_t = torch.as_tensor(rotulos, dtype=torch.int64) if rotulos else torch.zeros((0,), dtype=torch.int64)
        alvo = {"boxes": caixas_t, "labels": rotulos_t, "image_id": torch.tensor([id_img])}
        return torchvision.transforms.functional.to_tensor(img), alvo


class ConjuntoCocoSegmentacao(Dataset):
    """PT: polígono de classe única (category_id sempre 1 -> rótulo 1).
    EN: single-class polygon (category_id always 1 -> label 1).
    """

    def __init__(self, coco_json: Path, images_dir: Path, max_instances: int | None = None):
        with open(coco_json, encoding="utf-8") as f:
            d = json.load(f)
        self.images_dir = images_dir
        self.images = {im["id"]: im for im in d["images"]}
        self.by_image = {}
        for anot in d["annotations"]:
            self.by_image.setdefault(anot["image_id"], []).append(anot)
        self.ids = list(self.images.keys())
        self.max_instances = max_instances

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_img = self.ids[idx]
        meta_img = self.images[id_img]
        img = Image.open(self.images_dir / meta_img["file_name"]).convert("RGB")
        W, H = meta_img["width"], meta_img["height"]
        anots = self.by_image.get(id_img, [])
        if self.max_instances and len(anots) > self.max_instances:
            anots = random.sample(anots, self.max_instances)
        caixas, rotulos, mascaras = [], [], []
        for a in anots:
            x, y, w, h = a["bbox"]
            if w <= 0 or h <= 0:
                continue
            caixas.append([x, y, x + w, y + h])
            rotulos.append(1)
            img_mascara = Image.new("L", (W, H), 0)
            poly = a["segmentation"][0]
            ImageDraw.Draw(img_mascara).polygon(list(zip(poly[0::2], poly[1::2])), outline=1, fill=1)
            mascaras.append(torch.from_numpy(np.array(img_mascara, dtype="uint8")))
        caixas_t = torch.as_tensor(caixas, dtype=torch.float32) if caixas else torch.zeros((0, 4), dtype=torch.float32)
        rotulos_t = torch.as_tensor(rotulos, dtype=torch.int64) if rotulos else torch.zeros((0,), dtype=torch.int64)
        mascaras_t = torch.stack(mascaras) if mascaras else torch.zeros((0, H, W), dtype=torch.uint8)
        alvo = {"boxes": caixas_t, "labels": rotulos_t, "masks": mascaras_t, "image_id": torch.tensor([id_img])}
        return torchvision.transforms.functional.to_tensor(img), alvo


def agrupar_lote(lote):
    return tuple(zip(*lote))


# PT: Modelos
# EN: Models

def construir_fasterrcnn(num_classes_com_fundo: int):
    modelo = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    caracteristicas_entrada = modelo.roi_heads.box_predictor.cls_score.in_features
    modelo.roi_heads.box_predictor = FastRCNNPredictor(caracteristicas_entrada, num_classes_com_fundo)
    return modelo


def construir_maskrcnn(num_classes_com_fundo: int):
    modelo = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.DEFAULT)
    caracteristicas_entrada = modelo.roi_heads.box_predictor.cls_score.in_features
    modelo.roi_heads.box_predictor = FastRCNNPredictor(caracteristicas_entrada, num_classes_com_fundo)
    caracteristicas_entrada_mascara = modelo.roi_heads.mask_predictor.conv5_mask.in_channels
    modelo.roi_heads.mask_predictor = MaskRCNNPredictor(caracteristicas_entrada_mascara, 256, num_classes_com_fundo)
    return modelo


# PT: Treino (laço manual; o torchvision não tem um `.train()` pronto como o Ultralytics)
# EN: Training (manual loop; torchvision has no ready-made `.train()` like Ultralytics)

def laco_treino(modelo, conjunto, dispositivo, epocas: int, tamanho_lote: int = 4, lr: float = 1e-4, registrar_a_cada: int = 20):
    modelo.to(dispositivo)
    modelo.train()
    carregador = DataLoader(conjunto, batch_size=tamanho_lote, shuffle=True, num_workers=4, collate_fn=agrupar_lote)
    otimizador = torch.optim.AdamW([p for p in modelo.parameters() if p.requires_grad], lr=lr)
    historico = []
    t0 = time.time()
    for epoca in range(epocas):
        perda_epoca = 0.0
        n_lotes = 0
        for imagens, alvos in carregador:
            imagens = [img.to(dispositivo) for img in imagens]
            alvos = [{k: v.to(dispositivo) for k, v in t.items()} for t in alvos]
            dict_perdas = modelo(imagens, alvos)
            perda = sum(dict_perdas.values())
            otimizador.zero_grad()
            perda.backward()
            otimizador.step()
            perda_epoca += float(perda.item())
            n_lotes += 1
        perda_media = perda_epoca / max(n_lotes, 1)
        historico.append({"epoch": epoca, "loss": perda_media, "elapsed_sec": round(time.time() - t0, 1)})
        if epoca % registrar_a_cada == 0 or epoca == epocas - 1:
            print(f"  epoch {epoca:4d}/{epocas}  loss={perda_media:.4f}  elapsed={cc.duracao_legivel(time.time() - t0)}")
    return historico


# PT: Avaliação: precisão/recall (conf=0,15, IoU=0,5, o mesmo ponto de operação do resto do
#     projeto) + mAP50/mAP50:95 via torchmetrics.
# EN: Evaluation: precision/recall (conf=0.15, IoU=0.5, the same operating point as the rest
#     of the project) + mAP50/mAP50:95 via torchmetrics.

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


LIMIAR_CONF = 0.15
IOU_PAREAMENTO = 0.5


@torch.no_grad()
def avaliar(modelo, conjunto, dispositivo, avaliar_mascaras: bool = False) -> dict:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision

    modelo.to(dispositivo)
    modelo.eval()
    carregador = DataLoader(conjunto, batch_size=1, shuffle=False, num_workers=2, collate_fn=agrupar_lote)

    metrica_map = MeanAveragePrecision(box_format="xyxy", iou_type="bbox", backend="faster_coco_eval")
    metrica_mascara = MeanAveragePrecision(iou_type="segm", backend="faster_coco_eval") if avaliar_mascaras else None
    tp = fp = fn = 0
    tp_classe: dict[int, int] = {}
    fp_classe: dict[int, int] = {}
    fn_classe: dict[int, int] = {}

    for imagens, alvos in carregador:
        imagens_dev = [img.to(dispositivo) for img in imagens]
        preds = modelo(imagens_dev)
        pred = preds[0]
        alvo = alvos[0]

        manter = pred["scores"] >= LIMIAR_CONF
        caixas_pred = pred["boxes"][manter].cpu()
        rotulos_pred = pred["labels"][manter].cpu()
        pontuacoes_pred = pred["scores"][manter].cpu()

        metrica_map.update(
            [{"boxes": caixas_pred, "scores": pontuacoes_pred, "labels": rotulos_pred}],
            [{"boxes": alvo["boxes"], "labels": alvo["labels"]}],
        )

        if avaliar_mascaras and "masks" in pred:
            mascaras_pred = (pred["masks"][manter].cpu() > 0.5).squeeze(1)  # PT: (N,1,H,W) contínua -> (N,H,W) booleana / EN: (N,1,H,W) soft -> (N,H,W) bool
            mascaras_gt = alvo["masks"].cpu().bool()
            metrica_mascara.update(
                [{"masks": mascaras_pred, "scores": pontuacoes_pred, "labels": rotulos_pred}],
                [{"masks": mascaras_gt, "labels": alvo["labels"]}],
            )

        caixas_gt = alvo["boxes"].tolist()
        rotulos_gt = alvo["labels"].tolist()
        caixas_p = caixas_pred.tolist()
        rotulos_p = rotulos_pred.tolist()

        gt_pareado = set()
        candidatas = []
        for pi, (pb, pl) in enumerate(zip(caixas_p, rotulos_p)):
            for gi, (gb, gl) in enumerate(zip(caixas_gt, rotulos_gt)):
                if gl != pl:
                    continue
                iou = _iou_xyxy_interno(pb, gb)
                if iou >= IOU_PAREAMENTO:
                    candidatas.append((iou, pi, gi))
        candidatas.sort(reverse=True)
        pred_pareada = set()
        for iou, pi, gi in candidatas:
            if pi in pred_pareada or gi in gt_pareado:
                continue
            pred_pareada.add(pi)
            gt_pareado.add(gi)
        tp += len(pred_pareada)
        fp += len(caixas_p) - len(pred_pareada)
        fn += len(caixas_gt) - len(gt_pareado)

        for pi in range(len(caixas_p)):
            rot = rotulos_p[pi]
            if pi in pred_pareada:
                tp_classe[rot] = tp_classe.get(rot, 0) + 1
            else:
                fp_classe[rot] = fp_classe.get(rot, 0) + 1
        for gi in range(len(caixas_gt)):
            if gi not in gt_pareado:
                rot = rotulos_gt[gi]
                fn_classe[rot] = fn_classe.get(rot, 0) + 1

    # PT: Precisão/recall macro (média simples por classe, cada classe com o mesmo peso),
    #     mesma convenção de metrics.box.mp/mr do Ultralytics (usada por YOLOv10 e
    #     Benchnano-seg). A média micro (soma de TP/FP/FN de todas as classes) pesa cada
    #     partícula igualmente; com PE em 0,6% do teste e PP em 42%, os números não seriam
    #     comparáveis entre famílias de arquitetura.
    # EN: Macro precision/recall (simple per-class mean, every class weighted equally), same
    #     convention as Ultralytics metrics.box.mp/mr (used by YOLOv10 and Benchnano-seg).
    #     The micro average (pooled TP/FP/FN over all classes) weights every particle
    #     equally; with PE at 0.6% of the test set and PP at 42%, the numbers would not be
    #     comparable across architecture families.
    classes_com_gt = sorted(set(tp_classe) | set(fn_classe))
    dados_por_classe = {}
    for c in classes_com_gt:
        ctp, cfp, cfn = tp_classe.get(c, 0), fp_classe.get(c, 0), fn_classe.get(c, 0)
        p = ctp / (ctp + cfp) if (ctp + cfp) > 0 else 0.0
        r = ctp / (ctp + cfn) if (ctp + cfn) > 0 else 0.0
        dados_por_classe[c] = {"precision": p, "recall": r, "tp": ctp, "fp": cfp, "fn": cfn}
    precisao_macro = sum(v["precision"] for v in dados_por_classe.values()) / len(dados_por_classe) if dados_por_classe else 0.0
    recall_macro = sum(v["recall"] for v in dados_por_classe.values()) / len(dados_por_classe) if dados_por_classe else 0.0

    precisao_micro = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_micro = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    resultado_map = metrica_map.compute()

    saida = {
        "aggregate_metrics": {
            "precision": precisao_macro, "recall": recall_macro,
            "map50": float(resultado_map["map_50"]), "map50_95": float(resultado_map["map"]),
            "f1": (2 * precisao_macro * recall_macro / (precisao_macro + recall_macro))
                  if (precisao_macro + recall_macro) > 0 else 0.0,
        },
        "precision_micro": precisao_micro, "recall_micro": recall_micro,
        "per_class": dados_por_classe,
        "n_tp": tp, "n_fp": fp, "n_fn": fn,
    }
    if avaliar_mascaras:
        resultado_mascara = metrica_mascara.compute()
        saida["mask"] = {"ap50": float(resultado_mascara["map_50"]), "ap75": float(resultado_mascara["map_75"]), "ap": float(resultado_mascara["map"])}
    return saida
