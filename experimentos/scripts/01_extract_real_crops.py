"""Extrai, a partir das micrografias completas do MiNa + JSON COCO:

1. Recortes de classificação (`data/real_crops/{split}/{class}/*.png`):
   bbox + margem de 10px, RGB, contexto local real preservado (não é uma
   silhueta isolada -- é o que o VLM vai ver de fato).
2. Templates de forma para síntese (`data/shapes/{class}/*.png`), RGBA com
   alpha = polígono de segmentação COCO rasterizado, margem 4px -- SÓ a
   partir de imagens do split `train`, pra nunca vazar textura de
   imagens de val/teste para o gerador sintético (script 03).

A faixa de legenda gravada no rodapé de cada micrografia (kV/magnificação/
barra de escala, medida visualmente e por detecção de pixel branco na
sessão anterior -- começa por volta de y=830 numa imagem de 960px de altura)
é excluída dos dois tipos de recorte.
"""
import csv
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
FULL_ROOT = ROOT / "MPDataset" / "Full_Images" / "COCO Format"
EXPERIMENT_DIR = ROOT / "resultados" / "01_extract_real_crops"
SPLITS_CSV = ROOT / "resultados" / "00_split_images" / "data" / "image_splits.csv"
CROPS_OUT = EXPERIMENT_DIR / "data" / "real_crops"
SHAPES_OUT = EXPERIMENT_DIR / "data" / "shapes"
MANIFEST_OUT = EXPERIMENT_DIR / "data" / "manifest_real.csv"

CAPTION_TOP = 830  # acima disso é a barra de legenda gravada, não ciência
CROP_MARGIN = 10
SHAPE_MARGIN = 4
MIN_SIDE = 8
CAPS = {"train": 250, "val": 60, "test": 60}
SEED = 42


def load_coco(cls: str):
    import json
    d = json.load(open(FULL_ROOT / cls / f"{cls}_COCO.json"))
    anns_by_image = {}
    for a in d["annotations"]:
        anns_by_image.setdefault(a["image_id"], []).append(a)
    return anns_by_image


def clip_box(x0, y0, x1, y1, w, h):
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def extract_crop(img_arr, ann, margin, w, h):
    x, y, bw, bh = ann["bbox"]
    x0, y0, x1, y1 = x - margin, y - margin, x + bw + margin, y + bh + margin
    x0, y0, x1, y1 = clip_box(x0, y0, x1, y1, w, min(h, CAPTION_TOP))
    if (x1 - x0) < MIN_SIDE or (y1 - y0) < MIN_SIDE:
        return None
    return int(x0), int(y0), int(x1), int(y1)


def rasterize_mask(segmentation, x0, y0, w, h):
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    for poly in segmentation:
        pts = [(poly[i] - x0, poly[i + 1] - y0) for i in range(0, len(poly), 2)]
        if len(pts) >= 3:
            draw.polygon(pts, fill=255)
    return np.array(mask) > 0


def main():
    splits_df = pd.read_csv(SPLITS_CSV)
    rng = random.Random(SEED)

    manifest_rows = []
    shape_counts = {}

    for cls in ["PE", "PET", "PP", "PS"]:
        anns_by_image = load_coco(cls)
        cls_splits = splits_df[splits_df["class"] == cls]

        crop_candidates = {"train": [], "val": [], "test": []}
        n_shapes = 0

        for _, row in cls_splits.iterrows():
            split = row["split"]
            img_path = FULL_ROOT / cls / row["file_name"]
            img = Image.open(img_path).convert("RGB")
            img_arr = np.array(img)
            w, h = img.size
            anns = anns_by_image.get(row["image_id"], [])

            for ann in anns:
                box = extract_crop(img_arr, ann, CROP_MARGIN, w, h)
                if box is None:
                    continue
                x0, y0, x1, y1 = box
                crop_candidates[split].append((img, x0, y0, x1, y1, row["file_name"], ann["id"]))

                if split == "train":
                    sx0, sy0, sx1, sy1 = extract_crop(img_arr, ann, SHAPE_MARGIN, w, h) or (None,) * 4
                    if sx0 is None:
                        continue
                    mask = rasterize_mask(ann["segmentation"], sx0, sy0, sx1 - sx0, sy1 - sy0)
                    if mask.sum() < 20:
                        continue
                    rgb_crop = img_arr[sy0:sy1, sx0:sx1]
                    rgba = np.dstack([rgb_crop, (mask * 255).astype("uint8")])
                    out_dir = SHAPES_OUT / cls
                    out_dir.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(rgba, mode="RGBA").save(
                        out_dir / f"{row['file_name'].rsplit('.', 1)[0]}_{ann['id']}.png"
                    )
                    n_shapes += 1

        shape_counts[cls] = n_shapes

        for split, cap in CAPS.items():
            candidates = crop_candidates[split]
            rng.shuffle(candidates)
            selected = candidates[:cap]
            out_dir = CROPS_OUT / split / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for img, x0, y0, x1, y1, fname, ann_id in selected:
                crop = img.crop((x0, y0, x1, y1))
                out_name = f"{fname.rsplit('.', 1)[0]}_{ann_id}.png"
                crop.save(out_dir / out_name)
                manifest_rows.append({
                    "image_path": str((out_dir / out_name).relative_to(ROOT)),
                    "label": cls, "split": split, "source_image": fname,
                })
            print(f"{cls}/{split}: {len(selected)}/{len(candidates)} recortes salvos")

    MANIFEST_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST_OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "label", "split", "source_image"])
        w.writeheader()
        w.writerows(manifest_rows)
    print(f"\nmanifest real -> {MANIFEST_OUT} ({len(manifest_rows)} recortes)")
    print(f"shape templates (só train) por classe: {shape_counts}")


if __name__ == "__main__":
    main()
