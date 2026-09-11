"""Extrai tiles de textura de fundo (substrato/membrana filtrante) reais,
só das imagens do split `train` de cada classe, excluindo a faixa de
legenda gravada no rodapé e qualquer tile que sobreponha uma bbox anotada
(evita "fundo limpo" que na verdade tem partícula não mascarada -- checagem
que a linha anterior (`tarefa_vlm_nanoplastic_fewshot`) não precisava fazer,
porque lá o fundo vinha de regiões do Moon et al. sem anotação disponível
de qualquer forma).

Correção (Experimento #2 do plano, plano_experimentos_vlm.md): a extração
original usava um único tamanho de tile (384px, stride 192px) para todas
as classes, o que deixou PS com só 33 tiles limpos contra 196-254 nas
outras -- PS tem amostras muito mais densas de partícula por imagem, então
poucas regiões de 384px ficam livres de sobreposição. Em vez de tratar PS
como caso especial, a regra abaixo é geral: se uma classe não atinge
MIN_TILES_TARGET no tamanho padrão, tenta tamanhos de tile progressivamente
menores (mesma proporção stride=tile/2) até atingir o piso ou esgotar a
lista -- tiles menores cabem nos espaços livres entre partículas densas
sem relaxar o critério de sobreposição (MAX_ANN_OVERLAP continua o mesmo).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
FULL_ROOT = ROOT / "MPDataset" / "Full_Images" / "COCO Format"
EXPERIMENT_DIR = ROOT / "resultados" / "02_extract_backgrounds"
SPLITS_CSV = ROOT / "resultados" / "00_split_images" / "data" / "image_splits.csv"
OUT = EXPERIMENT_DIR / "data" / "backgrounds"

CAPTION_TOP = 830
TILE_SIZES = [384, 256, 192, 128]  # tentados em ordem até atingir MIN_TILES_TARGET
MIN_TILES_TARGET = 150
MAX_ANN_OVERLAP = 0.10


def box_overlap_area(bx0, by0, bx1, by1, tx0, ty0, tx1, ty1):
    ix0, iy0 = max(bx0, tx0), max(by0, ty0)
    ix1, iy1 = min(bx1, tx1), min(by1, ty1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def extract_tiles(cls, rows, anns_by_image, out_dir, tile):
    stride = tile // 2
    tile_area = tile * tile
    n = 0
    for _, row in rows.iterrows():
        img = Image.open(FULL_ROOT / cls / row["file_name"]).convert("L")
        arr = np.array(img)
        boxes = anns_by_image.get(row["image_id"], [])
        box_xyxy = [(x, y, x + w, y + h) for x, y, w, h in boxes]

        for y in range(0, CAPTION_TOP - tile, stride):
            for x in range(0, arr.shape[1] - tile, stride):
                t = arr[y:y + tile, x:x + tile]
                if t.std() < 8 or t.mean() > 245 or t.mean() < 10:
                    continue
                overlap = sum(
                    box_overlap_area(bx0, by0, bx1, by1, x, y, x + tile, y + tile)
                    for bx0, by0, bx1, by1 in box_xyxy
                )
                if overlap / tile_area > MAX_ANN_OVERLAP:
                    continue
                out_name = f"{row['file_name'].rsplit('.', 1)[0]}_{tile}_{y}_{x}.png"
                Image.fromarray(t).save(out_dir / out_name)
                n += 1
    return n


def main():
    splits_df = pd.read_csv(SPLITS_CSV)

    for cls in ["PE", "PET", "PP", "PS"]:
        d = json.load(open(FULL_ROOT / cls / f"{cls}_COCO.json"))
        anns_by_image = {}
        for a in d["annotations"]:
            anns_by_image.setdefault(a["image_id"], []).append(a["bbox"])

        out_dir = OUT / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in out_dir.glob("*.png"):
            f.unlink()

        rows = splits_df[(splits_df["class"] == cls) & (splits_df["split"] == "train")]

        n = 0
        tried = []
        for tile in TILE_SIZES:
            n = extract_tiles(cls, rows, anns_by_image, out_dir, tile)
            tried.append(f"{tile}px->{n}")
            if n >= MIN_TILES_TARGET:
                break
            for f in out_dir.glob("*.png"):
                f.unlink()
        print(f"{cls}: {n} tiles de fundo -> {out_dir}  (tentativas: {', '.join(tried)})")


if __name__ == "__main__":
    main()
