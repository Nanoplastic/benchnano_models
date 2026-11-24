"""Extração de blocos de fundo / Background tile extraction.

PT: Extrai blocos de textura de fundo reais (substrato/membrana filtrante) só das imagens da
    partição `train` de cada classe, excluindo a faixa de legenda do rodapé e qualquer bloco
    que sobreponha uma bbox anotada (evita "fundo limpo" que contenha partícula não
    mascarada).
    Tamanho de bloco adaptativo: se uma classe não atinge MIN_BLOCOS_ALVO no tamanho padrão
    (384 px, passo 192 px), tenta tamanhos progressivamente menores (sempre com passo =
    bloco/2) até atingir o piso ou esgotar a lista. Blocos menores cabem entre partículas
    densas (caso do PS) sem relaxar o critério de sobreposição (MAX_SOBREPOSICAO_ANOT é o
    mesmo para todos os tamanhos).
EN: Extracts real background texture tiles (substrate/filter membrane) only from the `train`
    images of each class, excluding the caption band at the bottom and any tile that
    overlaps an annotated bbox (avoids a "clean background" that contains an unmasked
    particle).
    Adaptive tile size: if a class does not reach MIN_BLOCOS_ALVO at the default size (384
    px, stride 192 px), progressively smaller sizes are tried (always stride = tile/2) until
    the floor is reached or the list is exhausted. Smaller tiles fit between dense particles
    (the PS case) without relaxing the overlap criterion (MAX_SOBREPOSICAO_ANOT is the same
    for every size).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

RAIZ = Path(__file__).resolve().parent.parent
RAIZ_COMPLETA = RAIZ / "MPDataset" / "Full_Images" / "COCO Format"
DIR_EXPERIMENTO_BASE = RAIZ / "resultados" / "02_extract_backgrounds"
CSV_PARTICOES = RAIZ / "resultados" / "00_split_images" / "data" / "image_splits.csv"
SAIDA = DIR_EXPERIMENTO_BASE / "data" / "backgrounds"

TOPO_LEGENDA = 830
TAMANHOS_BLOCO = [384, 256, 192, 128]  # PT: tentados em ordem até atingir MIN_BLOCOS_ALVO / EN: tried in order until MIN_BLOCOS_ALVO is reached
MIN_BLOCOS_ALVO = 150
MAX_SOBREPOSICAO_ANOT = 0.10


def area_sobreposicao_caixa(bx0, by0, bx1, by1, tx0, ty0, tx1, ty1):
    ix0, iy0 = max(bx0, tx0), max(by0, ty0)
    ix1, iy1 = min(bx1, tx1), min(by1, ty1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    return (ix1 - ix0) * (iy1 - iy0)


def extrair_blocos(cls, linhas_tab, anots_por_imagem, dir_saida, bloco_img):
    passo = bloco_img // 2
    area_bloco = bloco_img * bloco_img
    n = 0
    for _, linha_tab in linhas_tab.iterrows():
        img = Image.open(RAIZ_COMPLETA / cls / linha_tab["file_name"]).convert("L")
        matriz = np.array(img)
        caixas = anots_por_imagem.get(linha_tab["image_id"], [])
        caixa_xyxy = [(x, y, x + w, y + h) for x, y, w, h in caixas]

        for y in range(0, TOPO_LEGENDA - bloco_img, passo):
            for x in range(0, matriz.shape[1] - bloco_img, passo):
                t = matriz[y:y + bloco_img, x:x + bloco_img]
                if t.std() < 8 or t.mean() > 245 or t.mean() < 10:
                    continue
                sobreposicao = sum(
                    area_sobreposicao_caixa(bx0, by0, bx1, by1, x, y, x + bloco_img, y + bloco_img)
                    for bx0, by0, bx1, by1 in caixa_xyxy
                )
                if sobreposicao / area_bloco > MAX_SOBREPOSICAO_ANOT:
                    continue
                nome_saida = f"{linha_tab['file_name'].rsplit('.', 1)[0]}_{bloco_img}_{y}_{x}.png"
                Image.fromarray(t).save(dir_saida / nome_saida)
                n += 1
    return n


def principal():
    df_particoes = pd.read_csv(CSV_PARTICOES)

    for cls in ["PE", "PET", "PP", "PS"]:
        d = json.load(open(RAIZ_COMPLETA / cls / f"{cls}_COCO.json"))
        anots_por_imagem = {}
        for a in d["annotations"]:
            anots_por_imagem.setdefault(a["image_id"], []).append(a["bbox"])

        dir_saida = SAIDA / cls
        dir_saida.mkdir(parents=True, exist_ok=True)
        for f in dir_saida.glob("*.png"):
            f.unlink()

        linhas_tab = df_particoes[(df_particoes["class"] == cls) & (df_particoes["split"] == "train")]

        n = 0
        tentados = []
        for bloco_img in TAMANHOS_BLOCO:
            n = extrair_blocos(cls, linhas_tab, anots_por_imagem, dir_saida, bloco_img)
            tentados.append(f"{bloco_img}px->{n}")
            if n >= MIN_BLOCOS_ALVO:
                break
            for f in dir_saida.glob("*.png"):
                f.unlink()
        print(f"{cls}: {n} tiles de fundo -> {dir_saida}  (tentativas: {', '.join(tentados)})")


if __name__ == "__main__":
    principal()
