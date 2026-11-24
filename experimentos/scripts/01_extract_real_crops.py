"""Extração de recortes reais / Real crop extraction.

PT: A partir das micrografias completas do MiNa e do JSON COCO, gera:
    1. Recortes de classificação (`data/real_crops/{split}/{class}/*.png`): bbox + margem de
    10 px, RGB, preservando o contexto local real (é o que o VLM recebe).
    2. Moldes de forma para síntese (`data/shapes/{class}/*.png`): RGBA com alfa = polígono
    de segmentação COCO rasterizado, margem de 4 px, extraídos só de imagens da partição
    `train` para não vazar textura de val/teste para o gerador sintético (script 03).
    A faixa de legenda no rodapé de cada micrografia (kV, ampliação, barra de escala; começa
    por volta de y=830 numa imagem de 960 px de altura) é excluída dos dois tipos de
    recorte.
EN: From the full MiNa micrographs and the COCO JSON, produces:
    1. Classification crops (`data/real_crops/{split}/{class}/*.png`): bbox + 10 px margin,
    RGB, keeping the real local context (this is what the VLM receives).
    2. Shape templates for synthesis (`data/shapes/{class}/*.png`): RGBA with alpha =
    rasterized COCO segmentation polygon, 4 px margin, taken only from `train` images so
    that val/test texture never leaks into the synthetic generator (script 03).
    The caption band at the bottom of each micrograph (kV, magnification, scale bar; starts
    around y=830 in a 960 px tall image) is excluded from both crop types.
"""
import csv
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

RAIZ = Path(__file__).resolve().parent.parent
RAIZ_COMPLETA = RAIZ / "MPDataset" / "Full_Images" / "COCO Format"
DIR_EXPERIMENTO_BASE = RAIZ / "resultados" / "01_extract_real_crops"
CSV_PARTICOES = RAIZ / "resultados" / "00_split_images" / "data" / "image_splits.csv"
SAIDA_RECORTES = DIR_EXPERIMENTO_BASE / "data" / "real_crops"
SAIDA_FORMAS = DIR_EXPERIMENTO_BASE / "data" / "shapes"
SAIDA_MANIFESTO = DIR_EXPERIMENTO_BASE / "data" / "manifest_real.csv"

TOPO_LEGENDA = 830  # PT: abaixo desta linha fica a faixa de legenda, não a amostra / EN: below this row is the caption band, not the sample
MARGEM_RECORTE = 10
MARGEM_FORMA = 4
LADO_MIN = 8
LEGENDAS = {"train": 250, "val": 60, "test": 60}
SEMENTE = 42


def carregar_coco(cls: str):
    import json
    d = json.load(open(RAIZ_COMPLETA / cls / f"{cls}_COCO.json"))
    anots_por_imagem = {}
    for a in d["annotations"]:
        anots_por_imagem.setdefault(a["image_id"], []).append(a)
    return anots_por_imagem


def recortar_caixa(x0, y0, x1, y1, w, h):
    return max(0, x0), max(0, y0), min(w, x1), min(h, y1)


def extrair_recorte(matriz_img, anot, margem, w, h):
    x, y, bw, bh = anot["bbox"]
    x0, y0, x1, y1 = x - margem, y - margem, x + bw + margem, y + bh + margem
    x0, y0, x1, y1 = recortar_caixa(x0, y0, x1, y1, w, min(h, TOPO_LEGENDA))
    if (x1 - x0) < LADO_MIN or (y1 - y0) < LADO_MIN:
        return None
    return int(x0), int(y0), int(x1), int(y1)


def rasterizar_mascara(segmentacao, x0, y0, w, h):
    mascara = Image.new("L", (w, h), 0)
    desenho = ImageDraw.Draw(mascara)
    for poly in segmentacao:
        pts = [(poly[i] - x0, poly[i + 1] - y0) for i in range(0, len(poly), 2)]
        if len(pts) >= 3:
            desenho.polygon(pts, fill=255)
    return np.array(mascara) > 0


def principal():
    df_particoes = pd.read_csv(CSV_PARTICOES)
    gerador = random.Random(SEMENTE)

    linhas_manifesto = []
    contagens_formas = {}

    for cls in ["PE", "PET", "PP", "PS"]:
        anots_por_imagem = carregar_coco(cls)
        particoes_cls = df_particoes[df_particoes["class"] == cls]

        recortes_candidatos = {"train": [], "val": [], "test": []}
        n_formas = 0

        for _, linha_tab in particoes_cls.iterrows():
            particao = linha_tab["split"]
            caminho_img = RAIZ_COMPLETA / cls / linha_tab["file_name"]
            img = Image.open(caminho_img).convert("RGB")
            matriz_img = np.array(img)
            w, h = img.size
            anots = anots_por_imagem.get(linha_tab["image_id"], [])

            for anot in anots:
                caixa = extrair_recorte(matriz_img, anot, MARGEM_RECORTE, w, h)
                if caixa is None:
                    continue
                x0, y0, x1, y1 = caixa
                recortes_candidatos[particao].append((img, x0, y0, x1, y1, linha_tab["file_name"], anot["id"]))

                if particao == "train":
                    sx0, sy0, sx1, sy1 = extrair_recorte(matriz_img, anot, MARGEM_FORMA, w, h) or (None,) * 4
                    if sx0 is None:
                        continue
                    mascara = rasterizar_mascara(anot["segmentation"], sx0, sy0, sx1 - sx0, sy1 - sy0)
                    if mascara.sum() < 20:
                        continue
                    recorte_rgb = matriz_img[sy0:sy1, sx0:sx1]
                    rgba = np.dstack([recorte_rgb, (mascara * 255).astype("uint8")])
                    dir_saida = SAIDA_FORMAS / cls
                    dir_saida.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(rgba, mode="RGBA").save(
                        dir_saida / f"{linha_tab['file_name'].rsplit('.', 1)[0]}_{anot['id']}.png"
                    )
                    n_formas += 1

        contagens_formas[cls] = n_formas

        for particao, legenda_cap in LEGENDAS.items():
            candidatas = recortes_candidatos[particao]
            gerador.shuffle(candidatas)
            selecionados = candidatas[:legenda_cap]
            dir_saida = SAIDA_RECORTES / particao / cls
            dir_saida.mkdir(parents=True, exist_ok=True)
            for img, x0, y0, x1, y1, nome_arq, id_anot in selecionados:
                recorte = img.crop((x0, y0, x1, y1))
                nome_saida = f"{nome_arq.rsplit('.', 1)[0]}_{id_anot}.png"
                recorte.save(dir_saida / nome_saida)
                linhas_manifesto.append({
                    "image_path": str((dir_saida / nome_saida).relative_to(RAIZ)),
                    "label": cls, "split": particao, "source_image": nome_arq,
                })
            print(f"{cls}/{particao}: {len(selecionados)}/{len(candidatas)} recortes salvos")

    SAIDA_MANIFESTO.parent.mkdir(parents=True, exist_ok=True)
    with open(SAIDA_MANIFESTO, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "label", "split", "source_image"])
        w.writeheader()
        w.writerows(linhas_manifesto)
    print(f"\nmanifest real -> {SAIDA_MANIFESTO} ({len(linhas_manifesto)} recortes)")
    print(f"shape templates (só train) por classe: {contagens_formas}")


if __name__ == "__main__":
    principal()
