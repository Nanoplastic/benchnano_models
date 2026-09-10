#!/usr/bin/env python3
"""
Desenha, sobre uma imagem do MPDataset, a segmentacao de cada particula
anotada (COCO) com o valor medido em micrometros e a classificacao
nano/microplastico - no mesmo estilo da figura de referencia recebida
(caixa + rotulo colorido: vermelho = nanoplastico, verde = microplastico).

Reaproveita a calibracao (leitura do rotulo do MEV) e o calculo de tamanho
de medir_particulas.py - nada de logica duplicada.

Uso:
    # uma imagem so
    python3 desenhar_segmentacao.py --polimero PET --arquivo "PS-2-5000X.png"

    # o dataset inteiro (todas as imagens de PE/PET/PP/PS)
    python3 desenhar_segmentacao.py --todas
    python3 desenhar_segmentacao.py --todas --saida-dir "MPDataset/Full_Images/Segmentado"
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from medir_particulas import (
    area_poligono,
    calibrar_imagem,
    classificar,
)

COR_NANO = (255, 40, 40)
COR_MICRO = (40, 220, 40)
COR_FORA = (60, 140, 255)

POLIMEROS = ["PE", "PET", "PP", "PS"]


def cor_para_classe(classe: str) -> tuple[int, int, int]:
    return {"nanoplastico": COR_NANO, "microplastico": COR_MICRO}.get(classe, COR_FORA)


def _fonte() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", 13)
    except OSError:
        return ImageFont.load_default()


def desenhar_imagem(
    caminho_imagem: Path,
    nome_arquivo: str,
    anotacoes: list[dict],
    fonte: ImageFont.ImageFont,
) -> tuple[Image.Image, dict[str, int], str]:
    """Calibra pela imagem, desenha caixa+rotulo para cada anotacao.
    Retorna (imagem_desenhada, contagem_por_classe, aviso_ou_erro)."""
    calib = calibrar_imagem(caminho_imagem, nome_arquivo)
    if isinstance(calib, str):
        return None, {}, calib
    ampliacao, escala, aviso = calib

    im = Image.open(caminho_imagem).convert("RGB")
    draw = ImageDraw.Draw(im)
    contagem = {"nanoplastico": 0, "microplastico": 0, "fora_da_faixa": 0}

    for ann in anotacoes:
        seg = ann.get("segmentation")
        bbox = ann["bbox"]  # [x, y, w, h] em pixels
        if seg and len(seg) and len(seg[0]) >= 6:
            area_px2 = area_poligono(seg[0])
            diam_usado = 2 * np.sqrt((area_px2 * escala ** 2) / np.pi)
        else:
            diam_usado = max(bbox[2], bbox[3]) * escala

        classe = classificar(diam_usado)
        contagem[classe] += 1
        cor = cor_para_classe(classe)

        x, y, w, h = bbox
        draw.rectangle([x, y, x + w, y + h], outline=cor, width=1)
        rotulo = f"{diam_usado:.2f}um"
        tx, ty = x, max(0, y - 13)
        bbox_texto = draw.textbbox((tx, ty), rotulo, font=fonte)
        draw.rectangle(bbox_texto, fill=(0, 0, 0))
        draw.text((tx, ty), rotulo, fill=cor, font=fonte)

    legenda_y = 6
    draw.rectangle([6, legenda_y, 20, legenda_y + 12], fill=COR_NANO)
    draw.text((24, legenda_y - 1), "nanoplastico (<1um)", fill=COR_NANO, font=fonte)
    draw.rectangle([6, legenda_y + 16, 20, legenda_y + 28], fill=COR_MICRO)
    draw.text((24, legenda_y + 15), "microplastico (>=1um)", fill=COR_MICRO, font=fonte)

    return im, contagem, f"{ampliacao}X" + (f" [{aviso}]" if aviso else "")


def desenhar_uma(dataset_raiz: Path, polimero: str, arquivo: str, saida: Path) -> None:
    json_path = dataset_raiz / polimero / f"{polimero}_COCO.json"
    dados = json.loads(json_path.read_text())

    nome_relativo = f"{polimero}/{arquivo}"
    img_info = next((im for im in dados["images"] if im["file_name"] == nome_relativo), None)
    if img_info is None:
        raise SystemExit(f"Imagem '{nome_relativo}' nao encontrada no {json_path.name}")

    anotacoes = [a for a in dados["annotations"] if a["image_id"] == img_info["id"]]
    caminho_imagem = dataset_raiz / nome_relativo
    im, contagem, info = desenhar_imagem(caminho_imagem, arquivo, anotacoes, _fonte())
    if im is None:
        raise SystemExit(f"Nao foi possivel calibrar '{nome_relativo}': {info}")

    im.save(saida)
    print(f"Calibracao (lida do rotulo): {info}")
    print(f"{len(anotacoes)} particulas | nano={contagem['nanoplastico']} "
          f"micro={contagem['microplastico']} fora={contagem['fora_da_faixa']}")
    print(f"Imagem salva em: {saida}")


def desenhar_todas(dataset_raiz: Path, saida_raiz: Path) -> None:
    fonte = _fonte()
    total_geral = {"nanoplastico": 0, "microplastico": 0, "fora_da_faixa": 0}
    total_imagens = 0
    ignoradas: list[tuple[str, str]] = []

    for polimero in POLIMEROS:
        json_path = dataset_raiz / polimero / f"{polimero}_COCO.json"
        if not json_path.exists():
            continue
        dados = json.loads(json_path.read_text())
        anotacoes_por_imagem: dict[int, list[dict]] = {}
        for ann in dados["annotations"]:
            anotacoes_por_imagem.setdefault(ann["image_id"], []).append(ann)

        pasta_saida = saida_raiz / polimero
        pasta_saida.mkdir(parents=True, exist_ok=True)

        for img_info in dados["images"]:
            nome_relativo = img_info["file_name"]
            nome_arquivo = Path(nome_relativo).name
            caminho_imagem = dataset_raiz / nome_relativo
            anotacoes = anotacoes_por_imagem.get(img_info["id"], [])

            im, contagem, info = desenhar_imagem(caminho_imagem, nome_arquivo, anotacoes, fonte)
            if im is None:
                ignoradas.append((nome_relativo, info))
                print(f"  [ignorada] {nome_relativo}: {info}", file=sys.stderr)
                continue

            caminho_saida = pasta_saida / nome_arquivo
            im.save(caminho_saida)
            total_imagens += 1
            for k in total_geral:
                total_geral[k] += contagem[k]
            print(f"  {nome_relativo}: {info} | {len(anotacoes)} particulas "
                  f"(nano={contagem['nanoplastico']} micro={contagem['microplastico']}) "
                  f"-> {caminho_saida}")

    total_particulas = sum(total_geral.values())
    print(f"\n{total_imagens} imagens processadas, {len(ignoradas)} ignoradas.")
    if total_particulas:
        print(f"Total de particulas: {total_particulas}")
        print(f"  nanoplastico  : {total_geral['nanoplastico']} "
              f"({100 * total_geral['nanoplastico'] / total_particulas:.1f}%)")
        print(f"  microplastico : {total_geral['microplastico']} "
              f"({100 * total_geral['microplastico'] / total_particulas:.1f}%)")
    print(f"Imagens salvas em: {saida_raiz}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="MPDataset/Full_Images/COCO Format")
    ap.add_argument("--todas", action="store_true", help="processa todas as imagens do dataset")
    ap.add_argument("--saida-dir", default="MPDataset/Full_Images/Segmentado",
                     help="pasta de saida no modo --todas")
    ap.add_argument("--polimero", help="PE, PET, PP ou PS (modo uma imagem)")
    ap.add_argument("--arquivo", help='nome do arquivo, ex: "PS-2-5000X.png" (modo uma imagem)')
    ap.add_argument("--saida", default=None, help="arquivo de saida (modo uma imagem)")
    args = ap.parse_args()

    dataset_raiz = Path(args.dataset)

    if args.todas:
        desenhar_todas(dataset_raiz, Path(args.saida_dir))
        return

    if not args.polimero or not args.arquivo:
        ap.error("use --todas, ou --polimero + --arquivo para uma imagem so")

    saida = Path(args.saida) if args.saida else Path(f"segmentado_{args.polimero}_{args.arquivo}")
    desenhar_uma(dataset_raiz, args.polimero, args.arquivo, saida)


if __name__ == "__main__":
    main()
