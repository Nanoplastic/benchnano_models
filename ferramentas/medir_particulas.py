#!/usr/bin/env python3
"""
Mede o tamanho real (em micrometros) de cada particula anotada no MPDataset
(formato COCO, pasta Full_Images) e classifica cada uma como nanoplastico
ou microplastico.

Calibracao pixel -> micrometro (SEMPRE lida do rotulo da propria imagem)
-------------------------------------------------------------------------
O nome do arquivo NAO eh usado para calibrar (pode estar errado - foi o que
aconteceu com "PS-2-5000X.png", cujo rotulo real mostra X3,000, nao X5,000).
Em vez disso, cada imagem eh lida assim:

1. reconhecer_ampliacao(): recorta a regiao onde o rotulo do MEV escreve a
   ampliacao ("X100" ... "X10,000") e compara pixel a pixel com 8 recortes
   de referencia (um por ampliacao valida, extraidos e embutidos abaixo em
   base64). Testado nas 102 imagens do dataset: erro medio 0.06, maximo 0.2
   (numa escala 0-255) - ou seja, e um reconhecimento essencialmente exato,
   nao uma adivinhacao.
2. medir_barra_px(): mede em pixels o comprimento da barra de escala branca
   que aparece ao lado do rotulo reconhecido.
3. escala_um_por_px = valor_da_barra_para_essa_ampliacao / barra_medida_px

Se o rotulo nao for reconhecido ou a barra nao for encontrada, a imagem fica
sem calibracao e suas particulas ficam de fora do CSV (aviso_calibracao
explica o motivo) - nunca usamos um valor adivinhado.

Uso:
    python3 medir_particulas.py
    python3 medir_particulas.py --dataset "MPDataset/Full_Images/COCO Format" --saida particulas.csv
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# valor real da barra (em um) para cada ampliacao, conferido visualmente no
# rotulo de imagens de referencia do proprio dataset.
VALOR_BARRA_UM_ESPERADO = {
    100: 100, 200: 100, 500: 50, 1000: 10,
    2000: 10, 3000: 5, 5000: 5, 10000: 1,
}

# recorte fixo (linhas, colunas relativas ao topo/esquerda do rodape preto)
# onde o texto "X<ampliacao>" sempre aparece nas imagens deste dataset.
CROP_Y0, CROP_Y1 = 16, 60
CROP_X0, CROP_X1 = 270, 510

# recortes de referencia (rotulo "X100" .. "X10,000"), em PNG/base64, um por
# ampliacao valida - extraidos de imagens reais do dataset. Usados para
# reconhecer a ampliacao de qualquer outra imagem por comparacao de pixels.
_REF_CROPS_B64 = {
    100: "iVBORw0KGgoAAAANSUhEUgAAAPAAAAAsCAAAAAB+wmoaAAAAk0lEQVR42u3Y0Q5FMBRE0VPx/7/c+057uE2KsPaLxFRqYqZoBAAAAAAA36PcM23tTN06n43NrLT15WtPmGEdvqzHNelh+bPrfV2kGX4Z673LR90ce2POrAfndJFmWIdn9Xj0XZ51fa+LNMM6PON7eqTHR9e0dZFmWIdn/A+P9LgM6SLN8Mt4+L50TW7VvrRIMxwRP7ZKIT+lY5d4AAAAAElFTkSuQmCC",
    200: "iVBORw0KGgoAAAANSUhEUgAAAPAAAAAsCAAAAAB+wmoaAAAAnklEQVR42u2YQQ6AIAwEd43//zJeDYpFkgYjMxcTe2FiFynWMhRJ0qbFQBhhhBFGGAAAAOAOz5tNHbwvwVLH6hwtESbDWTmu8+sg32N1Whrhn7HP3T5K9Xz6pzrYD9RVp6URJsNZOX577j5n040cX+u0NMJkOOM83TMfv6m167Q0wmQ4Yx52MB+PjPLmCyO8Ah+9ly4dy+VempZGWNIBZyUmTvfLBPwAAAAASUVORK5CYII=",
    500: "iVBORw0KGgoAAAANSUhEUgAAAPAAAAAsCAAAAAB+wmoaAAAAoklEQVR42u2Y0QqAIAxF76L//+X1GqXOhGHZOS9RI/DAbjlNn8SH39z0MxBGGGGEEQYAAAAoYfPmWWs8947leqBSrrO1RJgMZ+X4fO/B0rySWQvrtDTCi7HP/Xz45Rr/R+uZVledlkaYDGflOMprK+e1/fe9TksjTIYz9tPR3vnJTN2u09IIk+GMedgezsMartPSCC/GS8+le+ZhzqVpaYQlHQNJJE4HSRdCAAAAAElFTkSuQmCC",
    1000: "iVBORw0KGgoAAAANSUhEUgAAAPAAAAAsCAAAAAB+wmoaAAAAn0lEQVR42u2YQQ6AIBADwfj/L9erMeBCgQvMXEyoKzZukZASAAAAAAAAwHlkr0yV0tK4zGkUvKqnX6d9YQyT4eYc6ydHecIa8R7zdVoaw5txj8Vfn2vtnlE0TaelMUyGazlegYK1oF+npTFMhkv7VyfH0T47eqan09IYJsOljDg5jvbZuaG+X6elMbwZi8+l9TNVy380BfX9Oi2N4c14AJVNLD66lBEzAAAAAElFTkSuQmCC",
    2000: "iVBORw0KGgoAAAANSUhEUgAAAPAAAAAsCAAAAAB+wmoaAAAAsElEQVR42u2YSw6AIAxEW+P9r1z3SumHuIE3GxMnSCbpo1SVY2QiInLJYSIwgQlMYAIjhBBCCI2k/dlSg/e2uFW0vudztSQwDHscv/nVJPfZM6Lyfd+npAm8me41/O31nPXEhePi01P7PiVNYBj2OK7euyvsqsNp3aekCQzDIz4z8/GMrw77PZ+SJjAMjxjRYD6e9fDuqN7zKWkCb6af/ktbYrtMH5Vgxq77lDSBN9MDRncxTS1NL70AAAAASUVORK5CYII=",
    3000: "iVBORw0KGgoAAAANSUhEUgAAAPAAAAAsCAAAAAB+wmoaAAAAp0lEQVR42u2Y0QqAIAxF1+j/P3n0Lm7OWQR6zkvQxeLOXdNEAAAAAAAAAM7jqg0zEdHBfWs0LbwjGl/T9bQZxjAZTue4za8mc59dI2ae7+u0NIY3416rlTXXr+por+m0NIbJsJfjUX50Ibvq5HRep6UxTIZ7+9coxxrsvaOaZ9aGeZ2WxjAZ7mVEE+dR7xtenYuaTktjeDN+/C+d3YNH4+d1WhrDm/EAPCwvQjdx1hQAAAAASUVORK5CYII=",
    5000: "iVBORw0KGgoAAAANSUhEUgAAAPAAAAAsCAAAAAB+wmoaAAAAqklEQVR42u2YSw6AIAxEqfH+V65bQ8D+cCG+tzE6KWSSTqBK+ySarjzaz8AwhjGMYQwDAAAAjJD8PCoP33XBdmrU5nSulhgmw7Mc39+1tvSw3lrfp9PSGN6MsxZ/7Z6RczKa6TU6LY1hMjzLsZXXzNms3T51nZbGMBmezb+VfLZE5nM6LY1hMjzKiCTmYTHuwJ76uE5LY3gzXvov7ZmHPeeoVR/XaWkMb8YFYowvTbHt1wcAAAAASUVORK5CYII=",
    10000: "iVBORw0KGgoAAAANSUhEUgAAAPAAAAAsCAAAAAB+wmoaAAAArElEQVR42u2Yyw6FIAxE4cb//+W6vVHo0xU9Z2PCUMjEjhjGAAAAAAAAAOjHXA/LRlqNa3MdW22x6nP6r9sbxnDTDK+yKUpOZiHrI/GtyOu0NIYP49LjLY/nbk70TI0in+m0NIb7ZHgY+Y1kbyZzbNXHdVoaw70yLIUcWzVi/M576uM6LY3hPhn+z0Amx9OhS7E+rtPSGD6M4r20KEt57o0953Rl/bdOS2P4MG4vfTQ9upcByQAAAABJRU5ErkJggg==",
}
_DIF_MAXIMA_ACEITAVEL = 3.0  # medido nas 102 imagens do dataset: max real = 0.2

# mantido so para preencher a coluna informativa "ampliacao_pelo_nome" e
# assim evidenciar arquivos com nome divergente do rotulo real (nao eh
# usado para calcular a escala).
RE_MAG_COM_X = re.compile(r"(\d+)\s*[xX]")


def _carregar_templates() -> dict[int, np.ndarray]:
    templates = {}
    for mag, b64 in _REF_CROPS_B64.items():
        arr = np.array(Image.open(io.BytesIO(base64.b64decode(b64))))
        templates[mag] = arr.astype(np.int16)
    return templates


_TEMPLATES = _carregar_templates()


def extrair_ampliacao_do_nome(nome_arquivo: str) -> int | None:
    """So para fins de comparacao/diagnostico - NAO calibra nada."""
    candidatos = [int(m) for m in RE_MAG_COM_X.findall(nome_arquivo)]
    candidatos = [c for c in candidatos if c in VALOR_BARRA_UM_ESPERADO]
    return candidatos[0] if candidatos else None


def ler_rodape(caminho_imagem: Path) -> np.ndarray | None:
    """Recorta o rodape preto do MEV (com rotulo e barra de escala)."""
    im = np.array(Image.open(caminho_imagem).convert("L"))
    linhas_media = im.mean(axis=1)
    linhas_escuras = np.where(linhas_media < 40)[0]
    if linhas_escuras.size == 0:
        return None
    return im[linhas_escuras.min():linhas_escuras.max() + 1, :]


def reconhecer_ampliacao(rodape: np.ndarray) -> tuple[int | None, float]:
    """Le a ampliacao comparando o recorte do rotulo com os 8 templates
    conhecidos (comparacao pixel a pixel, nao OCR probabilistico)."""
    if rodape.shape[0] < CROP_Y1:
        return None, float("inf")
    recorte = rodape[CROP_Y0:CROP_Y1, CROP_X0:CROP_X1].astype(np.int16)
    melhor_mag, melhor_dif = None, float("inf")
    for mag, tmpl in _TEMPLATES.items():
        if tmpl.shape != recorte.shape:
            continue
        dif = float(np.abs(tmpl - recorte).mean())
        if dif < melhor_dif:
            melhor_mag, melhor_dif = mag, dif
    if melhor_dif > _DIF_MAXIMA_ACEITAVEL:
        return None, melhor_dif
    return melhor_mag, melhor_dif


def medir_barra_px(rodape: np.ndarray) -> int | None:
    """Mede o comprimento em pixels da barra de escala branca no rodape do MEV."""
    melhor = 0
    for y in range(rodape.shape[0]):
        branco = rodape[y] > 200
        if branco.sum() < 10:
            continue
        idx = np.where(branco)[0]
        cortes = np.split(idx, np.where(np.diff(idx) != 1)[0] + 1)
        for c in cortes:
            if 20 < len(c) < 400:
                melhor = max(melhor, len(c))
    return melhor or None


def area_poligono(pontos_xy: list[float]) -> float:
    """Formula do shoelace: area de um poligono a partir de seus vertices (px)."""
    x = pontos_xy[0::2]
    y = pontos_xy[1::2]
    n = len(x)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += x[i] * y[j] - x[j] * y[i]
    return abs(area) / 2.0


def diametro_feret_max(pontos_xy: list[float]) -> float:
    """Maior distancia entre dois vertices quaisquer do poligono (em px)."""
    x = np.array(pontos_xy[0::2])
    y = np.array(pontos_xy[1::2])
    pts = np.stack([x, y], axis=1)
    d = 0.0
    for i in range(len(pts)):
        dif = pts[i + 1:] - pts[i]
        if len(dif):
            d = max(d, float(np.sqrt((dif ** 2).sum(axis=1)).max()))
    return d


def classificar(diam_um: float) -> str:
    if diam_um < 1.0:
        return "nanoplastico"
    if diam_um <= 5000.0:
        return "microplastico"
    return "fora_da_faixa"


def calibrar_imagem(caminho_completo: Path, nome_arquivo: str) -> tuple[int, float, str] | str:
    """Le ampliacao + barra de escala diretamente do rotulo da imagem e
    retorna (ampliacao, escala_um_por_px, aviso). Se nao for possivel
    calibrar com confianca, retorna uma string com o motivo (a imagem
    inteira fica de fora do CSV)."""
    if not caminho_completo.exists():
        return "imagem_nao_encontrada_no_disco"

    rodape = ler_rodape(caminho_completo)
    if rodape is None:
        return "rodape_do_mev_nao_encontrado"

    ampliacao, dif = reconhecer_ampliacao(rodape)
    if ampliacao is None:
        return f"rotulo_de_ampliacao_nao_reconhecido(dif={dif:.1f})"

    bar_px = medir_barra_px(rodape)
    if not bar_px:
        return "barra_de_escala_nao_encontrada"

    escala = VALOR_BARRA_UM_ESPERADO[ampliacao] / bar_px

    aviso = ""
    ampliacao_pelo_nome = extrair_ampliacao_do_nome(nome_arquivo)
    if ampliacao_pelo_nome is not None and ampliacao_pelo_nome != ampliacao:
        aviso = f"nome_do_arquivo_sugeria_{ampliacao_pelo_nome}X_mas_rotulo_diz_{ampliacao}X"

    return ampliacao, escala, aviso


def processar_coco(
    json_path: Path,
    pasta_imagens: Path,
    linhas_saida: list[dict],
    imagens_ignoradas: list[tuple[str, str]],
) -> None:
    dados = json.loads(json_path.read_text())
    imagens_por_id = {im["id"]: im for im in dados["images"]}
    categorias_por_id = {c["id"]: c["name"] for c in dados["categories"]}

    cache_calibracao: dict[str, tuple[int, float, str] | str] = {}

    for ann in dados["annotations"]:
        img_info = imagens_por_id[ann["image_id"]]
        nome_arquivo = img_info["file_name"]
        caminho_completo = pasta_imagens / nome_arquivo

        if nome_arquivo not in cache_calibracao:
            resultado = calibrar_imagem(caminho_completo, nome_arquivo)
            cache_calibracao[nome_arquivo] = resultado
            if isinstance(resultado, str):
                imagens_ignoradas.append((nome_arquivo, resultado))

        calib = cache_calibracao[nome_arquivo]
        if isinstance(calib, str):
            continue
        ampliacao, escala, aviso = calib

        seg = ann.get("segmentation")
        if seg and len(seg) and len(seg[0]) >= 6:
            pontos = seg[0]
            area_px2 = area_poligono(pontos)
            diam_area_um = 2 * np.sqrt((area_px2 * escala ** 2) / np.pi)
            diam_feret_um = diametro_feret_max(pontos) * escala
            diam_usado = diam_area_um
            metodo = "area_equivalente_poligono"
        else:
            bbox = ann["bbox"]  # [x, y, w, h] em pixels
            diam_usado = max(bbox[2], bbox[3]) * escala
            diam_area_um = None
            diam_feret_um = None
            metodo = "bbox_maior_lado"

        linhas_saida.append({
            "arquivo": nome_arquivo,
            "polimero": categorias_por_id.get(ann["category_id"], "?"),
            "ampliacao": ampliacao,
            "escala_um_por_px": round(escala, 6),
            "diametro_area_equiv_um": round(diam_area_um, 4) if diam_area_um is not None else "",
            "diametro_feret_max_um": round(diam_feret_um, 4) if diam_feret_um is not None else "",
            "diametro_usado_um": round(diam_usado, 4),
            "metodo_diametro": metodo,
            "classificacao": classificar(diam_usado),
            "aviso_calibracao": aviso,
        })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dataset",
        default="MPDataset/Full_Images/COCO Format",
        help="pasta com as subpastas PE/PET/PP/PS contendo os *_COCO.json",
    )
    ap.add_argument("--saida", default="particulas_medidas.csv")
    args = ap.parse_args()

    raiz = Path(args.dataset)
    linhas: list[dict] = []
    imagens_ignoradas: list[tuple[str, str]] = []
    for json_path in sorted(raiz.glob("*/*_COCO.json")):
        pasta_imagens = json_path.parent.parent
        print(f"Processando {json_path} ...", file=sys.stderr)
        processar_coco(json_path, pasta_imagens, linhas, imagens_ignoradas)

    if imagens_ignoradas:
        print(f"\n{len(imagens_ignoradas)} imagem(ns) ignorada(s) (sem calibracao confiavel):", file=sys.stderr)
        for nome, motivo in imagens_ignoradas:
            print(f"  - {nome}: {motivo}", file=sys.stderr)

    if not linhas:
        print("Nenhuma anotacao processada - confira o caminho --dataset.", file=sys.stderr)
        sys.exit(1)

    campos = list(linhas[0].keys())
    with open(args.saida, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)

    total = len(linhas)
    nano = sum(1 for l in linhas if l["classificacao"] == "nanoplastico")
    micro = sum(1 for l in linhas if l["classificacao"] == "microplastico")
    fora = total - nano - micro
    avisos = sum(1 for l in linhas if l["aviso_calibracao"])

    print(f"\nTotal de particulas medidas: {total}")
    print(f"  nanoplastico  : {nano} ({100 * nano / total:.1f}%)")
    print(f"  microplastico : {micro} ({100 * micro / total:.1f}%)")
    print(f"  fora da faixa : {fora} ({100 * fora / total:.1f}%)")
    if avisos:
        print(f"  avisos de calibracao: {avisos} (ver coluna aviso_calibracao no CSV)")
    print(f"\nResultado salvo em: {args.saida}")


if __name__ == "__main__":
    main()
