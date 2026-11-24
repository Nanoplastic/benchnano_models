"""Composição de imagens sintéticas / Synthetic image composition.

PT: Gera imagens sintéticas para as 4 classes MiNa (PE/PET/PP/PS) colando 1-3 moldes reais
    de partícula (script 01; RGBA com alfa = máscara COCO, só da partição `train`) sobre
    blocos de fundo reais (script 02, também só `train`).
    O deslocamento de brilho partícula-vs-fundo é calibrado nos próprios pixels do MiNa:
    diferença média real partícula-vs-fundo local = 65,6 ± 28,8; percentis 10/90 = 23/101
    sobre 60 moldes amostrados.
EN: Generates synthetic images for the 4 MiNa classes (PE/PET/PP/PS) by pasting 1-3 real
    particle templates (script 01; RGBA with alpha = COCO mask, `train` split only) onto
    real background tiles (script 02, also `train` only).
    The particle-vs-background brightness offset is calibrated on MiNa's own pixels: mean
    real particle-vs-local-background difference = 65.6 ± 28.8; 10th/90th percentiles =
    23/101 over 60 sampled templates.
"""
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

RAIZ = Path(__file__).resolve().parent.parent
DIR_EXPERIMENTO_BASE = RAIZ / "resultados" / "03_compose_synthetic"
FORMAS = RAIZ / "resultados" / "01_extract_real_crops" / "data" / "shapes"
FUNDOS = RAIZ / "resultados" / "02_extract_backgrounds" / "data" / "backgrounds"
SAIDA = DIR_EXPERIMENTO_BASE / "data" / "synthetic"
TELA = 384
CLASSES_POLIMERO = ["PE", "PET", "PP", "PS"]
N_POR_CLASSE = 250
SEMENTE = 42

# PT: faixa de deslocamento de brilho calibrada em data/shapes/* (percentis 10/90)
# EN: brightness offset range calibrated on data/shapes/* (10th/90th percentiles)
DELTA_INF, DELTA_SUP = 23, 101


def listar_arquivos(d: Path):
    return sorted(p for p in d.iterdir() if p.suffix == ".png")


def carregar_rgba(p: Path) -> Image.Image:
    return Image.open(p).convert("RGBA")


def recorte_fundo_aleatorio(caminhos_fundo, gerador: random.Random) -> Image.Image:
    origem = Image.open(gerador.choice(caminhos_fundo)).convert("L")
    escala = gerador.uniform(1.0, 1.6)
    origem = origem.resize((int(origem.width * escala), int(origem.height * escala)), Image.BICUBIC)
    if origem.width <= TELA or origem.height <= TELA:
        origem = origem.resize((TELA + 20, TELA + 20))
    x0 = gerador.randint(0, origem.width - TELA)
    y0 = gerador.randint(0, origem.height - TELA)
    bloco_img = origem.crop((x0, y0, x0 + TELA, y0 + TELA)).convert("RGB")
    bloco_img = ImageEnhance.Brightness(bloco_img).enhance(gerador.uniform(0.9, 1.1))
    bloco_img = ImageEnhance.Contrast(bloco_img).enhance(gerador.uniform(0.9, 1.1))
    return bloco_img


def preparar_forma(caminho: Path, gerador: random.Random) -> Image.Image:
    """PT: Redimensiona o molde por um fator relativo ao seu tamanho nativo (0,8x-2,0x), não
        para um alvo absoluto em pixels. Os moldes do MiNa são pequenos (mediana de 20-28 px
        no lado maior); um alvo absoluto forçaria ampliações de até ~10x com BICUBIC,
        gerando borrão e anéis de interpolação. Limitar o fator evita o artefato e mantém
        variação de tamanho e pose.
    EN: Rescales the template by a factor relative to its native size (0.8x-2.0x), not to an
        absolute pixel target. MiNa templates are small (median 20-28 px on the longest
        side); an absolute target would force upscaling of up to ~10x with BICUBIC,
        producing blur and interpolation ringing. Bounding the factor avoids the artifact
        while keeping size and pose variation.
    """
    forma = carregar_rgba(caminho)
    escala = gerador.uniform(0.8, 2.0)
    novo_tamanho = (max(int(forma.width * escala), 1), max(int(forma.height * escala), 1))
    forma = forma.resize(novo_tamanho, Image.BICUBIC)
    angulo = gerador.uniform(0, 360)
    forma = forma.rotate(angulo, expand=True, resample=Image.BICUBIC)
    return forma


def ajustar_tom(forma: Image.Image, media_fundo: float, gerador: random.Random) -> Image.Image:
    r, g, b, a = forma.split()
    matriz = np.array(r).astype(np.float32)
    mascara_a = np.array(a) > 10
    if not mascara_a.any():
        return forma
    m = matriz[mascara_a].mean()
    media_alvo = min(media_fundo + gerador.uniform(DELTA_INF, DELTA_SUP), 245)
    matriz = (matriz - m) * gerador.uniform(1.0, 1.3) + media_alvo
    matriz = np.clip(matriz, 0, 255).astype("uint8")
    cinza = Image.fromarray(matriz)
    return Image.merge("RGBA", (cinza, cinza, cinza, a))


def colar_forma(tela: Image.Image, forma: Image.Image, cx: int, cy: int):
    x0 = cx - forma.width // 2
    y0 = cy - forma.height // 2
    tela.paste(forma, (x0, y0), forma)


def compor_uma(caminhos_formas, caminhos_fundo, gerador: random.Random) -> Image.Image:
    tela = recorte_fundo_aleatorio(caminhos_fundo, gerador).convert("RGBA")
    media_fundo = float(np.array(tela.convert("L")).mean())

    n_obj = gerador.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
    for _ in range(n_obj):
        p = gerador.choice(caminhos_formas)
        s = ajustar_tom(preparar_forma(p, gerador), media_fundo, gerador)
        cx = gerador.randint(s.width // 2 + 5, max(TELA - s.width // 2 - 5, s.width // 2 + 6))
        cy = gerador.randint(s.height // 2 + 5, max(TELA - s.height // 2 - 5, s.height // 2 + 6))
        colar_forma(tela, s, cx, cy)

    tela = tela.convert("RGB")
    tela = tela.filter(ImageFilter.GaussianBlur(radius=gerador.uniform(0.2, 0.5)))
    matriz = np.array(tela).astype(np.float32)
    matriz += np.random.normal(0, gerador.uniform(1, 3), matriz.shape)
    matriz = np.clip(matriz, 0, 255).astype("uint8")
    return Image.fromarray(matriz)


def principal():
    gerador = random.Random(SEMENTE)
    np.random.seed(SEMENTE)
    linhas_manifesto = []

    for cls in CLASSES_POLIMERO:
        caminhos_formas = listar_arquivos(FORMAS / cls)
        caminhos_fundo = listar_arquivos(FUNDOS / cls)
        if not caminhos_formas or not caminhos_fundo:
            print(f"{cls}: SEM templates/fundo suficiente, pulando ({len(caminhos_formas)} shapes, {len(caminhos_fundo)} bg)")
            continue
        dir_saida = SAIDA / cls
        dir_saida.mkdir(parents=True, exist_ok=True)
        for i in range(N_POR_CLASSE):
            img = compor_uma(caminhos_formas, caminhos_fundo, gerador)
            nome = f"{cls}_{i:04d}.png"
            img.save(dir_saida / nome)
            linhas_manifesto.append({
                "image_path": str((dir_saida / nome).relative_to(RAIZ)),
                "label": cls,
            })
        print(f"{cls}: {N_POR_CLASSE} imagens sintéticas -> {dir_saida} "
              f"({len(caminhos_formas)} shape templates, {len(caminhos_fundo)} tiles de fundo disponíveis)")

    caminho_manifesto = DIR_EXPERIMENTO_BASE / "data" / "manifest_synthetic.csv"
    with open(caminho_manifesto, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "label"])
        w.writeheader()
        w.writerows(linhas_manifesto)
    print(f"manifest sintético -> {caminho_manifesto} ({len(linhas_manifesto)} linhas)")


if __name__ == "__main__":
    principal()
