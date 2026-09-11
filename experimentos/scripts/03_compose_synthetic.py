"""Compõe imagens sintéticas para as 4 classes MiNa (PE/PET/PP/PS) colando
1-3 templates reais de partícula (script 01, RGBA com alpha=máscara COCO,
só de imagens do split `train`) sobre tiles de fundo reais (script 02,
também só `train`).

Reaproveita a lógica de `tarefa_vlm_nanoplastic_fewshot/scripts/03_compose_synthetic.py`
(match_tone/prep_shape/paste_shape), mas o deslocamento de brilho
primeiro-plano vs. fundo é calibrado a partir dos PRÓPRIOS pixels do MiNa
(medido separadamente: Δmédia real partícula-vs-fundo local = 65.6±28.8,
percentis 10/90 = 23/101 sobre 60 templates amostrados) -- a linha anterior
usava um deslocamento ajustado a mão para o Moon et al., que não tem por que
valer aqui sem checar.
"""
import csv
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENT_DIR = ROOT / "resultados" / "03_compose_synthetic"
SHAPES = ROOT / "resultados" / "01_extract_real_crops" / "data" / "shapes"
BACKGROUNDS = ROOT / "resultados" / "02_extract_backgrounds" / "data" / "backgrounds"
OUT = EXPERIMENT_DIR / "data" / "synthetic"
CANVAS = 384
CLASSES = ["PE", "PET", "PP", "PS"]
N_PER_CLASS = 250
SEED = 42

# calibrado a partir de data/shapes/* real (ver docstring acima)
DELTA_LO, DELTA_HI = 23, 101


def list_files(d: Path):
    return sorted(p for p in d.iterdir() if p.suffix == ".png")


def load_rgba(p: Path) -> Image.Image:
    return Image.open(p).convert("RGBA")


def random_bg_crop(bg_paths, rng: random.Random) -> Image.Image:
    src = Image.open(rng.choice(bg_paths)).convert("L")
    scale = rng.uniform(1.0, 1.6)
    src = src.resize((int(src.width * scale), int(src.height * scale)), Image.BICUBIC)
    if src.width <= CANVAS or src.height <= CANVAS:
        src = src.resize((CANVAS + 20, CANVAS + 20))
    x0 = rng.randint(0, src.width - CANVAS)
    y0 = rng.randint(0, src.height - CANVAS)
    tile = src.crop((x0, y0, x0 + CANVAS, y0 + CANVAS)).convert("RGB")
    tile = ImageEnhance.Brightness(tile).enhance(rng.uniform(0.9, 1.1))
    tile = ImageEnhance.Contrast(tile).enhance(rng.uniform(0.9, 1.1))
    return tile


def prep_shape(path: Path, rng: random.Random) -> Image.Image:
    """Escala relativa ao tamanho NATIVO do template (0.8x-2.0x), não um
    alvo absoluto de pixels. Bug encontrado ao inspecionar as primeiras
    composições visualmente: a maioria dos templates reais do MiNa é
    pequena (mediana 20-28px de lado maior, todas as 4 classes), então um
    alvo absoluto tipo randint(60,200) forçava upscale de até ~10x via
    BICUBIC -- produzindo blobs borrados/com anel escuro (ringing de
    interpolação), não uma partícula sintética fiel. Limitar o fator de
    escala evita o artefato mantendo alguma variação de tamanho/pose."""
    shape = load_rgba(path)
    scale = rng.uniform(0.8, 2.0)
    new_size = (max(int(shape.width * scale), 1), max(int(shape.height * scale), 1))
    shape = shape.resize(new_size, Image.BICUBIC)
    angle = rng.uniform(0, 360)
    shape = shape.rotate(angle, expand=True, resample=Image.BICUBIC)
    return shape


def match_tone(shape: Image.Image, bg_mean: float, rng: random.Random) -> Image.Image:
    r, g, b, a = shape.split()
    arr = np.array(r).astype(np.float32)
    amask = np.array(a) > 10
    if not amask.any():
        return shape
    m = arr[amask].mean()
    target_mean = min(bg_mean + rng.uniform(DELTA_LO, DELTA_HI), 245)
    arr = (arr - m) * rng.uniform(1.0, 1.3) + target_mean
    arr = np.clip(arr, 0, 255).astype("uint8")
    gray = Image.fromarray(arr)
    return Image.merge("RGBA", (gray, gray, gray, a))


def paste_shape(canvas: Image.Image, shape: Image.Image, cx: int, cy: int):
    x0 = cx - shape.width // 2
    y0 = cy - shape.height // 2
    canvas.paste(shape, (x0, y0), shape)


def compose_one(shape_paths, bg_paths, rng: random.Random) -> Image.Image:
    canvas = random_bg_crop(bg_paths, rng).convert("RGBA")
    bg_mean = float(np.array(canvas.convert("L")).mean())

    n_obj = rng.choices([1, 2, 3], weights=[0.5, 0.35, 0.15])[0]
    for _ in range(n_obj):
        p = rng.choice(shape_paths)
        s = match_tone(prep_shape(p, rng), bg_mean, rng)
        cx = rng.randint(s.width // 2 + 5, max(CANVAS - s.width // 2 - 5, s.width // 2 + 6))
        cy = rng.randint(s.height // 2 + 5, max(CANVAS - s.height // 2 - 5, s.height // 2 + 6))
        paste_shape(canvas, s, cx, cy)

    canvas = canvas.convert("RGB")
    canvas = canvas.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.2, 0.5)))
    arr = np.array(canvas).astype(np.float32)
    arr += np.random.normal(0, rng.uniform(1, 3), arr.shape)
    arr = np.clip(arr, 0, 255).astype("uint8")
    return Image.fromarray(arr)


def main():
    rng = random.Random(SEED)
    np.random.seed(SEED)
    manifest_rows = []

    for cls in CLASSES:
        shape_paths = list_files(SHAPES / cls)
        bg_paths = list_files(BACKGROUNDS / cls)
        if not shape_paths or not bg_paths:
            print(f"{cls}: SEM templates/fundo suficiente, pulando ({len(shape_paths)} shapes, {len(bg_paths)} bg)")
            continue
        out_dir = OUT / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(N_PER_CLASS):
            img = compose_one(shape_paths, bg_paths, rng)
            name = f"{cls}_{i:04d}.png"
            img.save(out_dir / name)
            manifest_rows.append({
                "image_path": str((out_dir / name).relative_to(ROOT)),
                "label": cls,
            })
        print(f"{cls}: {N_PER_CLASS} imagens sintéticas -> {out_dir} "
              f"({len(shape_paths)} shape templates, {len(bg_paths)} tiles de fundo disponíveis)")

    manifest_path = EXPERIMENT_DIR / "data" / "manifest_synthetic.csv"
    with open(manifest_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image_path", "label"])
        w.writeheader()
        w.writerows(manifest_rows)
    print(f"manifest sintético -> {manifest_path} ({len(manifest_rows)} linhas)")


if __name__ == "__main__":
    main()
