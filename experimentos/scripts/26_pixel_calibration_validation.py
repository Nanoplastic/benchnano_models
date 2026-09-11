"""Validacao empirica da calibracao pixel->fisico (Eq. eq:pixel_calibration,
Secao sec:size_labels / Tabela tab:pixel_calibration) contra a barra de
escala gravada pelo software do MEV nas imagens de origem do MiNa.

Fecha o item da revisao de reprodutibilidade que apontava a Tabela
tab:pixel_calibration como a unica sem CSV/script/log de origem no
repositorio: as demais secoes numeradas (00-25) ja tem uma pasta rastreavel
em resultados/, esta nao tinha nenhuma.

Para cada uma das 7 ampliacoes nominais da tabela publicada (100x a
10000x), mede programaticamente o comprimento em pixel da barra de escala
branca sobrepesta na faixa de informacao no rodape da imagem (banda preta de
~130px na base, comum a todas as imagens do MiNa) e compara com o previsto
pela Eq. eq:pixel_calibration (100/ampliacao).

Deteccao da barra: dentro da banda do rodape, procura o maior run
horizontal contiguo de pixels quase-brancos (R,G,B > WHITE_THRESHOLD) em
qualquer linha. A barra e um retangulo solido bem mais longo que qualquer
traco de fonte do texto do rodape (voltagem/ampliacao/rotulo/detector), entao
o run mais longo da banda isola a barra de forma robusta -- verificado
visualmente contra as 7 imagens antes de fixar este metodo.

O rotulo fisico da barra (em micrometros) e lido diretamente do texto
gravado na imagem (ex.: "100um", "10um", "5um", "1um"), o mesmo texto usado
para compor a Tabela publicada; nao ha OCR aqui, o rotulo e fixado por
imagem a partir da inspecao visual documentada no relatorio de revisao.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import save_experiment_summary, setup_experiment_logging

EXP_NAME = "26_pixel_calibration_validation"
EXP_DIR, LOG_PATH, LOGGER, RUNTIME_METADATA = setup_experiment_logging(EXP_NAME, __file__)

ROOT = Path(__file__).resolve().parent.parent
SEGMENTADO_DIR = ROOT / "MPDataset" / "Full_Images" / "Segmentado"

WHITE_THRESHOLD = 200
FOOTER_BAND_PX = 130  # altura da faixa preta de informacao no rodape

# Uma imagem de origem por ampliacao nominal, a mesma selecao usada na
# Tabela tab:pixel_calibration publicada. label_um e lido do texto gravado
# na propria imagem (rotulo da barra), nao calculado.
CALIBRATION_IMAGES = [
    {"magnification": 100, "path": SEGMENTADO_DIR / "PP" / "PP 100X.png", "label_um": 100},
    {"magnification": 200, "path": SEGMENTADO_DIR / "PS" / "PS-200X.png", "label_um": 100},
    {"magnification": 1000, "path": SEGMENTADO_DIR / "PE" / "PE 1000X.png", "label_um": 10},
    {"magnification": 2000, "path": SEGMENTADO_DIR / "PE" / "PE 2000X.png", "label_um": 10},
    {"magnification": 3000, "path": SEGMENTADO_DIR / "PE" / "PE 3000X.png", "label_um": 5},
    {"magnification": 5000, "path": SEGMENTADO_DIR / "PE" / "PE-5000x.png", "label_um": 5},
    {"magnification": 10000, "path": SEGMENTADO_DIR / "PE" / "PE-10000X.png", "label_um": 1},
]


def measure_scale_bar_px(image_path: Path) -> int:
    """Maior run horizontal contiguo de pixels quase-brancos na banda do
    rodape. Retorna o comprimento em pixel da barra de escala detectada."""
    im = np.array(Image.open(image_path).convert("RGB"))
    height = im.shape[0]
    band = im[height - FOOTER_BAND_PX:height, :, :]
    white = np.all(band > WHITE_THRESHOLD, axis=2)

    best_len = 0
    for row in range(white.shape[0]):
        idx = np.where(white[row])[0]
        if len(idx) == 0:
            continue
        splits = np.where(np.diff(idx) > 1)[0]
        for run in np.split(idx, splits + 1):
            run_len = int(run[-1] - run[0] + 1)
            if run_len > best_len:
                best_len = run_len
    return best_len


def main():
    LOGGER.info("Validando calibracao pixel->fisico contra %d imagens de %s",
                len(CALIBRATION_IMAGES), SEGMENTADO_DIR)

    rows = []
    for entry in CALIBRATION_IMAGES:
        mag = entry["magnification"]
        path = entry["path"]
        label_um = entry["label_um"]
        assert path.exists(), f"imagem de calibracao ausente: {path}"

        bar_px = measure_scale_bar_px(path)
        um_per_px_measured = label_um / bar_px
        um_per_px_predicted = 100.0 / mag
        error_pct = (um_per_px_measured - um_per_px_predicted) / um_per_px_predicted * 100.0

        LOGGER.info(
            "%sx | imagem=%s | barra_px=%d | rotulo_um=%s | um_por_px_medido=%.4f | erro=%.2f%%",
            mag, path.relative_to(ROOT), bar_px, label_um, um_per_px_measured, error_pct,
        )

        rows.append({
            "magnification": mag,
            "image": str(path.relative_to(ROOT)),
            "scale_bar_px": bar_px,
            "label_um": label_um,
            "um_per_px_measured": um_per_px_measured,
            "um_per_px_predicted": um_per_px_predicted,
            "error_pct": error_pct,
        })

    max_abs_error_pct = max(abs(r["error_pct"]) for r in rows)
    LOGGER.info("Erro maximo absoluto observado: %.2f%%", max_abs_error_pct)

    metrics = {
        "white_threshold": WHITE_THRESHOLD,
        "footer_band_px": FOOTER_BAND_PX,
        "n_magnifications": len(rows),
        "rows": rows,
        "max_abs_error_pct": max_abs_error_pct,
    }
    summary_path = save_experiment_summary(EXP_NAME, RUNTIME_METADATA, metrics)
    LOGGER.info("Resumo salvo em %s", summary_path)


if __name__ == "__main__":
    main()
