"""Validação da calibração pixel->físico / Pixel-to-physical calibration check.

PT: Valida empiricamente a calibração µm/px = 100/ampliação contra a barra de escala gravada
    pelo software do MEV nas imagens de origem do MiNa, e grava CSV, script e log
    rastreáveis em resultados/.
    Para cada uma das 7 ampliações nominais (100x a 10000x), mede o comprimento em pixels da
    barra de escala branca sobreposta na faixa de informação do rodapé (faixa preta de ~130
    px na base, comum a todas as imagens do MiNa) e compara com o valor previsto
    (100/ampliação).
    Detecção da barra: dentro da faixa do rodapé, procura a maior sequência horizontal
    contígua de pixels quase brancos (R, G, B > LIMIAR_BRANCO) em qualquer linha. A barra é
    um retângulo sólido muito mais longo que qualquer traço do texto do rodapé (tensão,
    ampliação, rótulo, detector), então a sequência mais longa isola a barra.
    O rótulo físico da barra (em micrômetros, ex.: "100um", "10um", "5um", "1um") é o texto
    gravado em cada imagem, fixado por imagem a partir de inspeção visual; não há OCR.
EN: Empirically validates the calibration µm/px = 100/magnification against the scale bar
    burned in by the SEM software on the MiNa source images, and writes a traceable CSV,
    script and log under resultados/.
    For each of the 7 nominal magnifications (100x to 10000x), measures the pixel length of
    the white scale bar overlaid on the footer information band (a ~130 px black band at the
    bottom, common to all MiNa images) and compares it with the predicted value
    (100/magnification).
    Bar detection: within the footer band, finds the longest contiguous horizontal run of
    near-white pixels (R, G, B > LIMIAR_BRANCO) on any row. The bar is a solid rectangle
    much longer than any stroke of the footer text (voltage, magnification, label,
    detector), so the longest run isolates the bar.
    The bar's physical label (in micrometres, e.g. "100um", "10um", "5um", "1um") is the
    text burned into each image, fixed per image from visual inspection; there is no OCR.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import salvar_resumo_experimento, configurar_log_experimento

NOME_EXP = "26_pixel_calibration_validation"
DIR_EXP, CAMINHO_LOG, REGISTRADOR, METADADOS_EXECUCAO = configurar_log_experimento(NOME_EXP, __file__)

RAIZ = Path(__file__).resolve().parent.parent
SEGMENTADO_DIR = RAIZ / "MPDataset" / "Full_Images" / "Segmentado"

LIMIAR_BRANCO = 200
FAIXA_RODAPE_PX = 130  # PT: altura da faixa preta de informação no rodapé / EN: height of the black information band at the bottom

# PT: Uma imagem de origem por ampliação nominal. rotulo_um é lido do texto gravado na
#     própria imagem (rótulo da barra), não calculado.
# EN: One source image per nominal magnification. rotulo_um is read from the text burned
#     into the image itself (bar label), not computed.
IMAGENS_CALIBRACAO = [
    {"magnification": 100, "path": SEGMENTADO_DIR / "PP" / "PP 100X.png", "label_um": 100},
    {"magnification": 200, "path": SEGMENTADO_DIR / "PS" / "PS-200X.png", "label_um": 100},
    {"magnification": 1000, "path": SEGMENTADO_DIR / "PE" / "PE 1000X.png", "label_um": 10},
    {"magnification": 2000, "path": SEGMENTADO_DIR / "PE" / "PE 2000X.png", "label_um": 10},
    {"magnification": 3000, "path": SEGMENTADO_DIR / "PE" / "PE 3000X.png", "label_um": 5},
    {"magnification": 5000, "path": SEGMENTADO_DIR / "PE" / "PE-5000x.png", "label_um": 5},
    {"magnification": 10000, "path": SEGMENTADO_DIR / "PE" / "PE-10000X.png", "label_um": 1},
]


def medir_barra_escala_px(caminho_imagem: Path) -> int:
    """PT: Maior sequência horizontal contígua de pixels quase brancos na faixa do rodapé.
        Retorna o comprimento em pixels da barra de escala detectada.
    EN: Longest contiguous horizontal run of near-white pixels in the footer band. Returns
        the pixel length of the detected scale bar.
    """
    im = np.array(Image.open(caminho_imagem).convert("RGB"))
    altura = im.shape[0]
    faixa = im[altura - FAIXA_RODAPE_PX:altura, :, :]
    branco = np.all(faixa > LIMIAR_BRANCO, axis=2)

    melhor_comprimento = 0
    for linha_tab in range(branco.shape[0]):
        indice = np.where(branco[linha_tab])[0]
        if len(indice) == 0:
            continue
        particoes = np.where(np.diff(indice) > 1)[0]
        for executar in np.split(indice, particoes + 1):
            compr_execucao = int(executar[-1] - executar[0] + 1)
            if compr_execucao > melhor_comprimento:
                melhor_comprimento = compr_execucao
    return melhor_comprimento


def principal():
    REGISTRADOR.info("Validando calibracao pixel->fisico contra %d imagens de %s",
                len(IMAGENS_CALIBRACAO), SEGMENTADO_DIR)

    linhas_tab = []
    for entrada in IMAGENS_CALIBRACAO:
        ampl = entrada["magnification"]
        caminho = entrada["path"]
        rotulo_um = entrada["label_um"]
        assert caminho.exists(), f"imagem de calibracao ausente: {caminho}"

        barra_px = medir_barra_escala_px(caminho)
        um_por_px_medido = rotulo_um / barra_px
        um_por_px_previsto = 100.0 / ampl
        erro_pct = (um_por_px_medido - um_por_px_previsto) / um_por_px_previsto * 100.0

        REGISTRADOR.info(
            "%sx | imagem=%s | barra_px=%d | rotulo_um=%s | um_por_px_medido=%.4f | erro=%.2f%%",
            ampl, caminho.relative_to(RAIZ), barra_px, rotulo_um, um_por_px_medido, erro_pct,
        )

        linhas_tab.append({
            "magnification": ampl,
            "image": str(caminho.relative_to(RAIZ)),
            "scale_bar_px": barra_px,
            "label_um": rotulo_um,
            "um_per_px_measured": um_por_px_medido,
            "um_per_px_predicted": um_por_px_previsto,
            "error_pct": erro_pct,
        })

    erro_abs_max_pct = max(abs(r["error_pct"]) for r in linhas_tab)
    REGISTRADOR.info("Erro maximo absoluto observado: %.2f%%", erro_abs_max_pct)

    metricas = {
        "white_threshold": LIMIAR_BRANCO,
        "footer_band_px": FAIXA_RODAPE_PX,
        "n_magnifications": len(linhas_tab),
        "rows": linhas_tab,
        "max_abs_error_pct": erro_abs_max_pct,
    }
    caminho_resumo = salvar_resumo_experimento(NOME_EXP, METADADOS_EXECUCAO, metricas)
    REGISTRADOR.info("Resumo salvo em %s", caminho_resumo)


if __name__ == "__main__":
    principal()
