"""Dataset YOLO só com nanopartículas / Nano-only YOLO dataset.

PT: Gera um dataset YOLO só com partículas < 1000 nm para treinar um detector especializado
    em nanoplástico.
    Usa a calibração px->nm (100/ampliação) e o limiar NANO_MAX_NM = 1000 nm de
    23_recall_by_size_regime.py / 26_pixel_calibration_validation.py. Em treino e validação,
    mantém só as caixas com size_nm < 1000; caixas maiores são removidas do rótulo (não da
    imagem), para que o detector aprenda a NÃO marcar micropartículas. Imagens sem nenhuma
    caixa nano são excluídas de treino/val.
    A partição de teste (datasets_yolo26_v2/images|labels/test) é copiada literalmente, sem
    filtro; o filtro nano é aplicado só na avaliação (52_class_error_analysis_nano.py), como
    em 23_recall_by_size_regime.py, para não alterar a partição de teste em disco.
    Escreve só em datasets_yolo26_v2_nano/ e resultados/47_build_nano_only_yolo_dataset/
    (metadados).
EN: Builds a YOLO dataset with only < 1000 nm particles to train a nanoplastic-specialized
    detector.
    Uses the px->nm calibration (100/magnification) and the NANO_MAX_NM = 1000 nm threshold
    from 23_recall_by_size_regime.py / 26_pixel_calibration_validation.py. In train and val,
    only boxes with size_nm < 1000 are kept; larger boxes are removed from the label (not
    from the image), so the detector learns NOT to mark microparticles. Images with no nano
    box are dropped from train/val.
    The test split (datasets_yolo26_v2/images|labels/test) is copied verbatim, unfiltered;
    the nano filter is applied only at evaluation time (52_class_error_analysis_nano.py), as
    in 23_recall_by_size_regime.py, so the test split on disk is never altered.
    Writes only to datasets_yolo26_v2_nano/ and resultados/47_build_nano_only_yolo_dataset/
    (metadata).
"""
import re
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import salvar_resumo_experimento, configurar_log_experimento

NOME_EXP = "47_build_nano_only_yolo_dataset"
DIR_EXP, CAMINHO_LOG, REGISTRADOR, METADADOS_EXECUCAO = configurar_log_experimento(NOME_EXP, __file__)

RAIZ = Path(__file__).resolve().parent.parent
DIR_ORIGEM = RAIZ / "datasets_yolo26_v2"
DIR_SAIDA = RAIZ / "datasets_yolo26_v2_nano"
NOMES_CLASSES = ["PE", "PET", "PP", "PS"]
NANO_MAX_NM = 1000.0

RE_DUPLICADO = re.compile(r"^dup\d+_")
AMPL_X = re.compile(r"(\d+)\s*[xX]")
AMPL_TRACO = re.compile(r"^\D*(\d+)-\d+")


def interpretar_ampliacao(nome: str):
    m = AMPL_X.findall(nome)
    if m:
        return int(m[-1])
    m = AMPL_TRACO.match(nome)
    if m:
        return int(m.group(1))
    return None


def filtrar_rotulos_nano(caminho_rotulo: Path, W: int, H: int, um_por_px: float):
    """PT: Retorna (n_caixas_total, linhas_yolo_mantidas); mantém só caixas com size_nm <
        NANO_MAX_NM, no mesmo formato YOLO da entrada.
    EN: Returns (n_caixas_total, kept_yolo_lines); keeps only boxes with size_nm <
        NANO_MAX_NM, in the same YOLO format as the input.
    """
    if not caminho_rotulo.exists():
        return 0, []
    total = 0
    mantidos = []
    for linha in open(caminho_rotulo):
        partes = linha.split()
        if len(partes) < 5:
            continue
        total += 1
        id_cls = int(partes[0])
        cx, cy, w, h = map(float, partes[1:5])
        tamanho_px = max(w * W, h * H)
        tamanho_nm = tamanho_px * um_por_px * 1000.0
        if tamanho_nm < NANO_MAX_NM:
            mantidos.append(f"{id_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    return total, mantidos


def processar_particao(particao: str):
    dir_img = DIR_ORIGEM / "images" / particao
    dir_rotulo = DIR_ORIGEM / "labels" / particao
    dir_img_saida = DIR_SAIDA / "images" / particao
    dir_rotulo_saida = DIR_SAIDA / "labels" / particao
    dir_img_saida.mkdir(parents=True, exist_ok=True)
    dir_rotulo_saida.mkdir(parents=True, exist_ok=True)

    caminhos_img = sorted(dir_img.glob("*.png"))

    n_imagens_total = 0
    n_imagens_mantidas = 0
    n_caixas_total = 0
    n_caixas_nano = 0
    n_ampl_nao_resolvida = 0

    for caminho_img in caminhos_img:
        n_imagens_total += 1
        radical_base = RE_DUPLICADO.sub("", caminho_img.stem)
        ampl = interpretar_ampliacao(radical_base)
        if ampl is None:
            n_ampl_nao_resolvida += 1
            REGISTRADOR.warning("magnificacao nao resolvida, imagem descartada: %s", caminho_img.name)
            continue
        um_por_px = 100.0 / ampl

        caminho_rotulo = dir_rotulo / f"{caminho_img.stem}.txt"
        with Image.open(caminho_img) as im:
            W, H = im.size
        n_total, linhas_mantidas = filtrar_rotulos_nano(caminho_rotulo, W, H, um_por_px)
        n_caixas_total += n_total
        n_caixas_nano += len(linhas_mantidas)

        if not linhas_mantidas:
            continue

        n_imagens_mantidas += 1
        shutil.copy2(caminho_img, dir_img_saida / caminho_img.name)
        with open(dir_rotulo_saida / f"{caminho_img.stem}.txt", "w") as f:
            f.writelines(linhas_mantidas)

    estatisticas = {
        "n_images_total": n_imagens_total,
        "n_images_kept": n_imagens_mantidas,
        "n_boxes_total": n_caixas_total,
        "n_boxes_nano": n_caixas_nano,
        "n_unresolved_magnification": n_ampl_nao_resolvida,
    }
    REGISTRADOR.info("split=%s stats=%s", particao, estatisticas)
    print(f"{particao}: {n_imagens_total} imagens -> {n_imagens_mantidas} com >=1 caixa nano "
          f"({n_caixas_nano}/{n_caixas_total} caixas nano)")
    return estatisticas


def copiar_teste_literal():
    dir_img = DIR_ORIGEM / "images" / "test"
    dir_rotulo = DIR_ORIGEM / "labels" / "test"
    dir_img_saida = DIR_SAIDA / "images" / "test"
    dir_rotulo_saida = DIR_SAIDA / "labels" / "test"
    dir_img_saida.mkdir(parents=True, exist_ok=True)
    dir_rotulo_saida.mkdir(parents=True, exist_ok=True)
    n = 0
    for caminho_img in sorted(dir_img.glob("*.png")):
        shutil.copy2(caminho_img, dir_img_saida / caminho_img.name)
        caminho_rotulo = dir_rotulo / f"{caminho_img.stem}.txt"
        if caminho_rotulo.exists():
            shutil.copy2(caminho_rotulo, dir_rotulo_saida / caminho_rotulo.name)
        n += 1
    print(f"test: {n} imagens copiadas verbatim (SEM filtro nano -- filtro aplicado só na avaliação)")
    return {"n_images": n}


def principal():
    REGISTRADOR.info("=== construcao do dataset nano-only (NANO_MAX_NM=%.1f) ===", NANO_MAX_NM)

    if DIR_SAIDA.exists():
        REGISTRADOR.warning("removendo %s existente antes de reconstruir", DIR_SAIDA)
        shutil.rmtree(DIR_SAIDA)

    estatisticas = {}
    for particao in ("train", "val"):
        estatisticas[particao] = processar_particao(particao)
    estatisticas["test"] = copiar_teste_literal()

    caminho_yaml_dados = DIR_SAIDA / "data.yaml"
    caminho_yaml_dados.write_text(
        f"path: {DIR_SAIDA}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(NOMES_CLASSES)}\n"
        f"names:\n" + "\n".join(f"  - {c}" for c in NOMES_CLASSES) + "\n"
    )
    print(f"\ndata.yaml escrito em {caminho_yaml_dados}")

    salvar_resumo_experimento(NOME_EXP, METADADOS_EXECUCAO, {
        "nano_max_nm": NANO_MAX_NM,
        "output_dir": str(DIR_SAIDA),
        "data_yaml": str(caminho_yaml_dados),
        "stats": estatisticas,
    })
    print(f"Resumo salvo em {DIR_EXP / 'summary.json'}")


if __name__ == "__main__":
    principal()
