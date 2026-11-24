"""Dobras de validação cruzada só com nanopartículas / Nano-only cross-validation folds.

PT: Constrói 5 dobras de validação cruzada só com partículas < 1000 nm, com a mesma partição
    de imagens de origem, a mesma estratificação por classe e a mesma semente (42) de
    43_build_cv_folds.py, para comparação 1:1 dobra a dobra. Ao materializar os papéis
    "train"/"val" de cada dobra, filtra as caixas para o subconjunto nano (calibração px->nm
    = 100/ampliação e limiar NANO_MAX_NM = 1000 nm de 23_recall_by_size_regime.py).
    O papel "test" (held-out) de cada dobra é copiado literalmente, sem filtro; o filtro
    nano é aplicado só na avaliação (51_eval_cv_folds_nano.py), para não alterar o conjunto
    usado para medir o recall.
    Imagens sem nenhuma caixa nano são excluídas de train/val da dobra em que caíram, mas
    continuam presentes na dobra em que são held-out.
    Escreve só em datasets_yolo26_v2_cv_nano/ e resultados/49_build_nano_cv_folds/
    (metadados).
EN: Builds 5 cross-validation folds with only < 1000 nm particles, using the same
    source-image partition, class stratification and seed (42) as 43_build_cv_folds.py, for
    a 1:1 fold-by-fold comparison. When materializing each fold's "train"/"val" roles, boxes
    are filtered to the nano subset (px->nm = 100/magnification calibration and NANO_MAX_NM
    = 1000 nm threshold from 23_recall_by_size_regime.py).
    Each fold's "test" (held-out) role is copied verbatim, unfiltered; the nano filter is
    applied only at evaluation (51_eval_cv_folds_nano.py), so the set used to measure recall
    is never altered.
    Images with no nano box are dropped from the train/val of the fold they fall in, but
    remain present in the fold where they are held out.
    Writes only to datasets_yolo26_v2_cv_nano/ and resultados/49_build_nano_cv_folds/
    (metadata).
"""
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import salvar_resumo_experimento, configurar_log_experimento

NOME_EXP = "49_build_nano_cv_folds"
DIR_EXP, CAMINHO_LOG, REGISTRADOR, METADADOS_EXECUCAO = configurar_log_experimento(NOME_EXP, __file__)

RAIZ = Path(__file__).resolve().parent.parent
DIR_ORIGEM = RAIZ / "datasets_yolo26_v2"
DIR_SAIDA = RAIZ / "datasets_yolo26_v2_cv_nano"
NOMES_CLASSES = ["PE", "PET", "PP", "PS"]
K_DOBRAS = 5
FRACAO_VAL_DO_CONJUNTO = 0.15
SEMENTE = 42
RE_DUPLICADO = re.compile(r"^dup\d+_")
NANO_MAX_NM = 1000.0

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


def coletar_imagens_unicas():
    """PT: Como 43_build_cv_folds.coletar_imagens_unicas, mas conta partículas por regime
        físico (nano/micro), não por faixa de área normalizada; usa a ampliação e o tamanho
        em pixels da imagem para converter.
    EN: Like 43_build_cv_folds.coletar_imagens_unicas, but counts particles by physical
        regime (nano/micro), not by normalized area band; uses the image magnification and
        pixel size to convert.
    """
    grupos = {}
    for particao in ("train", "val", "test"):
        dir_img = DIR_ORIGEM / "images" / particao
        dir_rotulo = DIR_ORIGEM / "labels" / particao
        for caminho_img in sorted(dir_img.glob("*.png")):
            base = RE_DUPLICADO.sub("", caminho_img.stem)
            caminho_rotulo = dir_rotulo / f"{caminho_img.stem}.txt"
            grupos.setdefault(base, {"class": base.split("_", 1)[0], "variants": []})
            grupos[base]["variants"].append((caminho_img, caminho_rotulo, particao))

    for base, info in grupos.items():
        caminho_img, caminho_rotulo, _ = info["variants"][0]
        ampl = interpretar_ampliacao(base)
        n_nano, n_micro = 0, 0
        if ampl is not None and caminho_rotulo.exists():
            um_por_px = 100.0 / ampl
            with Image.open(caminho_img) as im:
                W, H = im.size
            for linha in open(caminho_rotulo):
                partes = linha.split()
                if len(partes) < 5:
                    continue
                w, h = float(partes[3]), float(partes[4])
                tamanho_px = max(w * W, h * H)
                tamanho_nm = tamanho_px * um_por_px * 1000.0
                if tamanho_nm < NANO_MAX_NM:
                    n_nano += 1
                else:
                    n_micro += 1
        info["magnification"] = ampl
        info["n_nano"] = n_nano
        info["n_micro"] = n_micro
    return grupos


def atribuicao_dobras_estratificada(grupos, k_dobras, semente):
    """PT: Idêntico a 43_build_cv_folds.atribuicao_dobras_estratificada (mesma semente, mesma
        lógica), para manter as 5 dobras comparáveis 1:1 com a validação cruzada original.
    EN: Identical to 43_build_cv_folds.atribuicao_dobras_estratificada (same seed, same
        logic), so the 5 folds stay 1:1 comparable with the original cross-validation.
    """
    gerador = np.random.default_rng(semente)
    por_classe = defaultdict(list)
    for base, info in grupos.items():
        por_classe[info["class"]].append(base)

    dobra_de = {}
    for cls, bases_img in por_classe.items():
        bases_img = sorted(bases_img)
        ordem = gerador.permutation(len(bases_img))
        for posto, indice in enumerate(ordem):
            dobra_de[bases_img[indice]] = posto % k_dobras
    return dobra_de


def dividir_conjunto_treino_val(bases_conjunto, fracao_val, semente, grupos):
    """PT: Idêntico a 43_build_cv_folds.dividir_conjunto_treino_val.
    EN: Identical to 43_build_cv_folds.dividir_conjunto_treino_val.
    """
    gerador = np.random.default_rng(semente)
    por_classe = defaultdict(list)
    for base in bases_conjunto:
        por_classe[grupos[base]["class"]].append(base)

    bases_val = set()
    for cls, bases_img in por_classe.items():
        bases_img = sorted(bases_img)
        ordem = gerador.permutation(len(bases_img))
        n_val = max(1, round(len(bases_img) * fracao_val)) if len(bases_img) > 2 else 0
        for indice in ordem[:n_val]:
            bases_val.add(bases_img[indice])
    bases_treino = set(bases_conjunto) - bases_val
    return bases_treino, bases_val


def filtrar_linhas_nano(caminho_rotulo: Path, W: int, H: int, um_por_px: float):
    if not caminho_rotulo.exists():
        return []
    mantidos = []
    for linha in open(caminho_rotulo):
        partes = linha.split()
        if len(partes) < 5:
            continue
        id_cls = int(partes[0])
        cx, cy, w, h = map(float, partes[1:5])
        tamanho_px = max(w * W, h * H)
        tamanho_nm = tamanho_px * um_por_px * 1000.0
        if tamanho_nm < NANO_MAX_NM:
            mantidos.append(f"{id_cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    return mantidos


def materializar_dobra(dobra_k, dobra_de, grupos):
    dir_dobra = DIR_SAIDA / f"fold{dobra_k}"
    for papel in ("train", "val", "test"):
        (dir_dobra / "images" / papel).mkdir(parents=True, exist_ok=True)
        (dir_dobra / "labels" / papel).mkdir(parents=True, exist_ok=True)

    bases_reservadas = [b for b, f in dobra_de.items() if f == dobra_k]
    bases_conjunto = [b for b, f in dobra_de.items() if f != dobra_k]
    bases_treino, bases_val = dividir_conjunto_treino_val(bases_conjunto, FRACAO_VAL_DO_CONJUNTO, SEMENTE + dobra_k, grupos)

    def copiar_papel_filtrado_nano(base, papel, sobreamostrar):
        info = grupos[base]
        ampl = info["magnification"]
        variantes = info["variants"] if sobreamostrar else info["variants"][:1]
        n_mantidos = 0
        for caminho_img, caminho_rotulo, _particao_original in variantes:
            if ampl is None:
                continue
            um_por_px = 100.0 / ampl
            with Image.open(caminho_img) as im:
                W, H = im.size
            linhas_mantidas = filtrar_linhas_nano(caminho_rotulo, W, H, um_por_px)
            if not linhas_mantidas:
                continue
            img_destino = dir_dobra / "images" / papel / caminho_img.name
            rotulo_destino = dir_dobra / "labels" / papel / caminho_rotulo.name
            shutil.copy2(caminho_img, img_destino)
            with open(rotulo_destino, "w") as f:
                f.writelines(linhas_mantidas)
            n_mantidos += 1
        return n_mantidos

    def copiar_papel_literal(base, papel, sobreamostrar):
        info = grupos[base]
        variantes = info["variants"] if sobreamostrar else info["variants"][:1]
        for caminho_img, caminho_rotulo, _particao_original in variantes:
            img_destino = dir_dobra / "images" / papel / caminho_img.name
            rotulo_destino = dir_dobra / "labels" / papel / caminho_rotulo.name
            shutil.copy2(caminho_img, img_destino)
            if caminho_rotulo.exists():
                shutil.copy2(caminho_rotulo, rotulo_destino)

    n_treino_mantidos = sum(copiar_papel_filtrado_nano(b, "train", sobreamostrar=True) for b in bases_treino)
    n_val_mantidos = sum(copiar_papel_filtrado_nano(b, "val", sobreamostrar=False) for b in bases_val)
    for base in bases_reservadas:
        copiar_papel_literal(base, "test", sobreamostrar=False)  # PT: sem filtro nano; o filtro é aplicado em 51, na avaliação / EN: no nano filter; filtering happens in 51, at evaluation

    yaml_dados = dir_dobra / "data.yaml"
    yaml_dados.write_text(
        f"path: {dir_dobra}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(NOMES_CLASSES)}\n"
        f"names:\n" + "\n".join(f"  - {c}" for c in NOMES_CLASSES) + "\n"
    )

    return {
        "fold": dobra_k,
        "n_train_sources": len(bases_treino),
        "n_val_sources": len(bases_val),
        "n_test_sources": len(bases_reservadas),
        "n_train_files_kept_nano": n_treino_mantidos,
        "n_val_files_kept_nano": n_val_mantidos,
    }


def principal():
    REGISTRADOR.info("=== construindo %d folds de cross-validation nano-only ===", K_DOBRAS)
    grupos = coletar_imagens_unicas()
    n_unicos = len(grupos)
    REGISTRADOR.info("imagens-fonte unicas (deduplicadas): %d", n_unicos)
    print(f"imagens-fonte unicas (deduplicadas): {n_unicos}")

    contagens_classe = defaultdict(int)
    n_nano_total = sum(info["n_nano"] for info in grupos.values())
    n_micro_total = sum(info["n_micro"] for info in grupos.values())
    for base, info in grupos.items():
        contagens_classe[info["class"]] += 1
    print(f"por classe: {dict(contagens_classe)}")
    print(f"caixas nano/micro no dataset inteiro (dedup): nano={n_nano_total} micro={n_micro_total}")

    dobra_de = atribuicao_dobras_estratificada(grupos, K_DOBRAS, SEMENTE)
    assert set(dobra_de.keys()) == set(grupos.keys())
    tamanhos_dobras = defaultdict(int)
    for base, f in dobra_de.items():
        tamanhos_dobras[f] += 1
    print(f"\ntamanho de cada fold (imagens-fonte held-out): {dict(sorted(tamanhos_dobras.items()))}")

    if DIR_SAIDA.exists():
        REGISTRADOR.warning("removendo %s existente antes de reconstruir", DIR_SAIDA)
        shutil.rmtree(DIR_SAIDA)

    relatorios_dobras = []
    for k in range(K_DOBRAS):
        relatorio = materializar_dobra(k, dobra_de, grupos)
        relatorios_dobras.append(relatorio)
        print(f"fold {k}: treino={relatorio['n_train_sources']} imagens-fonte "
              f"({relatorio['n_train_files_kept_nano']} arquivos com >=1 caixa nano), "
              f"val={relatorio['n_val_sources']} ({relatorio['n_val_files_kept_nano']} com nano), "
              f"held-out(test)={relatorio['n_test_sources']} (nao filtrado)")
        REGISTRADOR.info("fold=%d relatorio=%s", k, relatorio)

    csv_atribuicao = DIR_EXP / "data" / "fold_assignment.csv"
    csv_atribuicao.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_atribuicao, "w") as f:
        f.write("source_image,class,fold,n_variants,n_nano,n_micro\n")
        for base, info in sorted(grupos.items()):
            f.write(f"{base},{info['class']},{dobra_de[base]},{len(info['variants'])},"
                    f"{info['n_nano']},{info['n_micro']}\n")

    salvar_resumo_experimento(NOME_EXP, METADADOS_EXECUCAO, {
        "nano_max_nm": NANO_MAX_NM,
        "n_unique_source_images": n_unicos,
        "class_counts": dict(contagens_classe),
        "n_nano_boxes_total": n_nano_total,
        "n_micro_boxes_total": n_micro_total,
        "k_folds": K_DOBRAS,
        "val_fraction_of_pool": FRACAO_VAL_DO_CONJUNTO,
        "seed": SEMENTE,
        "fold_sizes_held_out": dict(sorted(tamanhos_dobras.items())),
        "fold_reports": relatorios_dobras,
        "output_dir": str(DIR_SAIDA),
        "assignment_csv": str(csv_atribuicao),
    })
    print(f"\nfolds materializados em {DIR_SAIDA}")
    print(f"assignment: {csv_atribuicao}")
    print(f"resumo: {DIR_EXP / 'summary.json'}")


if __name__ == "__main__":
    principal()
