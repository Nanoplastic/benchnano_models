"""Construção das dobras de validação cruzada / Cross-validation fold construction.

PT: Constrói K=5 dobras de validação cruzada por imagem de origem a partir das 102 imagens
    únicas do MiNa (train+val+test de datasets_yolo26_v2/). Cada dobra treina um YOLO26L
    (44_train_cv_fold.py) e cada imagem é avaliada só na dobra em que nunca foi vista
    (45_eval_cv_folds.py). Isso permite medir o recall por tamanho sobre 155 partículas
    médias e 30 grandes (contra 31/5 da partição de teste fixa), sem viés de treino ou de
    seleção de checkpoint.
    Deduplicação: o treino de datasets_yolo26_v2/ tem 90 arquivos PE que são cópias
    idênticas byte a byte (dup1_..dup5_ + original: 6 cópias de cada uma das 18 imagens PE
    únicas, para balanceamento de classe). As imagens são agrupadas pelo nome-base (sem o
    prefixo dupN_) antes do sorteio das dobras, para que a mesma imagem de origem nunca
    apareça em papéis diferentes (treino de uma dobra e held-out de outra, ou treino e
    held-out da mesma dobra). O total de 102 imagens únicas confirma o agrupamento.
    Divisão por imagem de origem (nunca por partícula), estratificada por classe, mesma
    lógica de 00_split_images.py (semente 42).
EN: Builds K=5 source-image cross-validation folds from the 102 unique MiNa images
    (train+val+test of datasets_yolo26_v2/). Each fold trains one YOLO26L
    (44_train_cv_fold.py) and each image is evaluated only in the fold where it was never
    seen (45_eval_cv_folds.py). This allows measuring recall by size over 155 medium and 30
    large particles (vs. 31/5 in the fixed test split), with no training or checkpoint
    selection bias.
    Deduplication: the datasets_yolo26_v2/ training set has 90 PE files that are
    byte-identical copies (dup1_..dup5_ + original: 6 copies of each of the 18 unique PE
    images, for class balancing). Images are grouped by base name (without the dupN_ prefix)
    before the folds are drawn, so that the same source image never appears in different
    roles (training in one fold and held-out in another, or training and held-out in the
    same fold). The total of 102 unique images confirms the grouping.
    Split by source image (never by particle), stratified by class, same logic as
    00_split_images.py (seed 42).
"""
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import salvar_resumo_experimento, configurar_log_experimento

NOME_EXP = "43_build_cv_folds"
DIR_EXP, CAMINHO_LOG, REGISTRADOR, METADADOS_EXECUCAO = configurar_log_experimento(NOME_EXP, __file__)

RAIZ = Path(__file__).resolve().parent.parent
DIR_ORIGEM = RAIZ / "datasets_yolo26_v2"
DIR_SAIDA = RAIZ / "datasets_yolo26_v2_cv"
NOMES_CLASSES = ["PE", "PET", "PP", "PS"]
K_DOBRAS = 5
FRACAO_VAL_DO_CONJUNTO = 0.15  # PT: mesma fração de validação da divisão original / EN: same validation fraction as the original split
SEMENTE = 42
RE_DUPLICADO = re.compile(r"^dup\d+_")

LIMIAR_PEQUENO = 0.01
LIMIAR_MEDIO = 0.05


def classificar_tamanho(area_normalizada: float) -> str:
    if area_normalizada < LIMIAR_PEQUENO:
        return "small"
    if area_normalizada < LIMIAR_MEDIO:
        return "medium"
    return "large"


def coletar_imagens_unicas():
    """PT: Varre images/labels de train+val+test e agrupa pelo nome-base deduplicado. Retorna
        dict nome_base -> {"class", "variants": [(caminho_img, caminho_rotulo,
        particao_original), ...], "n_particles_by_size": {...}}.
    EN: Scans images/labels of train+val+test and groups by deduplicated base name. Returns
        dict base_name -> {"class", "variants": [(img_path, label_path, original_split),
        ...], "n_particles_by_size": {...}}.
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
        # PT: partículas contadas uma vez por imagem de origem, a partir da primeira
        #     variante (todas são idênticas)
        # EN: particles counted once per source image, from the first variant (all are
        #     identical)
        _, caminho_rotulo, _ = info["variants"][0]
        faixas = defaultdict(int)
        if caminho_rotulo.exists():
            for linha in open(caminho_rotulo):
                partes = linha.split()
                if len(partes) < 5:
                    continue
                w, h = float(partes[3]), float(partes[4])
                faixas[classificar_tamanho(w * h)] += 1
        info["n_particles_by_size"] = dict(faixas)
    return grupos


def atribuicao_dobras_estratificada(grupos, k_dobras, semente):
    """PT: Distribui as imagens de origem em k dobras, estratificando por classe (como em
        00_split_images.py: embaralha dentro de cada classe e distribui em rodízio).
    EN: Distributes source images into k folds, stratified by class (as in
        00_split_images.py: shuffles within each class and deals round-robin).
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
    """PT: Dentro do conjunto de imagens de origem FORA da dobra k, separa uma fatia de
        validação (sem sobreamostragem), estratificada por classe, como em
        00_split_images.py.
    EN: Within the pool of source images OUTSIDE fold k, sets aside a validation slice (no
        oversampling), stratified by class, as in 00_split_images.py.
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


def materializar_dobra(dobra_k, dobra_de, grupos):
    dir_dobra = DIR_SAIDA / f"fold{dobra_k}"
    for papel in ("train", "val", "test"):
        (dir_dobra / "images" / papel).mkdir(parents=True, exist_ok=True)
        (dir_dobra / "labels" / papel).mkdir(parents=True, exist_ok=True)

    bases_reservadas = [b for b, f in dobra_de.items() if f == dobra_k]
    bases_conjunto = [b for b, f in dobra_de.items() if f != dobra_k]
    bases_treino, bases_val = dividir_conjunto_treino_val(bases_conjunto, FRACAO_VAL_DO_CONJUNTO, SEMENTE + dobra_k, grupos)

    def copiar_papel(base, papel, sobreamostrar):
        info = grupos[base]
        variantes = info["variants"] if sobreamostrar else info["variants"][:1]
        for caminho_img, caminho_rotulo, _particao_original in variantes:
            img_destino = dir_dobra / "images" / papel / caminho_img.name
            rotulo_destino = dir_dobra / "labels" / papel / caminho_rotulo.name
            shutil.copy2(caminho_img, img_destino)
            if caminho_rotulo.exists():
                shutil.copy2(caminho_rotulo, rotulo_destino)

    for base in bases_treino:
        copiar_papel(base, "train", sobreamostrar=True)
    for base in bases_val:
        copiar_papel(base, "val", sobreamostrar=False)
    for base in bases_reservadas:
        copiar_papel(base, "test", sobreamostrar=False)

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
        "n_train_files": sum(len(grupos[b]["variants"]) for b in bases_treino),
    }


def principal():
    REGISTRADOR.info("=== construindo %d folds de cross-validation ===", K_DOBRAS)
    grupos = coletar_imagens_unicas()
    n_unicos = len(grupos)
    REGISTRADOR.info("imagens-fonte unicas (deduplicadas): %d", n_unicos)
    print(f"imagens-fonte unicas (deduplicadas): {n_unicos}")

    contagens_classe = defaultdict(int)
    totais_faixa = defaultdict(int)
    for base, info in grupos.items():
        contagens_classe[info["class"]] += 1
        for tamanho, n in info["n_particles_by_size"].items():
            totais_faixa[tamanho] += n
    print(f"por classe: {dict(contagens_classe)}")
    print(f"total de particulas por tamanho (dataset inteiro, dedup): {dict(totais_faixa)}")

    dobra_de = atribuicao_dobras_estratificada(grupos, K_DOBRAS, SEMENTE)

    # PT: verificação: toda imagem de origem está em exatamente uma dobra
    # EN: check: every source image is in exactly one fold
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
              f"({relatorio['n_train_files']} arquivos c/ oversampling PE), "
              f"val={relatorio['n_val_sources']}, held-out(test)={relatorio['n_test_sources']}")
        REGISTRADOR.info("fold=%d relatorio=%s", k, relatorio)

    csv_atribuicao = DIR_EXP / "data" / "fold_assignment.csv"
    csv_atribuicao.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_atribuicao, "w") as f:
        f.write("source_image,class,fold,n_variants,n_small,n_medium,n_large\n")
        for base, info in sorted(grupos.items()):
            b = info["n_particles_by_size"]
            f.write(f"{base},{info['class']},{dobra_de[base]},{len(info['variants'])},"
                    f"{b.get('small',0)},{b.get('medium',0)},{b.get('large',0)}\n")

    salvar_resumo_experimento(NOME_EXP, METADADOS_EXECUCAO, {
        "n_unique_source_images": n_unicos,
        "class_counts": dict(contagens_classe),
        "particle_bucket_totals": dict(totais_faixa),
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
