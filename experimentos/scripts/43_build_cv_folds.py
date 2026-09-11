"""Item 3 do plano de ação (ICs mais robustos pro recall médio/grande):
constrói K=5 folds de cross-validation por imagem-fonte a partir das 102
imagens únicas do dataset MiNa inteiro (train+val+test de
datasets_yolo26_v2/), pra depois treinar 5 detectores YOLO26L
(44_train_cv_fold.py) e avaliar cada imagem SÓ no fold em que ela nunca foi
vista (45_eval_cv_folds.py) -- assim dá pra medir recall por tamanho sobre
155 partículas médias e 30 grandes (vs. as 31/5 da partição de teste
travada única usada no artigo), sem viés de treino/seleção de checkpoint.

Deduplicação: o treino de datasets_yolo26_v2/ tem 90 arquivos de imagem PE
que são duplicatas byte-idênticas (dup1_..dup5_ + original, 6 cópias de
cada uma das 18 imagens PE únicas -- balanceamento de classe do treino
original, ver 13_train_yolo26_corrected.py). Agrupamos por nome-base (sem o
prefixo dupN_) ANTES de sortear os folds, pra nunca ter a mesma
imagem-fonte em papéis diferentes (treino de um fold, held-out de outro
fold, ou pior, treino E held-out do MESMO fold) -- confirmado: 102 imagens
únicas bate com o "102 imagens do nosso split inteiro" citado no docstring
de 13_train_yolo26_corrected.py.

Split por imagem-fonte (nunca por partícula), estratificado por classe,
mesma lógica de 00_split_images.py (seed=42) -- partículas da mesma
micrografia compartilham textura de fundo, misturar entre papéis vazaria
informação.
"""
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import save_experiment_summary, setup_experiment_logging

EXP_NAME = "43_build_cv_folds"
EXP_DIR, LOG_PATH, LOGGER, RUNTIME_METADATA = setup_experiment_logging(EXP_NAME, __file__)

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "datasets_yolo26_v2"
OUT_DIR = ROOT / "datasets_yolo26_v2_cv"
CLASS_NAMES = ["PE", "PET", "PP", "PS"]
K_FOLDS = 5
VAL_FRACTION_OF_POOL = 0.15  # mesma fracao de val do split original
SEED = 42
DUP_RE = re.compile(r"^dup\d+_")

SMALL_THRESHOLD = 0.01
MEDIUM_THRESHOLD = 0.05


def classify_size(area_norm: float) -> str:
    if area_norm < SMALL_THRESHOLD:
        return "small"
    if area_norm < MEDIUM_THRESHOLD:
        return "medium"
    return "large"


def collect_unique_images():
    """Varre images/labels de train+val+test, agrupa por nome-base
    deduplicado. Retorna dict base_name -> {"class", "variants": [(img_path,
    label_path, original_split), ...], "n_particles_by_size": {...}}."""
    groups = {}
    for split in ("train", "val", "test"):
        img_dir = SRC_DIR / "images" / split
        label_dir = SRC_DIR / "labels" / split
        for img_path in sorted(img_dir.glob("*.png")):
            base = DUP_RE.sub("", img_path.stem)
            label_path = label_dir / f"{img_path.stem}.txt"
            groups.setdefault(base, {"class": base.split("_", 1)[0], "variants": []})
            groups[base]["variants"].append((img_path, label_path, split))

    for base, info in groups.items():
        # particulas contadas UMA vez por imagem-fonte, a partir de
        # qualquer uma das variantes (todas identicas -- so pega a 1a)
        _, label_path, _ = info["variants"][0]
        buckets = defaultdict(int)
        if label_path.exists():
            for line in open(label_path):
                parts = line.split()
                if len(parts) < 5:
                    continue
                w, h = float(parts[3]), float(parts[4])
                buckets[classify_size(w * h)] += 1
        info["n_particles_by_size"] = dict(buckets)
    return groups


def stratified_fold_assignment(groups, k_folds, seed):
    """Distribui as imagens-fonte em k folds, estratificado por classe
    (mesmo espirito de 00_split_images.py: embaralha dentro de cada classe,
    distribui round-robin)."""
    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)
    for base, info in groups.items():
        by_class[info["class"]].append(base)

    fold_of = {}
    for cls, bases in by_class.items():
        bases = sorted(bases)
        order = rng.permutation(len(bases))
        for rank, idx in enumerate(order):
            fold_of[bases[idx]] = rank % k_folds
    return fold_of


def split_pool_train_val(pool_bases, val_fraction, seed, groups):
    """Dentro do pool de imagens-fonte FORA do fold k, separa uma fatia de
    val (sem oversampling), estratificado por classe, mesma logica de
    00_split_images.py."""
    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)
    for base in pool_bases:
        by_class[groups[base]["class"]].append(base)

    val_bases = set()
    for cls, bases in by_class.items():
        bases = sorted(bases)
        order = rng.permutation(len(bases))
        n_val = max(1, round(len(bases) * val_fraction)) if len(bases) > 2 else 0
        for idx in order[:n_val]:
            val_bases.add(bases[idx])
    train_bases = set(pool_bases) - val_bases
    return train_bases, val_bases


def materialize_fold(fold_k, fold_of, groups):
    fold_dir = OUT_DIR / f"fold{fold_k}"
    for role in ("train", "val", "test"):
        (fold_dir / "images" / role).mkdir(parents=True, exist_ok=True)
        (fold_dir / "labels" / role).mkdir(parents=True, exist_ok=True)

    held_out_bases = [b for b, f in fold_of.items() if f == fold_k]
    pool_bases = [b for b, f in fold_of.items() if f != fold_k]
    train_bases, val_bases = split_pool_train_val(pool_bases, VAL_FRACTION_OF_POOL, SEED + fold_k, groups)

    def copy_role(base, role, oversample):
        info = groups[base]
        variants = info["variants"] if oversample else info["variants"][:1]
        for img_path, label_path, _orig_split in variants:
            dst_img = fold_dir / "images" / role / img_path.name
            dst_label = fold_dir / "labels" / role / label_path.name
            shutil.copy2(img_path, dst_img)
            if label_path.exists():
                shutil.copy2(label_path, dst_label)

    for base in train_bases:
        copy_role(base, "train", oversample=True)
    for base in val_bases:
        copy_role(base, "val", oversample=False)
    for base in held_out_bases:
        copy_role(base, "test", oversample=False)

    data_yaml = fold_dir / "data.yaml"
    data_yaml.write_text(
        f"path: {fold_dir}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n" + "\n".join(f"  - {c}" for c in CLASS_NAMES) + "\n"
    )

    return {
        "fold": fold_k,
        "n_train_sources": len(train_bases),
        "n_val_sources": len(val_bases),
        "n_test_sources": len(held_out_bases),
        "n_train_files": sum(len(groups[b]["variants"]) for b in train_bases),
    }


def main():
    LOGGER.info("=== construindo %d folds de cross-validation ===", K_FOLDS)
    groups = collect_unique_images()
    n_unique = len(groups)
    LOGGER.info("imagens-fonte unicas (deduplicadas): %d", n_unique)
    print(f"imagens-fonte unicas (deduplicadas): {n_unique}")

    class_counts = defaultdict(int)
    bucket_totals = defaultdict(int)
    for base, info in groups.items():
        class_counts[info["class"]] += 1
        for size, n in info["n_particles_by_size"].items():
            bucket_totals[size] += n
    print(f"por classe: {dict(class_counts)}")
    print(f"total de particulas por tamanho (dataset inteiro, dedup): {dict(bucket_totals)}")

    fold_of = stratified_fold_assignment(groups, K_FOLDS, SEED)

    # checagem de sanidade: toda imagem-fonte em exatamente 1 fold
    assert set(fold_of.keys()) == set(groups.keys())
    fold_sizes = defaultdict(int)
    for base, f in fold_of.items():
        fold_sizes[f] += 1
    print(f"\ntamanho de cada fold (imagens-fonte held-out): {dict(sorted(fold_sizes.items()))}")

    if OUT_DIR.exists():
        LOGGER.warning("removendo %s existente antes de reconstruir", OUT_DIR)
        shutil.rmtree(OUT_DIR)

    fold_reports = []
    for k in range(K_FOLDS):
        report = materialize_fold(k, fold_of, groups)
        fold_reports.append(report)
        print(f"fold {k}: treino={report['n_train_sources']} imagens-fonte "
              f"({report['n_train_files']} arquivos c/ oversampling PE), "
              f"val={report['n_val_sources']}, held-out(test)={report['n_test_sources']}")
        LOGGER.info("fold=%d relatorio=%s", k, report)

    assignment_csv = EXP_DIR / "data" / "fold_assignment.csv"
    assignment_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(assignment_csv, "w") as f:
        f.write("source_image,class,fold,n_variants,n_small,n_medium,n_large\n")
        for base, info in sorted(groups.items()):
            b = info["n_particles_by_size"]
            f.write(f"{base},{info['class']},{fold_of[base]},{len(info['variants'])},"
                    f"{b.get('small',0)},{b.get('medium',0)},{b.get('large',0)}\n")

    save_experiment_summary(EXP_NAME, RUNTIME_METADATA, {
        "n_unique_source_images": n_unique,
        "class_counts": dict(class_counts),
        "particle_bucket_totals": dict(bucket_totals),
        "k_folds": K_FOLDS,
        "val_fraction_of_pool": VAL_FRACTION_OF_POOL,
        "seed": SEED,
        "fold_sizes_held_out": dict(sorted(fold_sizes.items())),
        "fold_reports": fold_reports,
        "output_dir": str(OUT_DIR),
        "assignment_csv": str(assignment_csv),
    })
    print(f"\nfolds materializados em {OUT_DIR}")
    print(f"assignment: {assignment_csv}")
    print(f"resumo: {EXP_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
