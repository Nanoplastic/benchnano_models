"""Constrói 5 folds de cross-validation NANO-ONLY (partículas <1000nm) --
mesma partição de imagens-fonte, MESMA estratificação por classe e MESMO
seed=42 de 43_build_cv_folds.py (pra ficar comparável 1:1, fold a fold, com
a CV original), só que ao materializar os papéis "train"/"val" de cada fold
filtra as caixas de anotação pro subconjunto nano (mesma calibração
px->nm=100/ampliação e limiar NANO_MAX_NM=1000nm de
23_recall_by_size_regime.py).

O papel "test" (held-out) de cada fold é copiado VERBATIM, sem filtrar -- o
filtro nano nessa partição é aplicado só na hora de avaliar
(51_eval_cv_folds_nano.py), pra nunca alterar fisicamente o conjunto usado
pra medir recall.

Imagens que ficam sem nenhuma caixa nano são excluídas do papel train/val do
fold em que caíram (mas continuam presentes normalmente no fold em que são
held-out, já que lá não são filtradas).

Não sobrescreve nada: escreve só em datasets_yolo26_v2_cv_nano/ (novo) e
resultados/49_build_nano_cv_folds/ (metadados).
"""
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import save_experiment_summary, setup_experiment_logging

EXP_NAME = "49_build_nano_cv_folds"
EXP_DIR, LOG_PATH, LOGGER, RUNTIME_METADATA = setup_experiment_logging(EXP_NAME, __file__)

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "datasets_yolo26_v2"
OUT_DIR = ROOT / "datasets_yolo26_v2_cv_nano"
CLASS_NAMES = ["PE", "PET", "PP", "PS"]
K_FOLDS = 5
VAL_FRACTION_OF_POOL = 0.15
SEED = 42
DUP_RE = re.compile(r"^dup\d+_")
NANO_MAX_NM = 1000.0

MAG_X = re.compile(r"(\d+)\s*[xX]")
MAG_DASH = re.compile(r"^\D*(\d+)-\d+")


def parse_magnification(name: str):
    m = MAG_X.findall(name)
    if m:
        return int(m[-1])
    m = MAG_DASH.match(name)
    if m:
        return int(m.group(1))
    return None


def collect_unique_images():
    """Igual a 43_build_cv_folds.collect_unique_images, mas conta partículas
    por regime FÍSICO (nano/micro), não por bucket de área normalizada --
    precisa da magnificação e do tamanho em pixel da imagem pra converter."""
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
        img_path, label_path, _ = info["variants"][0]
        mag = parse_magnification(base)
        n_nano, n_micro = 0, 0
        if mag is not None and label_path.exists():
            um_per_px = 100.0 / mag
            with Image.open(img_path) as im:
                W, H = im.size
            for line in open(label_path):
                parts = line.split()
                if len(parts) < 5:
                    continue
                w, h = float(parts[3]), float(parts[4])
                px_size = max(w * W, h * H)
                size_nm = px_size * um_per_px * 1000.0
                if size_nm < NANO_MAX_NM:
                    n_nano += 1
                else:
                    n_micro += 1
        info["magnification"] = mag
        info["n_nano"] = n_nano
        info["n_micro"] = n_micro
    return groups


def stratified_fold_assignment(groups, k_folds, seed):
    """IDÊNTICO a 43_build_cv_folds.stratified_fold_assignment -- mesmo
    seed, mesma lógica -- pra manter os 5 folds comparáveis 1:1 com a CV
    original."""
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
    """IDÊNTICO a 43_build_cv_folds.split_pool_train_val."""
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


def filter_lines_to_nano(label_path: Path, W: int, H: int, um_per_px: float):
    if not label_path.exists():
        return []
    kept = []
    for line in open(label_path):
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        px_size = max(w * W, h * H)
        size_nm = px_size * um_per_px * 1000.0
        if size_nm < NANO_MAX_NM:
            kept.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    return kept


def materialize_fold(fold_k, fold_of, groups):
    fold_dir = OUT_DIR / f"fold{fold_k}"
    for role in ("train", "val", "test"):
        (fold_dir / "images" / role).mkdir(parents=True, exist_ok=True)
        (fold_dir / "labels" / role).mkdir(parents=True, exist_ok=True)

    held_out_bases = [b for b, f in fold_of.items() if f == fold_k]
    pool_bases = [b for b, f in fold_of.items() if f != fold_k]
    train_bases, val_bases = split_pool_train_val(pool_bases, VAL_FRACTION_OF_POOL, SEED + fold_k, groups)

    def copy_role_nano_filtered(base, role, oversample):
        info = groups[base]
        mag = info["magnification"]
        variants = info["variants"] if oversample else info["variants"][:1]
        n_kept = 0
        for img_path, label_path, _orig_split in variants:
            if mag is None:
                continue
            um_per_px = 100.0 / mag
            with Image.open(img_path) as im:
                W, H = im.size
            kept_lines = filter_lines_to_nano(label_path, W, H, um_per_px)
            if not kept_lines:
                continue
            dst_img = fold_dir / "images" / role / img_path.name
            dst_label = fold_dir / "labels" / role / label_path.name
            shutil.copy2(img_path, dst_img)
            with open(dst_label, "w") as f:
                f.writelines(kept_lines)
            n_kept += 1
        return n_kept

    def copy_role_verbatim(base, role, oversample):
        info = groups[base]
        variants = info["variants"] if oversample else info["variants"][:1]
        for img_path, label_path, _orig_split in variants:
            dst_img = fold_dir / "images" / role / img_path.name
            dst_label = fold_dir / "labels" / role / label_path.name
            shutil.copy2(img_path, dst_img)
            if label_path.exists():
                shutil.copy2(label_path, dst_label)

    n_train_kept = sum(copy_role_nano_filtered(b, "train", oversample=True) for b in train_bases)
    n_val_kept = sum(copy_role_nano_filtered(b, "val", oversample=False) for b in val_bases)
    for base in held_out_bases:
        copy_role_verbatim(base, "test", oversample=False)  # SEM filtro nano -- filtrado em 51 na avaliacao

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
        "n_train_files_kept_nano": n_train_kept,
        "n_val_files_kept_nano": n_val_kept,
    }


def main():
    LOGGER.info("=== construindo %d folds de cross-validation nano-only ===", K_FOLDS)
    groups = collect_unique_images()
    n_unique = len(groups)
    LOGGER.info("imagens-fonte unicas (deduplicadas): %d", n_unique)
    print(f"imagens-fonte unicas (deduplicadas): {n_unique}")

    class_counts = defaultdict(int)
    n_nano_total = sum(info["n_nano"] for info in groups.values())
    n_micro_total = sum(info["n_micro"] for info in groups.values())
    for base, info in groups.items():
        class_counts[info["class"]] += 1
    print(f"por classe: {dict(class_counts)}")
    print(f"caixas nano/micro no dataset inteiro (dedup): nano={n_nano_total} micro={n_micro_total}")

    fold_of = stratified_fold_assignment(groups, K_FOLDS, SEED)
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
              f"({report['n_train_files_kept_nano']} arquivos com >=1 caixa nano), "
              f"val={report['n_val_sources']} ({report['n_val_files_kept_nano']} com nano), "
              f"held-out(test)={report['n_test_sources']} (nao filtrado)")
        LOGGER.info("fold=%d relatorio=%s", k, report)

    assignment_csv = EXP_DIR / "data" / "fold_assignment.csv"
    assignment_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(assignment_csv, "w") as f:
        f.write("source_image,class,fold,n_variants,n_nano,n_micro\n")
        for base, info in sorted(groups.items()):
            f.write(f"{base},{info['class']},{fold_of[base]},{len(info['variants'])},"
                    f"{info['n_nano']},{info['n_micro']}\n")

    save_experiment_summary(EXP_NAME, RUNTIME_METADATA, {
        "nano_max_nm": NANO_MAX_NM,
        "n_unique_source_images": n_unique,
        "class_counts": dict(class_counts),
        "n_nano_boxes_total": n_nano_total,
        "n_micro_boxes_total": n_micro_total,
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
