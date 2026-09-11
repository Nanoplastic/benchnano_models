"""Gera um dataset YOLO nano-only (partículas <1000nm) para retreinar um
detector especializado em nanoplástico, seguindo a decisão de refazer os
experimentos novos (#42 análise de erro de classe, CV #43-45) restritos a
nanoplástico em vez do dataset misto.

Reaproveita a MESMA calibração px->nm (100/ampliação) e limiar NANO_MAX_NM=
1000nm já validados em 23_recall_by_size_regime.py / 26_pixel_calibration_
validation.py. Para cada caixa de anotação em treino+val, mantém só as que
ficam abaixo do limiar (size_nm<1000); caixas de partículas maiores viram
"background" (removidas do label, não da imagem) -- o objetivo é um detector
que aprenda a NÃO marcar partículas micro, não um detector geral reavaliado
depois. Imagens que ficam sem nenhuma caixa nano são excluídas do treino/val.

Escopo: só treino e validação. A partição de teste travada (datasets_yolo26_v2/
images(labels)/test) é copiada VERBATIM, sem filtrar -- o filtro nano nessa
partição é aplicado só na hora de avaliar (52_class_error_analysis_nano.py),
igual ao padrão já usado em 23_recall_by_size_regime.py, pra nunca alterar
fisicamente a partição travada.

Não sobrescreve nada existente: escreve só em datasets_yolo26_v2_nano/
(dataset novo) e resultados/47_build_nano_only_yolo_dataset/ (metadados).
"""
import re
import shutil
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import save_experiment_summary, setup_experiment_logging

EXP_NAME = "47_build_nano_only_yolo_dataset"
EXP_DIR, LOG_PATH, LOGGER, RUNTIME_METADATA = setup_experiment_logging(EXP_NAME, __file__)

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "datasets_yolo26_v2"
OUT_DIR = ROOT / "datasets_yolo26_v2_nano"
CLASS_NAMES = ["PE", "PET", "PP", "PS"]
NANO_MAX_NM = 1000.0

DUP_RE = re.compile(r"^dup\d+_")
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


def filter_labels_to_nano(label_path: Path, W: int, H: int, um_per_px: float):
    """Retorna (n_caixas_total, linhas_yolo_mantidas) -- so mantem caixas
    com size_nm < NANO_MAX_NM, no mesmo formato YOLO de entrada."""
    if not label_path.exists():
        return 0, []
    total = 0
    kept = []
    for line in open(label_path):
        parts = line.split()
        if len(parts) < 5:
            continue
        total += 1
        cls_id = int(parts[0])
        cx, cy, w, h = map(float, parts[1:5])
        px_size = max(w * W, h * H)
        size_nm = px_size * um_per_px * 1000.0
        if size_nm < NANO_MAX_NM:
            kept.append(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
    return total, kept


def process_split(split: str):
    img_dir = SRC_DIR / "images" / split
    label_dir = SRC_DIR / "labels" / split
    out_img_dir = OUT_DIR / "images" / split
    out_label_dir = OUT_DIR / "labels" / split
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)

    img_paths = sorted(img_dir.glob("*.png"))

    n_images_total = 0
    n_images_kept = 0
    n_boxes_total = 0
    n_boxes_nano = 0
    n_unresolved_mag = 0

    for img_path in img_paths:
        n_images_total += 1
        base_stem = DUP_RE.sub("", img_path.stem)
        mag = parse_magnification(base_stem)
        if mag is None:
            n_unresolved_mag += 1
            LOGGER.warning("magnificacao nao resolvida, imagem descartada: %s", img_path.name)
            continue
        um_per_px = 100.0 / mag

        label_path = label_dir / f"{img_path.stem}.txt"
        with Image.open(img_path) as im:
            W, H = im.size
        n_total, kept_lines = filter_labels_to_nano(label_path, W, H, um_per_px)
        n_boxes_total += n_total
        n_boxes_nano += len(kept_lines)

        if not kept_lines:
            continue

        n_images_kept += 1
        shutil.copy2(img_path, out_img_dir / img_path.name)
        with open(out_label_dir / f"{img_path.stem}.txt", "w") as f:
            f.writelines(kept_lines)

    stats = {
        "n_images_total": n_images_total,
        "n_images_kept": n_images_kept,
        "n_boxes_total": n_boxes_total,
        "n_boxes_nano": n_boxes_nano,
        "n_unresolved_magnification": n_unresolved_mag,
    }
    LOGGER.info("split=%s stats=%s", split, stats)
    print(f"{split}: {n_images_total} imagens -> {n_images_kept} com >=1 caixa nano "
          f"({n_boxes_nano}/{n_boxes_total} caixas nano)")
    return stats


def copy_test_verbatim():
    img_dir = SRC_DIR / "images" / "test"
    label_dir = SRC_DIR / "labels" / "test"
    out_img_dir = OUT_DIR / "images" / "test"
    out_label_dir = OUT_DIR / "labels" / "test"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_label_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for img_path in sorted(img_dir.glob("*.png")):
        shutil.copy2(img_path, out_img_dir / img_path.name)
        label_path = label_dir / f"{img_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, out_label_dir / label_path.name)
        n += 1
    print(f"test: {n} imagens copiadas verbatim (SEM filtro nano -- filtro aplicado só na avaliação)")
    return {"n_images": n}


def main():
    LOGGER.info("=== construcao do dataset nano-only (NANO_MAX_NM=%.1f) ===", NANO_MAX_NM)

    if OUT_DIR.exists():
        LOGGER.warning("removendo %s existente antes de reconstruir", OUT_DIR)
        shutil.rmtree(OUT_DIR)

    stats = {}
    for split in ("train", "val"):
        stats[split] = process_split(split)
    stats["test"] = copy_test_verbatim()

    data_yaml_path = OUT_DIR / "data.yaml"
    data_yaml_path.write_text(
        f"path: {OUT_DIR}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n" + "\n".join(f"  - {c}" for c in CLASS_NAMES) + "\n"
    )
    print(f"\ndata.yaml escrito em {data_yaml_path}")

    save_experiment_summary(EXP_NAME, RUNTIME_METADATA, {
        "nano_max_nm": NANO_MAX_NM,
        "output_dir": str(OUT_DIR),
        "data_yaml": str(data_yaml_path),
        "stats": stats,
    })
    print(f"Resumo salvo em {EXP_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
