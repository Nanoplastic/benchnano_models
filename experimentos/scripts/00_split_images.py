"""Split treino/val/teste por IMAGEM-FONTE (não por partícula) para cada
classe do dataset MiNa (PE/PET/PP/PS). Lê direto dos JSONs COCO em
`Full_Images/COCO Format/`, não dos `Image_Patches` pré-cortados (que têm
cobertura inconsistente entre classes/formatos -- ver docs/resultados.md).

Split por imagem, não por recorte de partícula, porque partículas da mesma
micrografia compartilham ruído/textura de fundo -- misturar recortes da
mesma imagem entre treino e teste vazaria informação.
"""
import csv
import json
import random
from pathlib import Path

from vlm_common import ROOT, save_experiment_summary, setup_experiment_logging

EXP_NAME = "00_split_images"
EXP_DIR, LOG_PATH, LOGGER, RUNTIME_METADATA = setup_experiment_logging(EXP_NAME, __file__)

MINA_ROOT = ROOT / "MPDataset" / "Full_Images" / "COCO Format"
OUT = EXP_DIR / "data" / "image_splits.csv"
CLASSES = ["PE", "PET", "PP", "PS"]
SEED = 42
SPLIT_RATIOS = {"train": 0.7, "val": 0.15, "test": 0.15}


def split_indices(n: int, rng: random.Random):
    idx = list(range(n))
    rng.shuffle(idx)
    n_train = max(1, round(n * SPLIT_RATIOS["train"]))
    n_val = max(1, round(n * SPLIT_RATIOS["val"])) if n > 2 else 0
    n_train = min(n_train, n - (1 if n > 1 else 0))
    remaining = n - n_train
    n_val = min(n_val, max(remaining - 1, 0)) if remaining > 1 else remaining
    n_test = n - n_train - n_val
    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    out = [None] * n
    for pos, i in enumerate(idx):
        out[i] = splits[pos]
    return out


def main():
    LOGGER.info("iniciando split por imagem para cada classe MiNa")
    rng = random.Random(SEED)
    rows = []
    class_totals = {}
    try:
        for cls in CLASSES:
            json_path = MINA_ROOT / cls / f"{cls}_COCO.json"
            LOGGER.info("processando classe=%s json=%s", cls, json_path)
            d = json.load(open(json_path))
            ann_count = {}
            for a in d["annotations"]:
                ann_count[a["image_id"]] = ann_count.get(a["image_id"], 0) + 1

            present = []
            for im in d["images"]:
                rel = im["file_name"].split("/", 1)[-1]
                fpath = MINA_ROOT / cls / rel
                if fpath.exists():
                    present.append((im["id"], rel, ann_count.get(im["id"], 0)))

            present.sort(key=lambda t: t[1])
            splits = split_indices(len(present), rng)

            counts = {"train": [0, 0], "val": [0, 0], "test": [0, 0]}
            for (image_id, fname, n_ann), split in zip(present, splits):
                rows.append({
                    "class": cls, "file_name": fname, "image_id": image_id,
                    "n_annotations": n_ann, "split": split,
                })
                counts[split][0] += 1
                counts[split][1] += n_ann

            class_totals[cls] = counts
            LOGGER.info(
                "%s: total_imagens=%s train=%s/%s val=%s/%s test=%s/%s",
                cls,
                len(present),
                counts["train"][0], counts["train"][1],
                counts["val"][0], counts["val"][1],
                counts["test"][0], counts["test"][1],
            )
            print(f"{cls}: {len(present)} imagens presentes -> "
                  f"train={counts['train'][0]}img/{counts['train'][1]}inst  "
                  f"val={counts['val'][0]}img/{counts['val'][1]}inst  "
                  f"test={counts['test'][0]}img/{counts['test'][1]}inst")

        OUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUT, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["class", "file_name", "image_id", "n_annotations", "split"])
            w.writeheader()
            w.writerows(rows)

        summary = {
            "total_images": len(rows),
            "total_annotations": sum(r["n_annotations"] for r in rows),
            "class_totals": class_totals,
            "seed": SEED,
            "split_ratios": SPLIT_RATIOS,
            "output_path": str(OUT),
        }
        save_experiment_summary(EXP_NAME, RUNTIME_METADATA, summary)
        LOGGER.info("split final salvo em %s total_imagens=%s total_annotations=%s", OUT, len(rows), summary["total_annotations"])
        print(f"\nsplit salvo em {OUT} ({len(rows)} imagens no total)")
    except Exception:
        LOGGER.exception("erro ao executar split do dataset")
        raise


if __name__ == "__main__":
    main()
