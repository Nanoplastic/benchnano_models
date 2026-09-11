"""Gera o manifesto de rótulo de regime de tamanho (NANOPLASTIC/MICROPLASTIC,
limiar de 1µm) para os mesmos recortes de partícula já extraídos em
`resultados/01_extract_real_crops/data/manifest_real.csv` -- reaproveita a
mesma calibração px->nm (µm/px = 100/magnificação) já usada e validada em
07_size_band_eval.py e 12_nano_subset_eval.py, só que agora aplicada às TRÊS
partições (treino/val/teste), não só ao teste.

Não extrai nenhum recorte novo -- os mesmos arquivos de imagem
(condição "reference-box": bbox de referência do MiNa, com margem de 10px)
continuam sendo usados; só troca o rótulo de polímero por regime de tamanho,
pra alimentar os scripts 16/17/18 (zero-shot / LoRA / full fine-tuning na
tarefa da Camada 2 do artigo).
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import REAL_MANIFEST, ROOT, RESULTS_ROOT

FULL_ROOT = ROOT / "MPDataset" / "Full_Images" / "COCO Format"
NANO_MAX_NM = 1000
OUT_PATH = RESULTS_ROOT / "15_build_size_regime_manifest" / "data" / "manifest_size_regime.csv"

MAG_X = re.compile(r"(\d+)\s*[xX]")
MAG_DASH = re.compile(r"^\D*(\d+)-\d+")


def parse_magnification(source_image: str):
    m = MAG_X.findall(source_image)
    if m:
        return int(m[-1])
    m = MAG_DASH.match(source_image)
    if m:
        return int(m.group(1))
    return None


def ann_id_of(row) -> int:
    stem = Path(row["image_path"]).stem
    src_stem = Path(row["source_image"]).stem
    assert stem.startswith(src_stem + "_"), (stem, src_stem)
    return int(stem[len(src_stem) + 1:])


def load_bboxes():
    bboxes = {}
    for cls in ["PE", "PET", "PP", "PS"]:
        d = json.load(open(FULL_ROOT / cls / f"{cls}_COCO.json"))
        for a in d["annotations"]:
            bboxes[(cls, a["id"])] = a["bbox"]
    return bboxes


def main():
    df = pd.read_csv(REAL_MANIFEST)
    df["ann_id"] = df.apply(ann_id_of, axis=1)
    df["magnification"] = df["source_image"].apply(parse_magnification)

    bboxes = load_bboxes()
    df["bbox_w"], df["bbox_h"] = zip(*[bboxes[(r.label, r.ann_id)][2:] for r in df.itertuples()])
    df["px_size"] = df[["bbox_w", "bbox_h"]].max(axis=1)

    unresolved = df["magnification"].isna().sum()
    df = df.dropna(subset=["magnification"]).copy()
    df["um_per_px"] = 100.0 / df["magnification"]
    df["size_nm"] = df["px_size"] * df["um_per_px"] * 1000.0

    # rótulo original de polímero preservado em outra coluna -- pode ser
    # útil pra análise cruzada (ex: acurácia de tamanho por polímero)
    df["polymer_label"] = df["label"]
    df["label"] = df["size_nm"].apply(lambda x: "NANOPLASTIC" if x < NANO_MAX_NM else "MICROPLASTIC")

    print(f"total: {len(df) + unresolved} | magnificação não resolvida (excluídas): {unresolved}")
    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split]
        n_nano = (sub["label"] == "NANOPLASTIC").sum()
        n_micro = (sub["label"] == "MICROPLASTIC").sum()
        print(f"  {split:6s}: n={len(sub):4d}  NANOPLASTIC={n_nano:4d} ({n_nano/len(sub)*100:.1f}%)  "
              f"MICROPLASTIC={n_micro:4d} ({n_micro/len(sub)*100:.1f}%)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nmanifesto salvo em {OUT_PATH}")


if __name__ == "__main__":
    main()
