"""Gera o manifesto de regime de tamanho pela Eq. (1) do artigo -- diâmetro
equivalente da área do polígono MiNa -- em vez da Eq. (2) (maior lado da
caixa) usada por 15_build_size_regime_manifest.py.

    Eq. (1)   d_eq = 2 * sqrt(A / pi) * kappa      A = area do poligono COCO
    Eq. (2)   d_box = max(w, h) * kappa            kappa = (100 / M) um/px

Motivo: o artigo define o regime de referência pelo diâmetro equivalente
(equation~(1)), mas o pipeline implementado sempre usou a aproximação pela
caixa. Como a caixa circunscreve o polígono, ela superestima o tamanho e
classifica como MICROPLASTIC partículas que a Eq. (1) classifica como
NANOPLASTIC -- nunca o contrário.

NÃO altera nada: lê `manifest_size_regime.csv` (Eq. 2) e escreve um arquivo
novo, `manifest_size_regime_eq1.csv`, com os mesmos 1.448 recortes, a mesma
partição e as mesmas colunas, só com `label` recalculado e duas colunas
extras (`area_px2`, `size_nm_eq1`) para auditoria.

Uso:
    python3 scripts/57_build_size_regime_manifest_eq1.py
"""
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import ROOT, RESULTS_ROOT

FULL_ROOT = ROOT / "MPDataset" / "Full_Images" / "COCO Format"
SRC = RESULTS_ROOT / "15_build_size_regime_manifest" / "data" / "manifest_size_regime.csv"
OUT_DIR = RESULTS_ROOT / "57_build_size_regime_manifest_eq1" / "data"
OUT_PATH = OUT_DIR / "manifest_size_regime_eq1.csv"
NANO_MAX_NM = 1000


def load_areas():
    """(classe, ann_id) -> area do poligono em px^2."""
    areas = {}
    for cls in ["PE", "PET", "PP", "PS"]:
        d = json.load(open(FULL_ROOT / cls / f"{cls}_COCO.json"))
        for a in d["annotations"]:
            areas[(cls, a["id"])] = a["area"]
    return areas


def main():
    df = pd.read_csv(SRC)
    areas = load_areas()

    keys = list(zip(df["polymer_label"], df["ann_id"]))
    missing = [k for k in keys if k not in areas]
    if missing:
        raise SystemExit(f"{len(missing)} recortes sem anotacao COCO correspondente: {missing[:5]}")

    df["area_px2"] = [areas[k] for k in keys]
    df["size_nm_eq1"] = 2 * (df["area_px2"] / math.pi) ** 0.5 * df["um_per_px"] * 1000.0

    label_eq2 = df["label"].copy()
    df["label"] = df["size_nm_eq1"].apply(
        lambda x: "NANOPLASTIC" if x < NANO_MAX_NM else "MICROPLASTIC"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    flips = (label_eq2 != df["label"]).sum()
    one_way = ((label_eq2 == "MICROPLASTIC") | (label_eq2 == df["label"])).all()
    print(f"escrito: {OUT_PATH}")
    print(f"recortes: {len(df)} | mudam de regime: {flips} "
          f"| todos MICRO->NANO: {bool(one_way)}")
    print()
    print(f"{'split':8s} {'n':>5s} | {'nano Eq.2':>10s} {'nano Eq.1':>10s} {'delta':>7s}")
    for split in ["train", "val", "test"]:
        m = df["split"] == split
        n2 = (label_eq2[m] == "NANOPLASTIC").sum()
        n1 = (df.loc[m, "label"] == "NANOPLASTIC").sum()
        print(f"{split:8s} {int(m.sum()):5d} | {n2:10d} {n1:10d} {n1 - n2:+7d}")


if __name__ == "__main__":
    main()
