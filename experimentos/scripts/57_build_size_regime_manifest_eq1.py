"""Manifesto de regime de tamanho pela regra eq1 / Size regime manifest with the eq1 rule.

PT: Gera o manifesto de regime de tamanho pelo diâmetro equivalente da área do polígono MiNa
    (regra "eq1"), em vez do maior lado da caixa (regra "eq2") usado por
    15_build_size_regime_manifest.py:
    eq1: d_eq = 2 * sqrt(A / pi) * kappa, A = área do polígono COCO
    eq2: d_box = max(w, h) * kappa, kappa = (100 / M) µm/px
    Como a caixa circunscreve o polígono, a regra eq2 superestima o tamanho e classifica
    como MICROPLASTIC partículas que a eq1 classifica como NANOPLASTIC, nunca o contrário.
    Não altera arquivos existentes: lê `manifest_size_regime.csv` (eq2) e grava
    `manifest_size_regime_eq1.csv`, com os mesmos 1.448 recortes, a mesma partição e as
    mesmas colunas, com `label` recalculado e duas colunas extras de conferência
    (`area_px2`, `size_nm_eq1`).
    Uso:
    python3 scripts/57_build_size_regime_manifest_eq1.py
EN: Builds the size regime manifest from the area-equivalent diameter of the MiNa polygon
    ("eq1" rule), instead of the longest box side ("eq2" rule) used by
    15_build_size_regime_manifest.py:
    eq1: d_eq = 2 * sqrt(A / pi) * kappa, A = COCO polygon area
    eq2: d_box = max(w, h) * kappa, kappa = (100 / M) µm/px
    Since the box circumscribes the polygon, the eq2 rule overestimates size and labels as
    MICROPLASTIC particles that eq1 labels as NANOPLASTIC, never the opposite.
    Does not modify existing files: reads `manifest_size_regime.csv` (eq2) and writes
    `manifest_size_regime_eq1.csv`, with the same 1,448 crops, the same split and the same
    columns, with `label` recomputed and two extra check columns (`area_px2`,
    `size_nm_eq1`).
    Usage:
    python3 scripts/57_build_size_regime_manifest_eq1.py
"""
import json
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import RAIZ, RAIZ_RESULTADOS

RAIZ_COMPLETA = RAIZ / "MPDataset" / "Full_Images" / "COCO Format"
ORIGEM_DADOS = RAIZ_RESULTADOS / "15_build_size_regime_manifest" / "data" / "manifest_size_regime.csv"
DIR_SAIDA = RAIZ_RESULTADOS / "57_build_size_regime_manifest_eq1" / "data"
CAMINHO_SAIDA = DIR_SAIDA / "manifest_size_regime_eq1.csv"
NANO_MAX_NM = 1000


def carregar_areas():
    """PT: (classe, ann_id) -> área do polígono em px^2.
    EN: (class, ann_id) -> polygon area in px^2.
    """
    areas = {}
    for cls in ["PE", "PET", "PP", "PS"]:
        d = json.load(open(RAIZ_COMPLETA / cls / f"{cls}_COCO.json"))
        for a in d["annotations"]:
            areas[(cls, a["id"])] = a["area"]
    return areas


def principal():
    df = pd.read_csv(ORIGEM_DADOS)
    areas = carregar_areas()

    chaves = list(zip(df["polymer_label"], df["ann_id"]))
    ausentes = [k for k in chaves if k not in areas]
    if ausentes:
        raise SystemExit(f"{len(ausentes)} recortes sem anotacao COCO correspondente: {ausentes[:5]}")

    df["area_px2"] = [areas[k] for k in chaves]
    df["size_nm_eq1"] = 2 * (df["area_px2"] / math.pi) ** 0.5 * df["um_per_px"] * 1000.0

    rotulo_eq2 = df["label"].copy()
    df["label"] = df["size_nm_eq1"].apply(
        lambda x: "NANOPLASTIC" if x < NANO_MAX_NM else "MICROPLASTIC"
    )

    DIR_SAIDA.mkdir(parents=True, exist_ok=True)
    df.to_csv(CAMINHO_SAIDA, index=False)

    inversoes = (rotulo_eq2 != df["label"]).sum()
    unidirecional = ((rotulo_eq2 == "MICROPLASTIC") | (rotulo_eq2 == df["label"])).all()
    print(f"escrito: {CAMINHO_SAIDA}")
    print(f"recortes: {len(df)} | mudam de regime: {inversoes} "
          f"| todos MICRO->NANO: {bool(unidirecional)}")
    print()
    print(f"{'split':8s} {'n':>5s} | {'nano Eq.2':>10s} {'nano Eq.1':>10s} {'delta':>7s}")
    for particao in ["train", "val", "test"]:
        m = df["split"] == particao
        n2 = (rotulo_eq2[m] == "NANOPLASTIC").sum()
        n1 = (df.loc[m, "label"] == "NANOPLASTIC").sum()
        print(f"{particao:8s} {int(m.sum()):5d} | {n2:10d} {n1:10d} {n1 - n2:+7d}")


if __name__ == "__main__":
    principal()
