"""Manifesto de regime de tamanho / Size regime manifest.

PT: Gera o rótulo de regime de tamanho (NANOPLASTIC/MICROPLASTIC, limiar de 1 µm) para os
    recortes de `resultados/01_extract_real_crops/data/manifest_real.csv`, nas três
    partições (treino/val/teste), usando a calibração px->nm µm/px = 100/ampliação.
    Não extrai recortes novos: usa os mesmos arquivos (condição "caixa de referência": bbox
    de referência do MiNa com margem de 10 px) e troca o rótulo de polímero pelo regime de
    tamanho. Alimenta os scripts 16/17/18 (zero-shot / LoRA / ajuste fino completo).
EN: Builds the size regime label (NANOPLASTIC/MICROPLASTIC, 1 µm threshold) for the crops in
    `resultados/01_extract_real_crops/data/manifest_real.csv`, across the three splits
    (train/val/test), using the px->nm calibration µm/px = 100/magnification.
    No new crops are extracted: the same files are reused ("reference box" condition: MiNa
    reference bbox with a 10 px margin) and the polymer label is replaced by the size
    regime. Feeds scripts 16/17/18 (zero-shot / LoRA / full fine-tuning).
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import MANIFESTO_REAL, RAIZ, RAIZ_RESULTADOS

RAIZ_COMPLETA = RAIZ / "MPDataset" / "Full_Images" / "COCO Format"
NANO_MAX_NM = 1000
CAMINHO_SAIDA = RAIZ_RESULTADOS / "15_build_size_regime_manifest" / "data" / "manifest_size_regime.csv"

AMPL_X = re.compile(r"(\d+)\s*[xX]")
AMPL_TRACO = re.compile(r"^\D*(\d+)-\d+")


def interpretar_ampliacao(imagem_origem: str):
    m = AMPL_X.findall(imagem_origem)
    if m:
        return int(m[-1])
    m = AMPL_TRACO.match(imagem_origem)
    if m:
        return int(m.group(1))
    return None


def id_anot_de(linha_tab) -> int:
    radical = Path(linha_tab["image_path"]).stem
    radical_origem = Path(linha_tab["source_image"]).stem
    assert radical.startswith(radical_origem + "_"), (radical, radical_origem)
    return int(radical[len(radical_origem) + 1:])


def carregar_bboxes():
    caixas_bbox = {}
    for cls in ["PE", "PET", "PP", "PS"]:
        d = json.load(open(RAIZ_COMPLETA / cls / f"{cls}_COCO.json"))
        for a in d["annotations"]:
            caixas_bbox[(cls, a["id"])] = a["bbox"]
    return caixas_bbox


def principal():
    df = pd.read_csv(MANIFESTO_REAL)
    df["ann_id"] = df.apply(id_anot_de, axis=1)
    df["magnification"] = df["source_image"].apply(interpretar_ampliacao)

    caixas_bbox = carregar_bboxes()
    df["bbox_w"], df["bbox_h"] = zip(*[caixas_bbox[(r.label, r.ann_id)][2:] for r in df.itertuples()])
    df["px_size"] = df[["bbox_w", "bbox_h"]].max(axis=1)

    nao_resolvidos = df["magnification"].isna().sum()
    df = df.dropna(subset=["magnification"]).copy()
    df["um_per_px"] = 100.0 / df["magnification"]
    df["size_nm"] = df["px_size"] * df["um_per_px"] * 1000.0

    # PT: o rótulo de polímero original é mantido em outra coluna para análises cruzadas
    #     (ex.: acurácia de tamanho por polímero)
    # EN: the original polymer label is kept in another column for cross analyses (e.g. size
    #     accuracy per polymer)
    df["polymer_label"] = df["label"]
    df["label"] = df["size_nm"].apply(lambda x: "NANOPLASTIC" if x < NANO_MAX_NM else "MICROPLASTIC")

    print(f"total: {len(df) + nao_resolvidos} | magnificação não resolvida (excluídas): {nao_resolvidos}")
    for particao in ["train", "val", "test"]:
        sub = df[df["split"] == particao]
        n_nano = (sub["label"] == "NANOPLASTIC").sum()
        n_micro = (sub["label"] == "MICROPLASTIC").sum()
        print(f"  {particao:6s}: n={len(sub):4d}  NANOPLASTIC={n_nano:4d} ({n_nano/len(sub)*100:.1f}%)  "
              f"MICROPLASTIC={n_micro:4d} ({n_micro/len(sub)*100:.1f}%)")

    CAMINHO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CAMINHO_SAIDA, index=False)
    print(f"\nmanifesto salvo em {CAMINHO_SAIDA}")


if __name__ == "__main__":
    principal()
