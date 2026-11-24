"""Divisão treino/val/teste por imagem / Train/val/test split by image.

PT: Divide cada classe do dataset MiNa (PE/PET/PP/PS) em treino/val/teste por imagem de
    origem, não por partícula. Lê os JSONs COCO de `Full_Images/COCO Format/` em vez dos
    `Image_Patches` pré-cortados, cuja cobertura é inconsistente entre classes e formatos.
    A divisão é por imagem porque partículas da mesma micrografia compartilham ruído e
    textura de fundo; misturar recortes da mesma imagem entre treino e teste vazaria
    informação.
EN: Splits each MiNa dataset class (PE/PET/PP/PS) into train/val/test by source image, not
    by particle. Reads the COCO JSONs from `Full_Images/COCO Format/` instead of the pre-cut
    `Image_Patches`, whose coverage is inconsistent across classes and formats.
    The split is per image because particles from the same micrograph share noise and
    background texture; mixing crops of the same image across train and test would leak
    information.
"""
import csv
import json
import random
from pathlib import Path

from vlm_common import RAIZ, salvar_resumo_experimento, configurar_log_experimento

NOME_EXP = "00_split_images"
DIR_EXP, CAMINHO_LOG, REGISTRADOR, METADADOS_EXECUCAO = configurar_log_experimento(NOME_EXP, __file__)

RAIZ_MINA = RAIZ / "MPDataset" / "Full_Images" / "COCO Format"
SAIDA = DIR_EXP / "data" / "image_splits.csv"
CLASSES_POLIMERO = ["PE", "PET", "PP", "PS"]
SEMENTE = 42
PROPORCOES_PARTICAO = {"train": 0.7, "val": 0.15, "test": 0.15}


def indices_particao(n: int, gerador: random.Random):
    indice = list(range(n))
    gerador.shuffle(indice)
    n_treino = max(1, round(n * PROPORCOES_PARTICAO["train"]))
    n_val = max(1, round(n * PROPORCOES_PARTICAO["val"])) if n > 2 else 0
    n_treino = min(n_treino, n - (1 if n > 1 else 0))
    restantes = n - n_treino
    n_val = min(n_val, max(restantes - 1, 0)) if restantes > 1 else restantes
    n_test = n - n_treino - n_val
    particoes = ["train"] * n_treino + ["val"] * n_val + ["test"] * n_test
    saida = [None] * n
    for posicao, i in enumerate(indice):
        saida[i] = particoes[posicao]
    return saida


def principal():
    REGISTRADOR.info("iniciando split por imagem para cada classe MiNa")
    gerador = random.Random(SEMENTE)
    linhas_tab = []
    totais_classe = {}
    try:
        for cls in CLASSES_POLIMERO:
            caminho_json = RAIZ_MINA / cls / f"{cls}_COCO.json"
            REGISTRADOR.info("processando classe=%s json=%s", cls, caminho_json)
            d = json.load(open(caminho_json))
            contagem_anot = {}
            for a in d["annotations"]:
                contagem_anot[a["image_id"]] = contagem_anot.get(a["image_id"], 0) + 1

            presentes = []
            for im in d["images"]:
                rel = im["file_name"].split("/", 1)[-1]
                caminho_arq = RAIZ_MINA / cls / rel
                if caminho_arq.exists():
                    presentes.append((im["id"], rel, contagem_anot.get(im["id"], 0)))

            presentes.sort(key=lambda t: t[1])
            particoes = indices_particao(len(presentes), gerador)

            contagens = {"train": [0, 0], "val": [0, 0], "test": [0, 0]}
            for (id_imagem, nome_arq, n_anot), particao in zip(presentes, particoes):
                linhas_tab.append({
                    "class": cls, "file_name": nome_arq, "image_id": id_imagem,
                    "n_annotations": n_anot, "split": particao,
                })
                contagens[particao][0] += 1
                contagens[particao][1] += n_anot

            totais_classe[cls] = contagens
            REGISTRADOR.info(
                "%s: total_imagens=%s train=%s/%s val=%s/%s test=%s/%s",
                cls,
                len(presentes),
                contagens["train"][0], contagens["train"][1],
                contagens["val"][0], contagens["val"][1],
                contagens["test"][0], contagens["test"][1],
            )
            print(f"{cls}: {len(presentes)} imagens presentes -> "
                  f"train={contagens['train'][0]}img/{contagens['train'][1]}inst  "
                  f"val={contagens['val'][0]}img/{contagens['val'][1]}inst  "
                  f"test={contagens['test'][0]}img/{contagens['test'][1]}inst")

        SAIDA.parent.mkdir(parents=True, exist_ok=True)
        with open(SAIDA, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["class", "file_name", "image_id", "n_annotations", "split"])
            w.writeheader()
            w.writerows(linhas_tab)

        resumo = {
            "total_images": len(linhas_tab),
            "total_annotations": sum(r["n_annotations"] for r in linhas_tab),
            "class_totals": totais_classe,
            "seed": SEMENTE,
            "split_ratios": PROPORCOES_PARTICAO,
            "output_path": str(SAIDA),
        }
        salvar_resumo_experimento(NOME_EXP, METADADOS_EXECUCAO, resumo)
        REGISTRADOR.info("split final salvo em %s total_imagens=%s total_annotations=%s", SAIDA, len(linhas_tab), resumo["total_annotations"])
        print(f"\nsplit salvo em {SAIDA} ({len(linhas_tab)} imagens no total)")
    except Exception:
        REGISTRADOR.exception("erro ao executar split do dataset")
        raise


if __name__ == "__main__":
    principal()
