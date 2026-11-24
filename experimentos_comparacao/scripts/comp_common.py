"""Utilitários da comparação com os métodos de referência do MiNa / Utilities for the MiNa reference method comparison.

PT: Utilitários compartilhados pela comparação entre o BenchNano (YOLO26L + SmolVLM) e os
    métodos de referência publicados com o dataset MiNa.
    Segue a convenção de `experimentos_controlados/scripts/ctrl_common.py` (summary.json por
    experimento com ambiente e proveniência), mas é independente: não importa
    `vlm_common.py` para logging porque `vlm_common.RAIZ` aponta para `experimentos/` e os
    registros seriam gravados fora desta pasta. Cada trilha grava só no próprio diretório.
    Não modifica nada em `experimentos/` nem em `experimentos_controlados/`; apenas lê a
    partição fixa e o dataset YOLO existentes.
EN: Utilities shared by the comparison between BenchNano (YOLO26L + SmolVLM) and the
    reference methods published with the MiNa dataset.
    Follows the convention of `experimentos_controlados/scripts/ctrl_common.py` (one
    summary.json per experiment with environment and provenance), but is standalone: it does
    not import `vlm_common.py` for logging because `vlm_common.RAIZ` points to
    `experimentos/` and logs would be written outside this folder. Each track writes only to
    its own directory.
    Does not modify anything in `experimentos/` or `experimentos_controlados/`; it only
    reads the existing fixed split and YOLO dataset.
"""
import csv
import json
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

RAIZ_COMP = Path(__file__).resolve().parent.parent  # PT: .../experimentos_comparacao / EN: .../experimentos_comparacao
RAIZ_PROJETO_ORIGEM = RAIZ_COMP.parent  # PT: raiz do repositório / EN: repository root
EXPERIMENTOS_DIR = RAIZ_PROJETO_ORIGEM / "experimentos"

# PT: -- fontes de dados existentes, reaproveitadas (nunca regeradas) --
# EN: -- existing data sources, reused (never regenerated) --
MANIFESTO_PARTICAO = EXPERIMENTOS_DIR / "resultados" / "00_split_images" / "data" / "image_splits.csv"
RAIZ_MINA_COCO = RAIZ_PROJETO_ORIGEM / "MPDataset" / "Full_Images" / "COCO Format"
DATASET_DET_YOLO = EXPERIMENTOS_DIR / "datasets_yolo26_v2"  # PT: 4 classes, partição fixa, a mesma usada pelo YOLO26L / EN: 4 classes, fixed split, the same used by YOLO26L
YAML_DADOS_DET = DATASET_DET_YOLO / "data.yaml"

# PT: variante "full" com o mesmo conjunto de imagens de datasets_baseline_seg/ (ver
#     16_build_det_full_matched.py), usada só com --matched-pool para eliminar a assimetria
#     162/15 vs 155/14 do protocolo de imagem inteira entre os métodos com e sem máscara
# EN: "full" variant with the same image pool as datasets_baseline_seg/ (see
#     16_build_det_full_matched.py), used only with --matched-pool to remove the 162/15 vs
#     155/14 asymmetry of the full-image protocol between methods with and without masks
YAML_DADOS_DET_PAREADO = RAIZ_COMP / "datasets_det_full_matched" / "data.yaml"

# PT: -- protocolo "recorte curado" (réplica corrigida do recorte 256x256 do MiNa, ver
#     08_build_curated_crop_dataset.py); usado só na comparação com os métodos de referência
#     do dataset, não pelo BenchNano principal
# EN: -- "curated crop" protocol (corrected replica of the MiNa 256x256 crop, see
#     08_build_curated_crop_dataset.py); used only in the comparison with the dataset's
#     reference methods, not by the main BenchNano pipeline
DIR_RECORTE_CURADO = RAIZ_COMP / "datasets_curated_crop"
TAMANHO_RECORTE = 256
MAX_INSTANCIAS_POR_IMAGEM = 500  # PT: limite para Faster/Mask R-CNN em micrografias densas (protocolo de imagem inteira) / EN: cap for Faster/Mask R-CNN on dense micrographs (full-image protocol)
CLASSES_POLIMERO = ["PE", "PET", "PP", "PS"]

# PT: -- saídas desta trilha --
# EN: -- outputs of this track --
DIR_DATASET_SEG = RAIZ_COMP / "datasets_baseline_seg"
RESULTADOS_DIR = RAIZ_COMP / "resultados"
DIR_EXECUCOES = RAIZ_COMP / "runs"
RELATORIO_FINAL_DIR = RAIZ_COMP / "relatorio_final"


# PT: Partição fixa (reaproveitada de experimentos/00_split_images)
# EN: Fixed split (reused from experimentos/00_split_images)

def carregar_manifesto_particao() -> list[dict]:
    """PT: Lê experimentos/resultados/00_split_images/data/image_splits.csv. Nunca regera a
        partição, só lê a já fixada (a mesma usada pelo YOLO26L principal). Cada linha:
        class, file_name, image_id, n_annotations, split.
    EN: Reads experimentos/resultados/00_split_images/data/image_splits.csv. Never
        regenerates the split, only reads the fixed one (the same used by the main YOLO26L).
        Each row: class, file_name, image_id, n_annotations, split.
    """
    if not MANIFESTO_PARTICAO.exists():
        raise FileNotFoundError(
            f"split travado não encontrado em {MANIFESTO_PARTICAO} -- rode "
            f"experimentos/scripts/00_split_images.py primeiro (não deveria "
            f"ser necessário; esse split já é usado pelo YOLO26L principal)."
        )
    with open(MANIFESTO_PARTICAO, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def yolo_det_imagem_para_particao() -> dict[str, str]:
    """PT: Mapeia cada arquivo em datasets_yolo26_v2/images/{split}/ (incluindo as duplicatas
        dupN_*, usadas para balancear a classe PE no treino) para a sua partição, sem
        reprocessar o CSV.
    EN: Maps every file in datasets_yolo26_v2/images/{split}/ (including the dupN_*
        duplicates used to balance the PE class in training) to its split, without
        reprocessing the CSV.
    """
    saida = {}
    for particao in ("train", "val", "test"):
        d = DATASET_DET_YOLO / "images" / particao
        if not d.exists():
            continue
        for p in d.glob("*.png"):
            saida[p.name] = particao
    return saida


def remover_prefixo_duplicado(nome_do_arquivo: str) -> str:
    """PT: 'dup3_PE_PE-10000X.png' -> 'PE_PE-10000X.png' (as duplicatas são cópias byte a byte
        do original, usadas só para sobrerrepresentar PE no treino). Nomes sem prefixo ficam
        inalterados.
    EN: 'dup3_PE_PE-10000X.png' -> 'PE_PE-10000X.png' (duplicates are byte-for-byte copies
        of the original, used only to oversample PE in training). Names without the prefix
        are left unchanged.
    """
    partes = nome_do_arquivo.split("_", 1)
    if len(partes) == 2 and partes[0].startswith("dup") and partes[0][3:].isdigit():
        return partes[1]
    return nome_do_arquivo


# PT: Ambiente / proveniência (mesmo padrão de ctrl_common.coletar_ambiente)
# EN: Environment / provenance (same pattern as ctrl_common.coletar_ambiente)

def _commit_git() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(RAIZ_PROJETO_ORIGEM), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return None


def coletar_ambiente() -> dict:
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "git_commit": _commit_git(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
    for nome_modulo in ("transformers", "ultralytics", "detectron2"):
        try:
            modulo = __import__(nome_modulo)
            info[f"{nome_modulo}_version"] = getattr(modulo, "__version__", "desconhecida")
        except Exception:
            info[f"{nome_modulo}_version"] = None
    return info


# PT: Resultado por experimento (execução única)
# EN: Per-experiment result (single run)

def dir_exp(experimento: str) -> Path:
    d = RESULTADOS_DIR / experimento
    d.mkdir(parents=True, exist_ok=True)
    return d


def salvar_resumo(experimento: str, metricas: dict, metadados_extras: dict | None = None) -> Path:
    d = dir_exp(experimento)
    conteudo = {
        "experiment": experimento,
        "environment": coletar_ambiente(),
        "metrics": metricas,
    }
    if metadados_extras:
        conteudo["metadata"] = metadados_extras
    caminho_saida = d / "summary.json"
    with open(caminho_saida, "w", encoding="utf-8") as f:
        json.dump(conteudo, f, indent=2, ensure_ascii=False)
    return caminho_saida


def carregar_resumo(experimento: str) -> dict | None:
    p = RESULTADOS_DIR / experimento / "summary.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def duracao_legivel(segundos: float) -> str:
    segundos = int(round(segundos))
    h, resto = divmod(segundos, 3600)
    m, s = divmod(resto, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# PT: Valores de referência publicados com o dataset MiNa. Fixos no código e nunca
#     comparados diretamente com os resultados deste repositório sem a ressalva de que a
#     partição de referência tem vazamento entre treino e teste (ver README.md desta pasta).
# EN: Reference values published with the MiNa dataset. Hard-coded and never compared
#     directly with this repository's results without the caveat that the reference split
#     leaks between train and test (see this folder's README.md).

REFERENCIA_PUBLICADA = {
    "citation": "Valores de referência publicados com o dataset MiNa",
    "split_caveat": "Split por patch 256x256 distribuído ALEATORIAMENTE entre "
                     "treino/val/teste (confirmado no README do repositório "
                     "do MiNa) -- patches da mesma "
                     "imagem-fonte podem cair em treino e teste. Não "
                     "comparável 1:1 com os números reproduzidos nesta "
                     "pasta (split travado por imagem-fonte inteira).",
    # PT: segmentação de instância sem classe, em recortes (valores de referência)
    # EN: class-agnostic instance segmentation, on crops (reference values)
    "segmentation": {
        "mask_rcnn_r101":   {"ap50": 75.878, "ap75": 44.723, "ap": 43.541},
        "mask_rcnn_x101":   {"ap50": 71.061, "ap75": 45.145, "ap": 42.365},
        "yolov8n_seg":      {"ap50": 81.58,  "ap75": 53.03,  "ap": 49.72},
        "yolov26n_seg":     {"ap50": 73.06,  "ap75": 44.46,  "ap": 42.59},
    },
    # PT: detecção + classificação de polímero (4 classes), em recortes (valores de
    #     referência)
    # EN: detection + polymer classification (4 classes), on crops (reference values)
    "detection_polymer": {
        "rtdetr_l":         {"map50": 74.00, "precision": 74.70, "recall": 70.10, "f1": 72.33},
        "faster_rcnn_r101": {"map50": 54.49, "precision": 58.00, "recall": 75.10, "f1": 65.45},
        "faster_rcnn_x101": {"map50": 57.12, "precision": 56.60, "recall": 75.70, "f1": 64.77},
        "yolov26s":         {"map50": 77.30, "precision": 74.40, "recall": 72.30, "f1": 73.33},
        "yolov8s":          {"map50": 74.20, "precision": 72.80, "recall": 69.30, "f1": 71.01},
    },
    # PT: SAM zero-shot / com prompt (precisão, recall, F1 e MAPE aproximados a partir do
    #     gráfico de referência)
    # EN: SAM zero-shot / prompted (precision, recall, F1 and MAPE approximated from the
    #     reference plot)
    "sam_zeroshot": {
        "micro_sam_ais":       {"precision": 59, "recall": 17, "f1": 25, "mape": 12.9},
        "micro_sam_amg":       {"precision": 49, "recall": 15, "f1": 22, "mape": 10.4},
        "sam_rtdetr_points":   {"precision": 88, "recall": 56, "f1": 65, "mape": 1.5},
        "sam_rtdetr_boxes":    {"precision": 90, "recall": 58, "f1": 68, "mape": 1.5},
    },
}
