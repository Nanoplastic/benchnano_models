"""Utilitários compartilhados pela comparação BenchNano (YOLO26L + SmolVLM)
vs. o método do paper que introduziu o dataset MiNa (Rezvani, Zarrabi et
al., Toronto Metropolitan University -- `/home/mi/Downloads/artigomina.pdf`,
arXiv:2409.13688). Ver plano completo em
/home/mi/.claude/plans/agile-herding-crescent.md.

Mesma convenção de `experimentos_controlados/scripts/ctrl_common.py`
(summary.json por experimento com ambiente/proveniência), mas standalone --
não importa `vlm_common.py` para logging porque `vlm_common.ROOT` resolve
para `experimentos/` e escreveria fora desta pasta, violando a regra do
projeto de que cada trilha escreve só no seu próprio diretório
(`experimentos/README.md`: "nenhum script deve escrever fora da sua
própria pasta numerada").

Não modifica nada em `experimentos/` nem `experimentos_controlados/` --
apenas LÊ o split travado e o dataset YOLO já existentes.
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

COMP_ROOT = Path(__file__).resolve().parent.parent  # .../experimentos_comparacao
DATASETMINA_ROOT = COMP_ROOT.parent  # .../datasetMINA
EXPERIMENTOS_DIR = DATASETMINA_ROOT / "experimentos"

# -- fontes de dados já existentes, reaproveitadas (nunca regeradas) --------
SPLIT_MANIFEST = EXPERIMENTOS_DIR / "resultados" / "00_split_images" / "data" / "image_splits.csv"
MINA_COCO_ROOT = DATASETMINA_ROOT / "MPDataset" / "Full_Images" / "COCO Format"
YOLO_DET_DATASET = EXPERIMENTOS_DIR / "datasets_yolo26_v2"  # 4 classes, split travado, já usado pelo YOLO26L
DET_DATA_YAML = YOLO_DET_DATASET / "data.yaml"

# variante "full" com o mesmo pool de imagens de datasets_baseline_seg/ (ver
# 16_build_det_full_matched.py) -- usada só com --matched-pool, pra tirar a
# assimetria 162/15 vs 155/14 do protocolo full-image entre os métodos com e
# sem máscara (main_pt.tex/main_v7.tex, Section "Comparison with the MiNa
# benchmark study")
DET_DATA_YAML_MATCHED = COMP_ROOT / "datasets_det_full_matched" / "data.yaml"

# -- protocolo "curated crop" (réplica corrigida do recorte 256x256 do MiNa,
# ver 08_build_curated_crop_dataset.py) -- só usado pela comparação com o
# paper do dataset (main_pt.tex, secao "Comparison with the MiNa benchmark
# study"), nao pelo BenchNano principal
CURATED_CROP_DIR = COMP_ROOT / "datasets_curated_crop"
CROP_SIZE = 256
MAX_INSTANCES_PER_IMAGE = 500  # cap p/ Faster/Mask R-CNN em micrografias densas (protocolo imagem-inteira)
CLASSES = ["PE", "PET", "PP", "PS"]

# -- saídas desta trilha ------------------------------------------------------
SEG_DATASET_DIR = COMP_ROOT / "datasets_baseline_seg"
RESULTADOS_DIR = COMP_ROOT / "resultados"
RUNS_DIR = COMP_ROOT / "runs"
RELATORIO_FINAL_DIR = COMP_ROOT / "relatorio_final"


# --------------------------------------------------------------------------
# Split travado (reaproveitado de experimentos/00_split_images.py)
# --------------------------------------------------------------------------

def load_split_manifest() -> list[dict]:
    """Lê experimentos/resultados/00_split_images/data/image_splits.csv --
    NUNCA regera o split, só lê o já travado (mesmo usado pelo YOLO26L
    principal). Cada linha: class, file_name, image_id, n_annotations, split."""
    if not SPLIT_MANIFEST.exists():
        raise FileNotFoundError(
            f"split travado não encontrado em {SPLIT_MANIFEST} -- rode "
            f"experimentos/scripts/00_split_images.py primeiro (não deveria "
            f"ser necessário; esse split já é usado pelo YOLO26L principal)."
        )
    with open(SPLIT_MANIFEST, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def yolo_det_image_to_split() -> dict[str, str]:
    """Mapeia todo arquivo em datasets_yolo26_v2/images/{split}/ (incluindo
    duplicatas dupN_*, usadas pra balancear a classe PE no treino) pro split
    a que pertence -- útil pra qualquer script que precise saber a origem de
    um arquivo já existente sem reprocessar o CSV manualmente."""
    out = {}
    for split in ("train", "val", "test"):
        d = YOLO_DET_DATASET / "images" / split
        if not d.exists():
            continue
        for p in d.glob("*.png"):
            out[p.name] = split
    return out


def strip_dup_prefix(filename: str) -> str:
    """'dup3_PE_PE-10000X.png' -> 'PE_PE-10000X.png' (duplicatas são cópias
    byte-a-byte do arquivo original, usadas só pra sobrerrepresentar PE no
    treino -- confirmado via md5sum). Sem-prefixo fica inalterado."""
    parts = filename.split("_", 1)
    if len(parts) == 2 and parts[0].startswith("dup") and parts[0][3:].isdigit():
        return parts[1]
    return filename


# --------------------------------------------------------------------------
# Ambiente / proveniência (mesmo padrão de ctrl_common.collect_environment)
# --------------------------------------------------------------------------

def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(DATASETMINA_ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return None


def collect_environment() -> dict:
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "git_commit": _git_commit(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
    for mod_name in ("transformers", "ultralytics", "detectron2"):
        try:
            mod = __import__(mod_name)
            info[f"{mod_name}_version"] = getattr(mod, "__version__", "desconhecida")
        except Exception:
            info[f"{mod_name}_version"] = None
    return info


# --------------------------------------------------------------------------
# Resultado por experimento (single-run -- ver nota no README sobre 10x)
# --------------------------------------------------------------------------

def exp_dir(experiment: str) -> Path:
    d = RESULTADOS_DIR / experiment
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_summary(experiment: str, metrics: dict, extra_metadata: dict | None = None) -> Path:
    d = exp_dir(experiment)
    payload = {
        "experiment": experiment,
        "environment": collect_environment(),
        "metrics": metrics,
    }
    if extra_metadata:
        payload["metadata"] = extra_metadata
    out_path = d / "summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path


def load_summary(experiment: str) -> dict | None:
    p = RESULTADOS_DIR / experiment / "summary.json"
    if not p.exists():
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def human_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# --------------------------------------------------------------------------
# Números PUBLICADOS pelo paper do dataset (Rezvani/Zarrabi et al.) --
# hardcoded com a citação exata, NUNCA comparados diretamente com os nossos
# sem o aviso de "split com vazamento" (ver README.md desta pasta e o plano).
# --------------------------------------------------------------------------

PUBLISHED_BASELINE = {
    "citation": "Rezvani, Zarrabi et al., \"Morphological Detection and "
                 "Classification of Microplastics and Nanoplastics Emerged "
                 "from Consumer Products by Deep Learning\", arXiv:2409.13688 "
                 "(artigomina.pdf)",
    "split_caveat": "Split por patch 256x256 distribuído ALEATORIAMENTE entre "
                     "treino/val/teste (confirmado no README de "
                     "github.com/naviiidz/MiNa-dataset) -- patches da mesma "
                     "imagem-fonte podem cair em treino e teste. Não "
                     "comparável 1:1 com os números reproduzidos nesta "
                     "pasta (split travado por imagem-fonte inteira).",
    # Tabela 3 do paper -- segmentação de instância classe-agnóstica, patches
    "segmentation": {
        "mask_rcnn_r101":   {"ap50": 75.878, "ap75": 44.723, "ap": 43.541},
        "mask_rcnn_x101":   {"ap50": 71.061, "ap75": 45.145, "ap": 42.365},
        "yolov8n_seg":      {"ap50": 81.58,  "ap75": 53.03,  "ap": 49.72},
        "yolov26n_seg":     {"ap50": 73.06,  "ap75": 44.46,  "ap": 42.59},
    },
    # Tabela 4 do paper -- detecção + classificação de polímero (4 classes), patches
    "detection_polymer": {
        "rtdetr_l":         {"map50": 74.00, "precision": 74.70, "recall": 70.10, "f1": 72.33},
        "faster_rcnn_r101": {"map50": 54.49, "precision": 58.00, "recall": 75.10, "f1": 65.45},
        "faster_rcnn_x101": {"map50": 57.12, "precision": 56.60, "recall": 75.70, "f1": 64.77},
        "yolov26s":         {"map50": 77.30, "precision": 74.40, "recall": 72.30, "f1": 73.33},
        "yolov8s":          {"map50": 74.20, "precision": 72.80, "recall": 69.30, "f1": 71.01},
    },
    # Figura 5e/f -- SAM zero-shot / promptado (Precisão, Recall, F1, MAPE aproximados do gráfico)
    "sam_zeroshot": {
        "micro_sam_ais":       {"precision": 59, "recall": 17, "f1": 25, "mape": 12.9},
        "micro_sam_amg":       {"precision": 49, "recall": 15, "f1": 22, "mape": 10.4},
        "sam_rtdetr_points":   {"precision": 88, "recall": 56, "f1": 65, "mape": 1.5},
        "sam_rtdetr_boxes":    {"precision": 90, "recall": 58, "f1": 68, "mape": 1.5},
    },
}
