"""Funções compartilhadas para o experimento de fine-tuning VLM no dataset
MiNa (PE/PET/PP/PS). Reaproveita a mesma receita de modelo/LoRA da linha
`tarefa_vlm_nanoplastic_fewshot` (que por sua vez reaproveitou a receita do
pipeline NIST, o único fine-tune que já funcionou no projeto: 19%->68%)."""
import json
import logging
import os
import platform
import re
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
RESULTS_ROOT = ROOT / "resultados"
REAL_MANIFEST = ROOT / "resultados" / "01_extract_real_crops" / "data" / "manifest_real.csv"
SYNTHETIC_MANIFEST = ROOT / "resultados" / "03_compose_synthetic" / "data" / "manifest_synthetic.csv"
CHECKPOINTS_DIR = ROOT / "resultados" / "checkpoints"
REPORTS_DIR = ROOT / "resultados" / "reports"

MODEL_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def ensure_results_root(results_root: Path = RESULTS_ROOT) -> Path:
    results_root.mkdir(parents=True, exist_ok=True)
    return results_root


def build_experiment_dir(experiment_name: str, results_root: Path = RESULTS_ROOT) -> Path:
    ensure_results_root(results_root)
    experiment_dir = results_root / experiment_name
    for child in ["logs", "reports", "checkpoints", "data", "metadata"]:
        (experiment_dir / child).mkdir(parents=True, exist_ok=True)
    return experiment_dir


def _read_meminfo_total_gb() -> float | None:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except OSError:
        return None
    return None


def collect_runtime_metadata(extra: dict | None = None) -> dict:
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "os": platform.system(),
        "os_release": platform.release(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "architecture": platform.machine(),
        "cpu": platform.processor() or platform.uname().processor,
        "cpu_count": os.cpu_count(),
        "ram_gb": _read_meminfo_total_gb(),
        "working_directory": str(ROOT),
        "results_root": str(RESULTS_ROOT),
        "device": DEVICE,
    }

    try:
        git_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        info["git_commit"] = git_commit
    except Exception:
        info["git_commit"] = None

    try:
        import numpy as np
        info["numpy_version"] = np.__version__
    except Exception:
        info["numpy_version"] = None

    try:
        import pandas as pd
        info["pandas_version"] = pd.__version__
    except Exception:
        info["pandas_version"] = None

    try:
        from PIL import __version__ as pil_version
        info["pillow_version"] = pil_version
    except Exception:
        info["pillow_version"] = None

    try:
        import transformers
        info["transformers_version"] = transformers.__version__
    except Exception:
        info["transformers_version"] = None

    try:
        import peft
        info["peft_version"] = peft.__version__
    except Exception:
        info["peft_version"] = None

    info["torch_version"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        info["gpu_capability"] = [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]
        info["cuda_device"] = torch.cuda.current_device()
    else:
        info["cuda_version"] = None
        info["cudnn_version"] = None
        info["gpu_count"] = 0
        info["gpu_names"] = []
        info["gpu_capability"] = []
        info["cuda_device"] = None

    if extra:
        info.update(extra)
    return info


def setup_experiment_logging(experiment_name: str, script_name: str | None = None, extra_metadata: dict | None = None):
    experiment_dir = build_experiment_dir(experiment_name)
    log_dir = experiment_dir / "logs"
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = log_dir / f"{experiment_name}_{timestamp}.log"

    logger = logging.getLogger(f"datasetMINA.{experiment_name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    metadata = collect_runtime_metadata({
        "experiment_name": experiment_name,
        "script_name": script_name,
        "results_dir": str(experiment_dir),
        "log_path": str(log_path),
    })
    if extra_metadata:
        metadata.update(extra_metadata)

    metadata_path = experiment_dir / "metadata" / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)

    logger.info("=== INÍCIO DO EXPERIMENTO ===")
    logger.info("experiment_name=%s", experiment_name)
    logger.info("script_name=%s", script_name)
    logger.info("results_dir=%s", experiment_dir)
    logger.info("log_path=%s", log_path)
    logger.info("runtime_metadata=%s", json.dumps(metadata, ensure_ascii=False, sort_keys=True))
    return experiment_dir, log_path, logger, metadata


def save_experiment_summary(experiment_name: str, metadata: dict, metrics: dict | None = None, summary_path: Path | None = None):
    experiment_dir = build_experiment_dir(experiment_name)
    if summary_path is None:
        summary_path = experiment_dir / "summary.json"
    payload = {"experiment_name": experiment_name, "metadata": metadata}
    if metrics is not None:
        payload["metrics"] = metrics
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    return summary_path


def write_run_report(experiment_name: str, report_name: str, data: dict):
    experiment_dir = build_experiment_dir(experiment_name)
    report_dir = experiment_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    out_path = report_dir / report_name
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
    return out_path

# "PET" contém "PE" como substring -- checar PET (e as outras) ANTES de PE
# em extract_label, senão toda resposta "PET" seria lida como "PE".
LABELS = ["PET", "PP", "PS", "PE"]

CLASSIFY_PROMPT = (
    "This is a scanning electron microscope (SEM) image from a controlled "
    "reference sample containing a single known plastic polymer, showing "
    "particle(s) deposited on a filter membrane. Which polymer is this "
    "sample made of: PE (polyethylene), PET (polyethylene terephthalate), "
    "PP (polypropylene), or PS (polystyrene)? Answer with just one label: "
    "PE, PET, PP, or PS."
)

_LABEL_RE = {label: re.compile(rf"\b{label}\b", re.IGNORECASE) for label in LABELS}

# Tarefa alternativa: regime de tamanho (NANOPLASTIC/MICROPLASTIC), usada pela
# Camada 2 do artigo do colega (SmolVLM decide o regime de tamanho a partir da
# bbox, não o polímero -- ver scripts/15_build_size_regime_manifest.py).
# "NANOPLASTIC" tem que ser checado antes de qualquer coisa que contenha
# "PLASTIC" sozinho não ambíguo aqui, mas mantemos a mesma convenção de lista
# ordenada por segurança caso o prompt mude no futuro.
SIZE_LABELS = ["NANOPLASTIC", "MICROPLASTIC"]

SIZE_CLASSIFY_PROMPT = (
    "You are analyzing a scanning electron microscopy image containing "
    "plastic particles. The highlighted particle was detected by an "
    "object-detection model together with the image's physical scale "
    "information. Return NANOPLASTIC when the particle's characteristic "
    "size is below 1 micrometer. Return MICROPLASTIC otherwise. Answer "
    "with just one label: NANOPLASTIC or MICROPLASTIC."
)

import functools


@functools.lru_cache(maxsize=8)
def _label_regexes(labels: tuple) -> dict:
    return {label: re.compile(rf"\b{label}\b", re.IGNORECASE) for label in labels}


def extract_label(text: str, labels: list | None = None) -> str:
    labels = labels or LABELS
    regexes = _label_regexes(tuple(labels))
    for label in labels:  # ordem importa, ver comentário acima
        if regexes[label].search(text):
            return label
    return "Unresolved"


def load_real_split(split_name: str) -> pd.DataFrame:
    df = pd.read_csv(REAL_MANIFEST)
    return df[df["split"] == split_name].reset_index(drop=True)


def load_synthetic() -> pd.DataFrame:
    return pd.read_csv(SYNTHETIC_MANIFEST)


def load_base_model(model_id: str = MODEL_ID):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processor = AutoProcessor.from_pretrained(model_id)
    processor.image_processor.do_image_splitting = False
    model = AutoModelForVision2Seq.from_pretrained(
        model_id, torch_dtype=torch.bfloat16
    ).to(DEVICE)
    return processor, model


def build_supervised_example(processor, image_source, question: str, answer: str):
    image = image_source.convert("RGB") if isinstance(image_source, Image.Image) else Image.open(image_source).convert("RGB")
    user_msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question}]}]
    prompt_only = processor.apply_chat_template(user_msg, add_generation_prompt=True)
    full_msgs = user_msg + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}]
    full_text = processor.apply_chat_template(full_msgs, add_generation_prompt=False)

    prompt_inputs = processor(text=prompt_only, images=[image], return_tensors="pt")
    full_inputs = processor(text=full_text, images=[image], return_tensors="pt")

    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels = full_inputs["input_ids"].clone()
    labels[:, :prompt_len] = -100
    full_inputs["labels"] = labels
    return {k: v.to(DEVICE) for k, v in full_inputs.items()}


@torch.no_grad()
def classify_image(processor, model, image_path, prompt: str | None = None) -> str:
    """image_path aceita um Path (abre do disco) ou uma PIL.Image ja em
    memoria (util pra recortes gerados em tempo real, ex: crops a partir de
    caixas previstas pelo YOLO, sem precisar salvar arquivo temporario)."""
    model.eval()
    prompt_text = prompt or CLASSIFY_PROMPT
    image = image_path.convert("RGB") if isinstance(image_path, Image.Image) else Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt_text}]}]
    chat_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=chat_prompt, images=[image], return_tensors="pt").to(DEVICE)
    output_ids = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0]


def run_classification_eval(processor, model, df: pd.DataFrame, tag: str, out_dir: Path = REPORTS_DIR,
                             prompt: str | None = None, labels: list | None = None):
    rows = []
    for _, r in df.iterrows():
        img_path = ROOT / r["image_path"]
        raw = classify_image(processor, model, img_path, prompt=prompt)
        pred = extract_label(raw, labels=labels)
        rows.append({
            "image_path": r["image_path"],
            "true_label": r["label"],
            "raw_answer": raw.strip(),
            "pred_label": pred,
            "correct": pred == r["label"],
        })
    out_df = pd.DataFrame(rows)
    acc = out_df["correct"].mean() if len(out_df) else float("nan")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_dir / f"{tag}_predictions.csv", index=False)
    return out_df, acc


def finetune_lora(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                   tag: str, epochs: int = 6, grad_accum: int = 8, lr: float = 1e-4, seed: int = 42,
                   model_id: str = MODEL_ID, prompt: str | None = None, labels: list | None = None):
    """Loop de treino LoRA compartilhado pelos Experimentos A (05, só real) e
    B (06, real+sintético) -- mesma receita/hiperparâmetros dos dois, única
    diferença é o conteúdo de train_df, então factorar aqui evita duplicar o
    loop inteiro. Seleção de checkpoint SEMPRE por val_df real (nunca
    sintético) -- correção deliberada sobre a falha já documentada na linha
    anterior (`tarefa_vlm_nanoplastic_fewshot`), onde seleção por val
    sintético escolheu checkpoints que não generalizavam para dado real."""
    import random as _random

    from peft import LoraConfig, get_peft_model
    from torch.optim import AdamW

    # fixa também a RNG global do PyTorch (e do Python) antes de carregar o
    # modelo -- get_peft_model(..., init_lora_weights="gaussian") abaixo
    # sorteia os pesos iniciais do adapter usando torch.rand*, que até aqui
    # não tinha seed fixada (só a ordem de embaralhamento do treino tinha,
    # via rng/_random.Random(seed) mais abaixo)
    _random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    prompt_text = prompt or CLASSIFY_PROMPT

    processor, model = load_base_model(model_id)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_cfg = LoraConfig(
        r=8, lora_alpha=8, lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        init_lora_weights="gaussian",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    adapter_dir = CHECKPOINTS_DIR / tag
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    rng = _random.Random(seed)
    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        order = list(train_df.index)
        rng.shuffle(order)
        running_loss, n_loss = 0.0, 0
        optimizer.zero_grad()
        for i, idx in enumerate(order):
            r = train_df.loc[idx]
            inputs = build_supervised_example(processor, ROOT / r["image_path"], prompt_text, r["label"])
            out = model(**inputs)
            loss = out.loss / grad_accum
            loss.backward()
            running_loss += out.loss.item()
            n_loss += 1
            if (i + 1) % grad_accum == 0 or (i + 1) == len(order):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step()
                optimizer.zero_grad()

        avg_loss = running_loss / max(n_loss, 1)

        model.config.use_cache = True
        val_results, val_acc = run_classification_eval(processor, model, val_df, f"{tag}_epoch{epoch}_val",
                                                         prompt=prompt, labels=labels)
        model.config.use_cache = False
        print(f"epoch {epoch}/{epochs}  train_loss={avg_loss:.4f}  real_val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            adapter_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(adapter_dir)
            print(f"  -> novo melhor adapter (real_val_acc={val_acc:.3f}) salvo em {adapter_dir}")

    print(f"\nMelhor acurácia em real_val: {best_val_acc:.3f}. Adapter em {adapter_dir}")

    # recarrega o melhor adapter salvo (não necessariamente o da última época)
    from peft import PeftModel
    del model
    torch.cuda.empty_cache() if DEVICE == "cuda" else None
    base_processor, base_model = load_base_model(model_id)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.config.use_cache = True

    test_results, test_acc = run_classification_eval(base_processor, model, test_df, f"{tag}_real_test",
                                                       prompt=prompt, labels=labels)
    print(f"\nAcurácia final REAL TEST ({tag}, n={len(test_df)}): {test_acc:.3f}")
    return {"best_val_acc": best_val_acc, "test_acc": test_acc, "test_results": test_results}


def finetune_full(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                   tag: str, epochs: int = 6, grad_accum: int = 8, lr: float = 1e-5, seed: int = 42,
                   model_id: str = MODEL_ID, prompt: str | None = None, labels: list | None = None):
    """Full fine-tuning (todos os parâmetros, sem LoRA) -- mesma estrutura de
    loop do `finetune_lora`, mas ajustando o modelo inteiro em vez de um
    adapter de baixo posto. Motivação: o blog oficial do SmolVLM2 recomenda
    full fine-tuning em vez de LoRA/QLoRA especificamente para a variante
    500M ("since the 500M variant is small, it's better to apply full
    fine-tuning instead of QLoRA or LoRA" -- ver raw/smolvlm_finetuning_
    guidance.md), recomendação nunca testada nos experimentos anteriores
    deste projeto (todos usaram LoRA). lr default bem menor que o do LoRA
    (1e-5 vs 1e-4) porque atualizar todos os pesos com a mesma taxa do LoRA
    tende a divergir rápido; ajustar se o treino ficar instável."""
    import random as _random

    from torch.optim import AdamW

    # mesma correção de finetune_lora: fixa a RNG global do PyTorch (e do
    # Python) antes de carregar o modelo, não só a ordem de embaralhamento
    # do treino (rng/_random.Random(seed) abaixo)
    _random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    prompt_text = prompt or CLASSIFY_PROMPT

    processor, model = load_base_model(model_id)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.train()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"full fine-tuning: {n_params/1e6:.1f}M parametros treinaveis")

    ckpt_dir = CHECKPOINTS_DIR / tag
    optimizer = AdamW(model.parameters(), lr=lr)
    rng = _random.Random(seed)
    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        model.train()
        order = list(train_df.index)
        rng.shuffle(order)
        running_loss, n_loss = 0.0, 0
        optimizer.zero_grad()
        for i, idx in enumerate(order):
            r = train_df.loc[idx]
            inputs = build_supervised_example(processor, ROOT / r["image_path"], prompt_text, r["label"])
            out = model(**inputs)
            loss = out.loss / grad_accum
            loss.backward()
            running_loss += out.loss.item()
            n_loss += 1
            if (i + 1) % grad_accum == 0 or (i + 1) == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        avg_loss = running_loss / max(n_loss, 1)

        model.config.use_cache = True
        val_results, val_acc = run_classification_eval(processor, model, val_df, f"{tag}_epoch{epoch}_val",
                                                         prompt=prompt, labels=labels)
        model.config.use_cache = False
        print(f"epoch {epoch}/{epochs}  train_loss={avg_loss:.4f}  real_val_acc={val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            processor.save_pretrained(ckpt_dir)
            print(f"  -> novo melhor checkpoint (real_val_acc={val_acc:.3f}) salvo em {ckpt_dir}")

    print(f"\nMelhor acurácia em real_val: {best_val_acc:.3f}. Checkpoint em {ckpt_dir}")

    del model
    torch.cuda.empty_cache() if DEVICE == "cuda" else None
    from transformers import AutoModelForVision2Seq, AutoProcessor
    test_processor = AutoProcessor.from_pretrained(ckpt_dir)
    test_processor.image_processor.do_image_splitting = False
    test_model = AutoModelForVision2Seq.from_pretrained(ckpt_dir, torch_dtype=torch.bfloat16).to(DEVICE)
    test_model.config.use_cache = True

    test_results, test_acc = run_classification_eval(test_processor, test_model, test_df, f"{tag}_real_test",
                                                       prompt=prompt, labels=labels)
    print(f"\nAcurácia final REAL TEST ({tag}, n={len(test_df)}): {test_acc:.3f}")
    return {"best_val_acc": best_val_acc, "test_acc": test_acc, "test_results": test_results}
