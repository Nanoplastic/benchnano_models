"""Funções comuns dos experimentos com VLM / Shared VLM experiment utilities.

PT: Funções compartilhadas pelos experimentos de ajuste fino de VLM no dataset MiNa
    (PE/PET/PP/PS): carga de modelo, prompts, extração de rótulo, avaliação, treino LoRA e
    completo, registro de logs e metadados de execução.
EN: Functions shared by the VLM fine-tuning experiments on the MiNa dataset (PE/PET/PP/PS):
    model loading, prompts, label extraction, evaluation, LoRA and full training, logging
    and run metadata.
"""
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

DIR_SCRIPT = Path(__file__).resolve().parent
RAIZ = DIR_SCRIPT.parent
RAIZ_RESULTADOS = RAIZ / "resultados"
MANIFESTO_REAL = RAIZ / "resultados" / "01_extract_real_crops" / "data" / "manifest_real.csv"
MANIFESTO_SINTETICO = RAIZ / "resultados" / "03_compose_synthetic" / "data" / "manifest_synthetic.csv"
DIR_CHECKPOINTS = RAIZ / "resultados" / "checkpoints"
DIR_RELATORIOS = RAIZ / "resultados" / "reports"

ID_MODELO = "HuggingFaceTB/SmolVLM-500M-Instruct"
DISPOSITIVO = "cuda" if torch.cuda.is_available() else "cpu"


def garantir_raiz_resultados(raiz_resultados: Path = RAIZ_RESULTADOS) -> Path:
    raiz_resultados.mkdir(parents=True, exist_ok=True)
    return raiz_resultados


def construir_dir_experimento(nome_experimento: str, raiz_resultados: Path = RAIZ_RESULTADOS) -> Path:
    garantir_raiz_resultados(raiz_resultados)
    dir_experimento = raiz_resultados / nome_experimento
    for filho in ["logs", "reports", "checkpoints", "data", "metadata"]:
        (dir_experimento / filho).mkdir(parents=True, exist_ok=True)
    return dir_experimento


def _ler_memoria_total_gb() -> float | None:
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as marcador:
            for linha in marcador:
                if linha.startswith("MemTotal:"):
                    kb = int(linha.split()[1])
                    return kb / (1024 * 1024)
    except OSError:
        return None
    return None


def coletar_metadados_execucao(extra: dict | None = None) -> dict:
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
        "ram_gb": _ler_memoria_total_gb(),
        "working_directory": str(RAIZ),
        "results_root": str(RAIZ_RESULTADOS),
        "device": DISPOSITIVO,
    }

    try:
        commit_git = subprocess.check_output(["git", "-C", str(RAIZ), "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        info["git_commit"] = commit_git
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


def configurar_log_experimento(nome_experimento: str, nome_script: str | None = None, metadados_extras: dict | None = None):
    dir_experimento = construir_dir_experimento(nome_experimento)
    dir_log = dir_experimento / "logs"
    carimbo_tempo = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    caminho_log = dir_log / f"{nome_experimento}_{carimbo_tempo}.log"

    registrador = logging.getLogger(f"benchnano.{nome_experimento}")
    registrador.setLevel(logging.INFO)
    registrador.handlers.clear()
    registrador.propagate = False

    manipulador_arquivo = logging.FileHandler(caminho_log, encoding="utf-8")
    manipulador_arquivo.setLevel(logging.INFO)
    formatador = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    manipulador_arquivo.setFormatter(formatador)
    registrador.addHandler(manipulador_arquivo)

    metadados = coletar_metadados_execucao({
        "experiment_name": nome_experimento,
        "script_name": nome_script,
        "results_dir": str(dir_experimento),
        "log_path": str(caminho_log),
    })
    if metadados_extras:
        metadados.update(metadados_extras)

    caminho_metadados = dir_experimento / "metadata" / "metadata.json"
    caminho_metadados.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho_metadados, "w", encoding="utf-8") as marcador:
        json.dump(metadados, marcador, indent=2, ensure_ascii=False)

    registrador.info("=== INÍCIO DO EXPERIMENTO ===")
    registrador.info("experiment_name=%s", nome_experimento)
    registrador.info("script_name=%s", nome_script)
    registrador.info("results_dir=%s", dir_experimento)
    registrador.info("log_path=%s", caminho_log)
    registrador.info("runtime_metadata=%s", json.dumps(metadados, ensure_ascii=False, sort_keys=True))
    return dir_experimento, caminho_log, registrador, metadados


def salvar_resumo_experimento(nome_experimento: str, metadados: dict, metricas: dict | None = None, caminho_resumo: Path | None = None):
    dir_experimento = construir_dir_experimento(nome_experimento)
    if caminho_resumo is None:
        caminho_resumo = dir_experimento / "summary.json"
    conteudo = {"experiment_name": nome_experimento, "metadata": metadados}
    if metricas is not None:
        conteudo["metrics"] = metricas
    with open(caminho_resumo, "w", encoding="utf-8") as marcador:
        json.dump(conteudo, marcador, indent=2, ensure_ascii=False)
    return caminho_resumo


def gravar_relatorio_execucao(nome_experimento: str, nome_relatorio: str, dados: dict):
    dir_experimento = construir_dir_experimento(nome_experimento)
    dir_relatorio = dir_experimento / "reports"
    dir_relatorio.mkdir(parents=True, exist_ok=True)
    caminho_saida = dir_relatorio / nome_relatorio
    with open(caminho_saida, "w", encoding="utf-8") as marcador:
        json.dump(dados, marcador, indent=2, ensure_ascii=False)
    return caminho_saida

# PT: "PET" contém "PE" como substring: PET (e as demais) são verificadas ANTES de PE em
#     extrair_rotulo, senão toda resposta "PET" seria lida como "PE".
# EN: "PET" contains "PE" as a substring: PET (and the others) are checked BEFORE PE in
#     extrair_rotulo, otherwise every "PET" answer would be read as "PE".
ROTULOS = ["PET", "PP", "PS", "PE"]

PROMPT_CLASSIFICACAO = (
    "This is a scanning electron microscope (SEM) image from a controlled "
    "reference sample containing a single known plastic polymer, showing "
    "particle(s) deposited on a filter membrane. Which polymer is this "
    "sample made of: PE (polyethylene), PET (polyethylene terephthalate), "
    "PP (polypropylene), or PS (polystyrene)? Answer with just one label: "
    "PE, PET, PP, or PS."
)

_RE_ROTULO = {rotulo: re.compile(rf"\b{rotulo}\b", re.IGNORECASE) for rotulo in ROTULOS}

# PT: Tarefa alternativa: regime de tamanho (NANOPLASTIC/MICROPLASTIC). O SmolVLM decide o
#     regime de tamanho a partir do recorte da bbox, não o polímero (ver
#     scripts/15_build_size_regime_manifest.py). A lista mantém a mesma convenção de ordem
#     de verificação por segurança, caso o prompt mude.
# EN: Alternative task: size regime (NANOPLASTIC/MICROPLASTIC). SmolVLM decides the size
#     regime from the bbox crop, not the polymer (see
#     scripts/15_build_size_regime_manifest.py). The list keeps the same check-order
#     convention as a safeguard in case the prompt changes.
ROTULOS_TAMANHO = ["NANOPLASTIC", "MICROPLASTIC"]

PROMPT_CLASSIFICACAO_TAMANHO = (
    "You are analyzing a scanning electron microscopy image containing "
    "plastic particles. The highlighted particle was detected by an "
    "object-detection model together with the image's physical scale "
    "information. Return NANOPLASTIC when the particle's characteristic "
    "size is below 1 micrometer. Return MICROPLASTIC otherwise. Answer "
    "with just one label: NANOPLASTIC or MICROPLASTIC."
)

import functools


@functools.lru_cache(maxsize=8)
def _regexes_rotulo(rotulos: tuple) -> dict:
    return {rotulo: re.compile(rf"\b{rotulo}\b", re.IGNORECASE) for rotulo in rotulos}


def extrair_rotulo(texto: str, rotulos: list | None = None) -> str:
    rotulos = rotulos or ROTULOS
    regexes = _regexes_rotulo(tuple(rotulos))
    for rotulo in rotulos:  # PT: a ordem importa, ver comentário acima / EN: order matters, see comment above
        if regexes[rotulo].search(texto):
            return rotulo
    return "Unresolved"


def carregar_particao_real(nome_particao: str) -> pd.DataFrame:
    df = pd.read_csv(MANIFESTO_REAL)
    return df[df["split"] == nome_particao].reset_index(drop=True)


def carregar_sintetico() -> pd.DataFrame:
    return pd.read_csv(MANIFESTO_SINTETICO)


def carregar_modelo_base(id_modelo: str = ID_MODELO):
    from transformers import AutoModelForVision2Seq, AutoProcessor

    processador = AutoProcessor.from_pretrained(id_modelo)
    processador.image_processor.do_image_splitting = False
    modelo = AutoModelForVision2Seq.from_pretrained(
        id_modelo, torch_dtype=torch.bfloat16
    ).to(DISPOSITIVO)
    return processador, modelo


def construir_exemplo_supervisionado(processador, origem_imagem, pergunta: str, resposta: str):
    imagem = origem_imagem.convert("RGB") if isinstance(origem_imagem, Image.Image) else Image.open(origem_imagem).convert("RGB")
    msg_usuario = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": pergunta}]}]
    so_prompt = processador.apply_chat_template(msg_usuario, add_generation_prompt=True)
    msgs_completas = msg_usuario + [{"role": "assistant", "content": [{"type": "text", "text": resposta}]}]
    texto_completo = processador.apply_chat_template(msgs_completas, add_generation_prompt=False)

    entradas_prompt = processador(text=so_prompt, images=[imagem], return_tensors="pt")
    entradas_completas = processador(text=texto_completo, images=[imagem], return_tensors="pt")

    compr_prompt = entradas_prompt["input_ids"].shape[1]
    rotulos = entradas_completas["input_ids"].clone()
    rotulos[:, :compr_prompt] = -100
    entradas_completas["labels"] = rotulos
    return {k: v.to(DISPOSITIVO) for k, v in entradas_completas.items()}


@torch.no_grad()
def classificar_imagem(processador, modelo, caminho_imagem, prompt: str | None = None) -> str:
    """PT: caminho_imagem aceita um Path (abre do disco) ou uma PIL.Image já em memória (útil
        para recortes gerados em tempo real, ex.: a partir de caixas previstas pelo YOLO,
        sem arquivo temporário).
    EN: caminho_imagem accepts a Path (opened from disk) or an in-memory PIL.Image (useful
        for crops generated on the fly, e.g. from YOLO predicted boxes, with no temporary
        file).
    """
    modelo.eval()
    texto_prompt = prompt or PROMPT_CLASSIFICACAO
    imagem = caminho_imagem.convert("RGB") if isinstance(caminho_imagem, Image.Image) else Image.open(caminho_imagem).convert("RGB")
    mensagens = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": texto_prompt}]}]
    prompt_chat = processador.apply_chat_template(mensagens, add_generation_prompt=True)
    entradas = processador(text=prompt_chat, images=[imagem], return_tensors="pt").to(DISPOSITIVO)
    ids_saida = modelo.generate(**entradas, max_new_tokens=20, do_sample=False)
    gerado = ids_saida[:, entradas["input_ids"].shape[1]:]
    return processador.batch_decode(gerado, skip_special_tokens=True)[0]


def avaliar_classificacao(processador, modelo, df: pd.DataFrame, rotulo_exp: str, dir_saida: Path = DIR_RELATORIOS,
                             prompt: str | None = None, rotulos: list | None = None):
    linhas_tab = []
    for _, r in df.iterrows():
        caminho_img = RAIZ / r["image_path"]
        bruto = classificar_imagem(processador, modelo, caminho_img, prompt=prompt)
        pred = extrair_rotulo(bruto, rotulos=rotulos)
        linhas_tab.append({
            "image_path": r["image_path"],
            "true_label": r["label"],
            "raw_answer": bruto.strip(),
            "pred_label": pred,
            "correct": pred == r["label"],
        })
    df_saida = pd.DataFrame(linhas_tab)
    acur = df_saida["correct"].mean() if len(df_saida) else float("nan")
    dir_saida.mkdir(parents=True, exist_ok=True)
    df_saida.to_csv(dir_saida / f"{rotulo_exp}_predictions.csv", index=False)
    return df_saida, acur


def ajuste_fino_lora(df_treino: pd.DataFrame, df_val: pd.DataFrame, df_teste: pd.DataFrame,
                   rotulo_exp: str, epocas: int = 6, acum_grad: int = 8, lr: float = 1e-4, semente: int = 42,
                   id_modelo: str = ID_MODELO, prompt: str | None = None, rotulos: list | None = None):
    """PT: Laço de treino LoRA compartilhado pelos scripts 05 (só real) e 06 (real +
        sintético): mesma receita e hiperparâmetros; só o conteúdo de df_treino muda. A
        seleção de checkpoint é sempre pela df_val real (nunca sintética), porque validar em
        dados sintéticos escolhe checkpoints que não generalizam para dados reais.
    EN: LoRA training loop shared by scripts 05 (real only) and 06 (real + synthetic): same
        recipe and hyperparameters; only the content of df_treino changes. Checkpoint
        selection is always on the real df_val (never synthetic), because validating on
        synthetic data picks checkpoints that do not generalize to real data.
    """
    import random as _random

    from peft import LoraConfig, get_peft_model
    from torch.optim import AdamW

    # PT: fixa também a RNG global do PyTorch (e do Python) antes de carregar o modelo:
    #     get_peft_model(..., init_lora_weights="gaussian") sorteia os pesos iniciais do
    #     adaptador com torch.rand*, além da ordem de embaralhamento do treino
    #     (gerador/_random.Random(semente) abaixo)
    # EN: also seeds the global PyTorch (and Python) RNG before loading the model:
    #     get_peft_model(..., init_lora_weights="gaussian") draws the adapter's initial
    #     weights with torch.rand*, in addition to the training shuffle order
    #     (gerador/_random.Random(semente) below)
    _random.seed(semente)
    torch.manual_seed(semente)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semente)

    texto_prompt = prompt or PROMPT_CLASSIFICACAO

    processador, modelo = carregar_modelo_base(id_modelo)
    modelo.config.use_cache = False
    modelo.gradient_checkpointing_enable()

    cfg_lora = LoraConfig(
        r=8, lora_alpha=8, lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        init_lora_weights="gaussian",
    )
    modelo = get_peft_model(modelo, cfg_lora)
    modelo.print_trainable_parameters()

    dir_adaptador = DIR_CHECKPOINTS / rotulo_exp
    otimizador = AdamW([p for p in modelo.parameters() if p.requires_grad], lr=lr)
    gerador = _random.Random(semente)
    melhor_acur_val = -1.0

    for epoca in range(1, epocas + 1):
        modelo.train()
        ordem = list(df_treino.index)
        gerador.shuffle(ordem)
        perda_acumulada, n_perda = 0.0, 0
        otimizador.zero_grad()
        for i, indice in enumerate(ordem):
            r = df_treino.loc[indice]
            entradas = construir_exemplo_supervisionado(processador, RAIZ / r["image_path"], texto_prompt, r["label"])
            saida = modelo(**entradas)
            perda = saida.loss / acum_grad
            perda.backward()
            perda_acumulada += saida.loss.item()
            n_perda += 1
            if (i + 1) % acum_grad == 0 or (i + 1) == len(ordem):
                torch.nn.utils.clip_grad_norm_([p for p in modelo.parameters() if p.requires_grad], 1.0)
                otimizador.step()
                otimizador.zero_grad()

        perda_media = perda_acumulada / max(n_perda, 1)

        modelo.config.use_cache = True
        resultados_val, acur_val = avaliar_classificacao(processador, modelo, df_val, f"{rotulo_exp}_epoch{epoca}_val",
                                                         prompt=prompt, rotulos=rotulos)
        modelo.config.use_cache = False
        print(f"epoch {epoca}/{epocas}  train_loss={perda_media:.4f}  real_val_acc={acur_val:.3f}")

        if acur_val > melhor_acur_val:
            melhor_acur_val = acur_val
            dir_adaptador.mkdir(parents=True, exist_ok=True)
            modelo.save_pretrained(dir_adaptador)
            print(f"  -> novo melhor adapter (real_val_acc={acur_val:.3f}) salvo em {dir_adaptador}")

    print(f"\nMelhor acurácia em real_val: {melhor_acur_val:.3f}. Adapter em {dir_adaptador}")

    # PT: recarrega o melhor adaptador salvo (não necessariamente o da última época)
    # EN: reloads the best saved adapter (not necessarily the last epoch's)
    from peft import PeftModel
    del modelo
    torch.cuda.empty_cache() if DISPOSITIVO == "cuda" else None
    processador_base, modelo_base = carregar_modelo_base(id_modelo)
    modelo = PeftModel.from_pretrained(modelo_base, dir_adaptador)
    modelo.config.use_cache = True

    resultados_teste, acur_teste = avaliar_classificacao(processador_base, modelo, df_teste, f"{rotulo_exp}_real_test",
                                                       prompt=prompt, rotulos=rotulos)
    print(f"\nAcurácia final REAL TEST ({rotulo_exp}, n={len(df_teste)}): {acur_teste:.3f}")
    return {"best_val_acc": melhor_acur_val, "test_acc": acur_teste, "test_results": resultados_teste}


def ajuste_fino_completo(df_treino: pd.DataFrame, df_val: pd.DataFrame, df_teste: pd.DataFrame,
                   rotulo_exp: str, epocas: int = 6, acum_grad: int = 8, lr: float = 1e-5, semente: int = 42,
                   id_modelo: str = ID_MODELO, prompt: str | None = None, rotulos: list | None = None):
    """PT: Ajuste fino completo (todos os parâmetros, sem LoRA), com a mesma estrutura de laço
        de `ajuste_fino_lora`, mas ajustando o modelo inteiro em vez de um adaptador de
        baixo posto. A documentação do SmolVLM2 recomenda ajuste fino completo em vez de
        LoRA/QLoRA para a variante de 500M. O lr padrão é menor que o do LoRA (1e-5 vs 1e-4)
        porque atualizar todos os pesos com a taxa do LoRA tende a divergir; ajustar se o
        treino ficar instável.
    EN: Full fine-tuning (all parameters, no LoRA), with the same loop structure as
        `ajuste_fino_lora`, but tuning the whole model instead of a low-rank adapter. The
        SmolVLM2 documentation recommends full fine-tuning instead of LoRA/QLoRA for the
        500M variant. The default lr is lower than LoRA's (1e-5 vs 1e-4) because updating
        every weight at the LoRA rate tends to diverge; adjust if training becomes unstable.
    """
    import random as _random

    from torch.optim import AdamW

    # PT: como em ajuste_fino_lora: fixa a RNG global do PyTorch (e do Python) antes de
    #     carregar o modelo, não só a ordem de embaralhamento do treino
    #     (gerador/_random.Random(semente) abaixo)
    # EN: as in ajuste_fino_lora: seeds the global PyTorch (and Python) RNG before loading
    #     the model, not only the training shuffle order (gerador/_random.Random(semente)
    #     below)
    _random.seed(semente)
    torch.manual_seed(semente)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semente)

    texto_prompt = prompt or PROMPT_CLASSIFICACAO

    processador, modelo = carregar_modelo_base(id_modelo)
    modelo.config.use_cache = False
    modelo.gradient_checkpointing_enable()
    modelo.train()

    n_parametros = sum(p.numel() for p in modelo.parameters())
    print(f"full fine-tuning: {n_parametros/1e6:.1f}M parametros treinaveis")

    dir_ckpt = DIR_CHECKPOINTS / rotulo_exp
    otimizador = AdamW(modelo.parameters(), lr=lr)
    gerador = _random.Random(semente)
    melhor_acur_val = -1.0

    for epoca in range(1, epocas + 1):
        modelo.train()
        ordem = list(df_treino.index)
        gerador.shuffle(ordem)
        perda_acumulada, n_perda = 0.0, 0
        otimizador.zero_grad()
        for i, indice in enumerate(ordem):
            r = df_treino.loc[indice]
            entradas = construir_exemplo_supervisionado(processador, RAIZ / r["image_path"], texto_prompt, r["label"])
            saida = modelo(**entradas)
            perda = saida.loss / acum_grad
            perda.backward()
            perda_acumulada += saida.loss.item()
            n_perda += 1
            if (i + 1) % acum_grad == 0 or (i + 1) == len(ordem):
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
                otimizador.step()
                otimizador.zero_grad()

        perda_media = perda_acumulada / max(n_perda, 1)

        modelo.config.use_cache = True
        resultados_val, acur_val = avaliar_classificacao(processador, modelo, df_val, f"{rotulo_exp}_epoch{epoca}_val",
                                                         prompt=prompt, rotulos=rotulos)
        modelo.config.use_cache = False
        print(f"epoch {epoca}/{epocas}  train_loss={perda_media:.4f}  real_val_acc={acur_val:.3f}")

        if acur_val > melhor_acur_val:
            melhor_acur_val = acur_val
            dir_ckpt.mkdir(parents=True, exist_ok=True)
            modelo.save_pretrained(dir_ckpt)
            processador.save_pretrained(dir_ckpt)
            print(f"  -> novo melhor checkpoint (real_val_acc={acur_val:.3f}) salvo em {dir_ckpt}")

    print(f"\nMelhor acurácia em real_val: {melhor_acur_val:.3f}. Checkpoint em {dir_ckpt}")

    del modelo
    torch.cuda.empty_cache() if DISPOSITIVO == "cuda" else None
    from transformers import AutoModelForVision2Seq, AutoProcessor
    processador_teste = AutoProcessor.from_pretrained(dir_ckpt)
    processador_teste.image_processor.do_image_splitting = False
    modelo_teste = AutoModelForVision2Seq.from_pretrained(dir_ckpt, torch_dtype=torch.bfloat16).to(DISPOSITIVO)
    modelo_teste.config.use_cache = True

    resultados_teste, acur_teste = avaliar_classificacao(processador_teste, modelo_teste, df_teste, f"{rotulo_exp}_real_test",
                                                       prompt=prompt, rotulos=rotulos)
    print(f"\nAcurácia final REAL TEST ({rotulo_exp}, n={len(df_teste)}): {acur_teste:.3f}")
    return {"best_val_acc": melhor_acur_val, "test_acc": acur_teste, "test_results": resultados_teste}
