"""Utilitários dos experimentos controlados / Controlled experiment utilities.

PT: Utilitários compartilhados pelos experimentos controlados (10 repetições, semente
    explícita 0-9) dos 6 treinos: YOLO26L principal (13), SmolVLM LoRA de polímero só real /
    real + sintético / VLM maior (05/06/09) e SmolVLM LoRA/completo de regime de tamanho
    (17/18), além das reavaliações com max_det=2000 e com a regra de regime de tamanho
    "eq1" (ver experimentos/scripts/60-81 e final_report.py).
    Não modifica nada em experimentos/scripts/. Toda saída nova (summary.json por repetição,
    log da máquina, relatório final) vai para experimentos_controlados/. Checkpoints e CSVs
    de predição do SmolVLM ficam na pasta compartilhada
    (experimentos/resultados/checkpoints/ e reports/, ver vlm_common.py), com rótulos únicos
    "_ctrl_seedN", a mesma convenção de _run2/_run3/_run4, padronizada para 10 execuções com
    semente explícita.
EN: Utilities shared by the controlled experiments (10 repetitions, explicit seed 0-9) of
    the 6 training runs: main YOLO26L (13), SmolVLM polymer LoRA real only / real +
    synthetic / larger VLM (05/06/09) and SmolVLM LoRA/full size regime (17/18), plus the
    re-evaluations with max_det=2000 and with the "eq1" size regime rule (see
    experimentos/scripts/60-81 and final_report.py).
    Does not modify anything in experimentos/scripts/. All new output (per-repetition
    summary.json, machine log, final report) goes to experimentos_controlados/. SmolVLM
    checkpoints and prediction CSVs stay in the shared folder
    (experimentos/resultados/checkpoints/ and reports/, see vlm_common.py), with unique
    "_ctrl_seedN" tags, the same convention as _run2/_run3/_run4, standardized to 10 runs
    with explicit seeds.
"""
import csv
import json
import os
import platform
import random
import socket
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

RAIZ_CTRL = Path(__file__).resolve().parent.parent  # PT: .../experimentos_controlados / EN: .../experimentos_controlados
RAIZ_PROJETO_ORIGEM = RAIZ_CTRL.parent  # PT: raiz do repositório / EN: repository root
SCRIPTS_EXPERIMENTOS = RAIZ_PROJETO_ORIGEM / "experimentos" / "scripts"
sys.path.insert(0, str(SCRIPTS_EXPERIMENTOS))  # PT: permite `import vlm_common` etc. / EN: allows `import vlm_common` etc.

RESULTADOS_DIR = RAIZ_CTRL / "resultados"
DIR_EXECUCOES = RAIZ_CTRL / "runs"
DIR_RELATORIOS_EXPERIMENTOS = RAIZ_PROJETO_ORIGEM / "experimentos" / "resultados" / "reports"
LOGS_MAQUINA_DIR = RAIZ_CTRL / "logs_maquina"
RELATORIO_FINAL_DIR = RAIZ_CTRL / "relatorio_final"
CAMINHO_ESTADO_FILA = RAIZ_CTRL / ".queue_state.json"
CAMINHO_TAREFA_ATUAL = LOGS_MAQUINA_DIR / "current_job.json"

# PT: Log de execução: uma linha JSON por repetição, com início/fim/duração, máquina,
#     versões de software e uso de recursos (ver run_queue.py:executar_rep_subprocesso). A
#     saída bruta (stdout/stderr) de cada repetição fica em DIR_STDOUT_EXECUCAO. As sessões
#     da fila (que pode ser pausada e retomada em dias diferentes) ficam em
#     CAMINHO_SESSOES_FILA.
# EN: Execution log: one JSON line per repetition, with start/end/duration, machine,
#     software versions and resource usage (see run_queue.py:executar_rep_subprocesso). Each
#     repetition's raw output (stdout/stderr) goes to DIR_STDOUT_EXECUCAO. Queue sessions
#     (the queue can be paused and resumed on different days) go to CAMINHO_SESSOES_FILA.
CAMINHO_LOG_EXECUCAO = LOGS_MAQUINA_DIR / "execution_log.jsonl"
DIR_STDOUT_EXECUCAO = LOGS_MAQUINA_DIR / "exec_logs"
CAMINHO_SESSOES_FILA = LOGS_MAQUINA_DIR / "queue_sessions.jsonl"

N_REPS = 10
SEMENTES = list(range(N_REPS))  # PT: 0..9; a semente da repetição N é sempre N / EN: 0..9; the seed of repetition N is always N


# PT: Controle de aleatoriedade
# EN: Randomness control

def fixar_sementes(semente: int):
    """PT: Fixa todas as fontes de aleatoriedade do treino.
        vlm_common.ajuste_fino_lora/ajuste_fino_completo semeiam o embaralhamento dos dados
        (random.Random(semente) local); a inicialização dos pesos LoRA
        (init_lora_weights='gaussian') e o dropout usam a RNG global do PyTorch. Chamar esta
        função antes de cada repetição torna a semente rastreável de ponta a ponta.
        Nos treinos do YOLO26L (Ultralytics), a semente vai no argumento `seed=` de
        `model.train()` (o Ultralytics é determinístico bit a bit com semente fixa); chamar
        esta função antes também não interfere.
    EN: Seeds every source of randomness in training.
        vlm_common.ajuste_fino_lora/ajuste_fino_completo seed the data shuffle (local
        random.Random(semente)); LoRA weight initialization (init_lora_weights='gaussian')
        and dropout use the global PyTorch RNG. Calling this function before each repetition
        makes the seed traceable end to end.
        In YOLO26L (Ultralytics) training, the seed goes into the `seed=` argument of
        `model.train()` (Ultralytics is bit-for-bit deterministic with a fixed seed);
        calling this function beforehand does not interfere either.
    """
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(semente)


# PT: Tarefa atual (lida pelo system_monitor.py para rotular cada amostra)
# EN: Current job (read by system_monitor.py to label each sample)

def definir_tarefa_atual(experimento: str, rep: int, extra: dict | None = None):
    LOGS_MAQUINA_DIR.mkdir(parents=True, exist_ok=True)
    conteudo = {
        "experiment": experimento,
        "repetition": rep,
        "seed": rep,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        conteudo.update(extra)
    with open(CAMINHO_TAREFA_ATUAL, "w", encoding="utf-8") as f:
        json.dump(conteudo, f, indent=2, ensure_ascii=False)


def limpar_tarefa_atual():
    if CAMINHO_TAREFA_ATUAL.exists():
        CAMINHO_TAREFA_ATUAL.unlink()


# PT: Ambiente / proveniência
# EN: Environment / provenance

def _commit_git() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(RAIZ_PROJETO_ORIGEM), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return None


def _modelo_cpu() -> str | None:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for linha in f:
                if linha.lower().startswith("model name"):
                    return linha.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or None


def _versao_driver_nvidia() -> str | None:
    try:
        saida = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        return saida.splitlines()[0] if saida else None
    except Exception:
        return None


def coletar_ambiente() -> dict:
    """PT: Retrato da máquina e do software, usado no summary.json de cada repetição e no log
        de execução (registrar_execucao): hardware (CPU/GPU/RAM), driver e versões do
        software relevante.
    EN: Machine and software snapshot, used in each repetition's summary.json and in the
        execution log (registrar_execucao): hardware (CPU/GPU/RAM), driver and versions of
        the relevant software.
    """
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "git_commit": _commit_git(),
        "cpu_model": _modelo_cpu(),
        "n_cpus_logical": os.cpu_count(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    try:
        import psutil
        info["ram_total_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    except Exception:
        info["ram_total_gb"] = None
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["nvidia_driver_version"] = _versao_driver_nvidia()
    for nome_modulo in ("transformers", "peft", "ultralytics"):
        try:
            modulo = __import__(nome_modulo)
            info[f"{nome_modulo}_version"] = modulo.__version__
        except Exception:
            info[f"{nome_modulo}_version"] = None
    return info


def gravar_manifesto_sessao():
    """PT: Retrato único gravado no início da fila: pip freeze completo + o que
        coletar_ambiente() já cobre. Complementa o summary.json de cada repetição.
    EN: Single snapshot written at the start of the queue: full pip freeze + what
        coletar_ambiente() already covers. Complements each repetition's summary.json.
    """
    LOGS_MAQUINA_DIR.mkdir(parents=True, exist_ok=True)
    manifesto = coletar_ambiente()
    try:
        manifesto["pip_freeze"] = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        ).splitlines()
    except Exception:
        manifesto["pip_freeze"] = None
    caminho_saida = LOGS_MAQUINA_DIR / "session_manifest.json"
    with open(caminho_saida, "w", encoding="utf-8") as f:
        json.dump(manifesto, f, indent=2, ensure_ascii=False)
    return caminho_saida


# PT: Resultado por repetição
# EN: Per-repetition result

def dir_rep(experimento: str, rep: int, base: Path = RESULTADOS_DIR) -> Path:
    d = base / experimento / f"rep_{rep:02d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def salvar_resumo_rep(experimento: str, rep: int, semente: int, metricas: dict,
                      metadados_extras: dict | None = None) -> Path:
    d = dir_rep(experimento, rep)
    conteudo = {
        "experiment": experimento,
        "repetition_index": rep,
        "repetition_seed": semente,
        "environment": coletar_ambiente(),
        "metrics": metricas,
    }
    if metadados_extras:
        conteudo["metadata"] = metadados_extras
    caminho_saida = d / "summary.json"
    with open(caminho_saida, "w", encoding="utf-8") as f:
        json.dump(conteudo, f, indent=2, ensure_ascii=False)
    return caminho_saida


def carregar_todos_resumos_rep(experimento: str, base: Path = RESULTADOS_DIR) -> list:
    dir_exp = base / experimento
    saida = []
    if not dir_exp.exists():
        return saida
    for rep_d in sorted(dir_exp.glob("rep_*")):
        caminho_resumo = rep_d / "summary.json"
        if caminho_resumo.exists():
            with open(caminho_resumo, encoding="utf-8") as f:
                saida.append(json.load(f))
    return saida


# PT: Estatística: agregação das 10 repetições
# EN: Statistics: aggregation of the 10 repetitions

def media_dp_ic95(valores: list) -> dict:
    """PT: IC 95% pela distribuição t (n=10, amostra pequena), relatando média e dispersão
        juntas.
    EN: 95% CI from the t distribution (n=10, small sample), reporting mean and spread
        together.
    """
    from scipy import stats

    matriz = np.asarray(valores, dtype=float)
    n = len(matriz)
    media = float(matriz.mean())
    dp = float(matriz.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        t_critico = stats.t.ppf(0.975, df=n - 1)
        margem = t_critico * dp / np.sqrt(n)
        ic_inf, ic_sup = media - margem, media + margem
    else:
        ic_inf = ic_sup = media
    return {
        "n": n, "mean": media, "std": dp,
        "ci95_low": float(ic_inf), "ci95_high": float(ic_sup),
        "values": [float(v) for v in valores],
    }


# PT: Estado da fila (retomável)
# EN: Queue state (resumable)

def carregar_estado_fila() -> dict:
    if CAMINHO_ESTADO_FILA.exists():
        with open(CAMINHO_ESTADO_FILA, encoding="utf-8") as f:
            return json.load(f)
    return {"completed_reps": {}}


def salvar_estado_fila(estado: dict):
    with open(CAMINHO_ESTADO_FILA, "w", encoding="utf-8") as f:
        json.dump(estado, f, indent=2, ensure_ascii=False)


def marcar_rep_concluida(estado: dict, experimento: str, rep: int):
    estado.setdefault("completed_reps", {}).setdefault(experimento, [])
    if rep not in estado["completed_reps"][experimento]:
        estado["completed_reps"][experimento].append(rep)
    salvar_estado_fila(estado)


def rep_concluida(estado: dict, experimento: str, rep: int) -> bool:
    return rep in estado.get("completed_reps", {}).get(experimento, [])


# PT: Log de execução: tempo de cada repetição, tempo total, máquina e versões de software
#     (ver run_queue.py:executar_rep_subprocesso)
# EN: Execution log: time of each repetition, total time, machine and software versions (see
#     run_queue.py:executar_rep_subprocesso)

def duracao_legivel(segundos: float) -> str:
    segundos = int(round(segundos))
    h, resto = divmod(segundos, 3600)
    m, s = divmod(resto, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def anexar_jsonl(caminho: Path, registro: dict):
    caminho.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False) + "\n")


def carregar_jsonl(caminho: Path) -> list:
    if not caminho.exists():
        return []
    saida = []
    with open(caminho, encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if linha:
                saida.append(json.loads(linha))
    return saida


def registrar_execucao(registro: dict):
    """PT: Acrescenta uma linha em CAMINHO_LOG_EXECUCAO (execution_log.jsonl): um registro por
        repetição executada, de qualquer experimento da fila.
    EN: Appends one line to CAMINHO_LOG_EXECUCAO (execution_log.jsonl): one record per
        executed repetition, for any experiment in the queue.
    """
    anexar_jsonl(CAMINHO_LOG_EXECUCAO, registro)


def carregar_log_execucao() -> list:
    return carregar_jsonl(CAMINHO_LOG_EXECUCAO)


def resumir_janela_recursos(inicio_utc: datetime, fim_utc: datetime) -> dict:
    """PT: Resume CPU/RAM/GPU no intervalo [inicio_utc, fim_utc] a partir dos CSVs diários do
        system_monitor.py (o intervalo pode abranger mais de um dia).
    EN: Summarizes CPU/RAM/GPU over [inicio_utc, fim_utc] from the daily system_monitor.py
        CSVs (the interval may span more than one day).
    """
    dias = set()
    d = inicio_utc.date()
    while d <= fim_utc.date():
        dias.add(d)
        d += timedelta(days=1)

    amostras = []
    for dia in sorted(dias):
        caminho = LOGS_MAQUINA_DIR / f"system_monitor_{dia.strftime('%Y%m%d')}.csv"
        if not caminho.exists():
            continue
        with open(caminho, encoding="utf-8") as f:
            for linha_tab in csv.DictReader(f):
                try:
                    ts = datetime.fromisoformat(linha_tab["timestamp_utc"])
                except Exception:
                    continue
                if inicio_utc <= ts <= fim_utc:
                    amostras.append(linha_tab)

    if not amostras:
        return {"n_samples": 0}

    def _agregar(chave):
        vals = []
        for r in amostras:
            v = r.get(chave)
            if v not in (None, ""):
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
        if not vals:
            return {"avg": None, "max": None}
        return {"avg": round(sum(vals) / len(vals), 2), "max": round(max(vals), 2)}

    return {
        "n_samples": len(amostras),
        "cpu_pct": _agregar("cpu_pct"),
        "ram_used_gb": _agregar("ram_used_gb"),
        "gpu_util_pct": _agregar("gpu_util_pct"),
        "gpu_mem_used_mb": _agregar("gpu_mem_used_mb"),
        "gpu_temp_c": _agregar("gpu_temp_c"),
        "gpu_power_w": _agregar("gpu_power_w"),
    }
