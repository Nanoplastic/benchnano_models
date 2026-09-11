"""Utilitários compartilhados pelos experimentos controlados (10 repetições,
seed explícita 0-9) -- só para os 6 treinos citados no artigo main_V5.tex:
YOLO26L principal (13), SmolVLM LoRA polímero real-only/real+sintético/
bigvlm (05/06/09), SmolVLM LoRA/full regime de tamanho (17/18). Ver plano em
/home/mi/.claude/plans/parallel-giggling-ocean.md.

Não modifica nada em experimentos/scripts/ -- todo output novo (summary.json
por repetição, log de máquina, relatório final) vai em
experimentos_controlados/. Checkpoints e CSVs de predição do SmolVLM
continuam na mesma pasta compartilhada que o projeto já usa
(experimentos/resultados/checkpoints/ e reports/, ver vlm_common.py), só com
tags únicas "_ctrl_repN" -- é a MESMA convenção que os scripts originais já
usam pras 4 execuções ad hoc (_run2/_run3/_run4), só que padronizada e
completa (10 execuções, seed explícita).
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

CTRL_ROOT = Path(__file__).resolve().parent.parent  # .../experimentos_controlados
DATASETMINA_ROOT = CTRL_ROOT.parent  # .../datasetMINA
EXPERIMENTOS_SCRIPTS = DATASETMINA_ROOT / "experimentos" / "scripts"
sys.path.insert(0, str(EXPERIMENTOS_SCRIPTS))  # permite `import vlm_common` etc.

RESULTADOS_DIR = CTRL_ROOT / "resultados"
RUNS_DIR = CTRL_ROOT / "runs"
EXPERIMENTOS_REPORTS_DIR = DATASETMINA_ROOT / "experimentos" / "resultados" / "reports"
LOGS_MAQUINA_DIR = CTRL_ROOT / "logs_maquina"
RELATORIO_FINAL_DIR = CTRL_ROOT / "relatorio_final"
QUEUE_STATE_PATH = CTRL_ROOT / ".queue_state.json"
CURRENT_JOB_PATH = LOGS_MAQUINA_DIR / "current_job.json"

# Log de execução (para o artigo): uma linha JSON por repetição rodada, com
# tempo de início/fim/duração, máquina, versões de software e uso de
# recursos -- ver run_queue.py:run_rep_subprocess. Log bruto (stdout/stderr)
# de cada repetição fica em EXEC_STDOUT_DIR. Sessões da fila (pode ser
# pausada/retomada em dias diferentes) ficam em QUEUE_SESSIONS_PATH.
EXEC_LOG_PATH = LOGS_MAQUINA_DIR / "execution_log.jsonl"
EXEC_STDOUT_DIR = LOGS_MAQUINA_DIR / "exec_logs"
QUEUE_SESSIONS_PATH = LOGS_MAQUINA_DIR / "queue_sessions.jsonl"

N_REPS = 10
SEEDS = list(range(N_REPS))  # 0..9 -- convenção: seed da repetição N é sempre N


# --------------------------------------------------------------------------
# Controle de aleatoriedade
# --------------------------------------------------------------------------

def seed_everything(seed: int):
    """Controla toda fonte de aleatoriedade do treino.

    vlm_common.finetune_lora/finetune_full só seedam o embaralhamento dos
    dados (random.Random(seed) local) -- a inicialização dos pesos LoRA
    (init_lora_weights='gaussian') e o dropout usam a RNG global do PyTorch,
    que NÃO é fixada hoje (documentado na Seção de Reprodutibilidade do
    artigo). Sem chamar isto antes de cada repetição, "10 repetições" do
    SmolVLM não teria seed rastreável ponta a ponta.

    Para os treinos do YOLO26L (via ultralytics), a seed vai direto no
    kwarg `seed=` de `model.train()` (Ultralytics já é determinístico
    bit-a-bit com seed fixa, ver comentário em
    experimentos/scripts/44_train_cv_fold.py) -- chamar esta função antes
    também não atrapalha nesse caso.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------
# Job atual (lido pelo system_monitor.py pra rotular cada amostra)
# --------------------------------------------------------------------------

def set_current_job(experiment: str, rep: int, extra: dict | None = None):
    LOGS_MAQUINA_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment": experiment,
        "repetition": rep,
        "seed": rep,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    with open(CURRENT_JOB_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def clear_current_job():
    if CURRENT_JOB_PATH.exists():
        CURRENT_JOB_PATH.unlink()


# --------------------------------------------------------------------------
# Ambiente / proveniência
# --------------------------------------------------------------------------

def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(DATASETMINA_ROOT), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return None


def _cpu_model() -> str | None:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or None


def _nvidia_driver_version() -> str | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
        return out.splitlines()[0] if out else None
    except Exception:
        return None


def collect_environment() -> dict:
    """Snapshot de máquina + software -- usado tanto no summary.json por
    repetição quanto no log de execução (log_execution). Inclui o que os
    checklists de reprodutibilidade de ML pedem sobre infraestrutura
    computacional (NeurIPS/AAAI/McGill ML Reproducibility Checklist):
    hardware (CPU/GPU/RAM), driver, e versões de todo software relevante."""
    info = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "git_commit": _git_commit(),
        "cpu_model": _cpu_model(),
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
        info["nvidia_driver_version"] = _nvidia_driver_version()
    for mod_name in ("transformers", "peft", "ultralytics"):
        try:
            mod = __import__(mod_name)
            info[f"{mod_name}_version"] = mod.__version__
        except Exception:
            info[f"{mod_name}_version"] = None
    return info


def write_session_manifest():
    """Snapshot único gravado no início da fila inteira -- pip freeze
    completo + o que collect_environment() já cobre. Complementa (não
    substitui) o summary.json por repetição."""
    LOGS_MAQUINA_DIR.mkdir(parents=True, exist_ok=True)
    manifest = collect_environment()
    try:
        manifest["pip_freeze"] = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        ).splitlines()
    except Exception:
        manifest["pip_freeze"] = None
    out_path = LOGS_MAQUINA_DIR / "session_manifest.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return out_path


# --------------------------------------------------------------------------
# Resultado por repetição
# --------------------------------------------------------------------------

def rep_dir(experiment: str, rep: int, base: Path = RESULTADOS_DIR) -> Path:
    d = base / experiment / f"rep_{rep:02d}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_rep_summary(experiment: str, rep: int, seed: int, metrics: dict,
                      extra_metadata: dict | None = None) -> Path:
    d = rep_dir(experiment, rep)
    payload = {
        "experiment": experiment,
        "repetition_index": rep,
        "repetition_seed": seed,
        "environment": collect_environment(),
        "metrics": metrics,
    }
    if extra_metadata:
        payload["metadata"] = extra_metadata
    out_path = d / "summary.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return out_path


def load_all_rep_summaries(experiment: str, base: Path = RESULTADOS_DIR) -> list:
    exp_dir = base / experiment
    out = []
    if not exp_dir.exists():
        return out
    for rep_d in sorted(exp_dir.glob("rep_*")):
        summary_path = rep_d / "summary.json"
        if summary_path.exists():
            with open(summary_path, encoding="utf-8") as f:
                out.append(json.load(f))
    return out


# --------------------------------------------------------------------------
# Estatística: agregação das 10 repetições
# --------------------------------------------------------------------------

def mean_std_ci95(values: list) -> dict:
    """IC 95% via distribuição t (n=10, amostra pequena) -- mesma disciplina
    de relato de média+variância de Jain, "The Art of Computer Systems
    Performance Analysis" cap. 13.2, já citado no artigo
    (Jain1991ArtOfPerformanceAnalysis) exatamente para este tipo de
    situação (poucas repetições, reportar média e dispersão junto)."""
    from scipy import stats

    arr = np.asarray(values, dtype=float)
    n = len(arr)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if n > 1 else 0.0
    if n > 1:
        t_crit = stats.t.ppf(0.975, df=n - 1)
        margin = t_crit * std / np.sqrt(n)
        ci_low, ci_high = mean - margin, mean + margin
    else:
        ci_low = ci_high = mean
    return {
        "n": n, "mean": mean, "std": std,
        "ci95_low": float(ci_low), "ci95_high": float(ci_high),
        "values": [float(v) for v in values],
    }


# --------------------------------------------------------------------------
# Estado da fila (resumível)
# --------------------------------------------------------------------------

def load_queue_state() -> dict:
    if QUEUE_STATE_PATH.exists():
        with open(QUEUE_STATE_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {"completed_reps": {}}


def save_queue_state(state: dict):
    with open(QUEUE_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def mark_rep_done(state: dict, experiment: str, rep: int):
    state.setdefault("completed_reps", {}).setdefault(experiment, [])
    if rep not in state["completed_reps"][experiment]:
        state["completed_reps"][experiment].append(rep)
    save_queue_state(state)


def is_rep_done(state: dict, experiment: str, rep: int) -> bool:
    return rep in state.get("completed_reps", {}).get(experiment, [])


# --------------------------------------------------------------------------
# Log de execução -- para o artigo: tempo de cada repetição, tempo total,
# máquina, versões de software (ver run_queue.py:run_rep_subprocess)
# --------------------------------------------------------------------------

def human_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def append_jsonl(path: Path, record: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def log_execution(record: dict):
    """Acrescenta uma linha em EXEC_LOG_PATH (execution_log.jsonl) -- um
    registro por repetição executada, de qualquer experimento da fila."""
    append_jsonl(EXEC_LOG_PATH, record)


def load_execution_log() -> list:
    return load_jsonl(EXEC_LOG_PATH)


def summarize_resource_window(start_utc: datetime, end_utc: datetime) -> dict:
    """Resume CPU/RAM/GPU no intervalo [start_utc, end_utc] a partir dos CSVs
    diários do system_monitor.py (pode abranger mais de um dia)."""
    days = set()
    d = start_utc.date()
    while d <= end_utc.date():
        days.add(d)
        d += timedelta(days=1)

    samples = []
    for day in sorted(days):
        path = LOGS_MAQUINA_DIR / f"system_monitor_{day.strftime('%Y%m%d')}.csv"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                try:
                    ts = datetime.fromisoformat(row["timestamp_utc"])
                except Exception:
                    continue
                if start_utc <= ts <= end_utc:
                    samples.append(row)

    if not samples:
        return {"n_samples": 0}

    def _agg(key):
        vals = []
        for r in samples:
            v = r.get(key)
            if v not in (None, ""):
                try:
                    vals.append(float(v))
                except ValueError:
                    pass
        if not vals:
            return {"avg": None, "max": None}
        return {"avg": round(sum(vals) / len(vals), 2), "max": round(max(vals), 2)}

    return {
        "n_samples": len(samples),
        "cpu_pct": _agg("cpu_pct"),
        "ram_used_gb": _agg("ram_used_gb"),
        "gpu_util_pct": _agg("gpu_util_pct"),
        "gpu_mem_used_mb": _agg("gpu_mem_used_mb"),
        "gpu_temp_c": _agg("gpu_temp_c"),
        "gpu_power_w": _agg("gpu_power_w"),
    }
