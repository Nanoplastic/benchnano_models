"""Monitor contínuo da máquina / Continuous machine monitor.

PT: Registra o estado da máquina em paralelo à fila de repetições controladas (iniciado e
    encerrado por run_queue.py).
    A cada `--interval` segundos (padrão 10 s) amostra
    temperatura/potência/utilização/memória da GPU (pynvml), uso de CPU e RAM (psutil),
    espaço livre em disco e a tarefa/repetição em execução (lida de
    logs_maquina/current_job.json, atualizado por ctrl_common.definir_tarefa_atual a cada
    treino). Grava uma linha por amostra em logs_maquina/system_monitor_YYYYMMDD.csv (um
    arquivo por dia).
    Uso:
    python3 system_monitor.py --interval 10
    Roda até receber SIGTERM/SIGINT; run_queue.py o inicia como subprocesso e o encerra no
    fim da fila ou se detectar que ele parou.
EN: Records the machine state in parallel with the controlled repetitions queue (started and
    stopped by run_queue.py).
    Every `--interval` seconds (default 10 s) samples GPU
    temperature/power/utilization/memory (pynvml), CPU and RAM usage (psutil), free disk
    space and the running job/repetition (read from logs_maquina/current_job.json, updated
    by ctrl_common.definir_tarefa_atual for each training run). Writes one line per sample
    to logs_maquina/system_monitor_YYYYMMDD.csv (one file per day).
    Usage:
    python3 system_monitor.py --interval 10
    Runs until it receives SIGTERM/SIGINT; run_queue.py starts it as a subprocess and stops
    it at the end of the queue or if it detects that it has died.
"""
import argparse
import csv
import json
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ctrl_common import CAMINHO_TAREFA_ATUAL, LOGS_MAQUINA_DIR

_PARAR = False


def _tratar_sinal(num_sinal, quadro):
    global _PARAR
    _PARAR = True


def _ler_tarefa_atual() -> dict:
    if CAMINHO_TAREFA_ATUAL.exists():
        try:
            with open(CAMINHO_TAREFA_ATUAL, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _estatisticas_gpu() -> dict:
    try:
        import pynvml
        pynvml.nvmlInit()
        marcador = pynvml.nvmlDeviceGetHandleByIndex(0)
        temp = pynvml.nvmlDeviceGetTemperature(marcador, pynvml.NVML_TEMPERATURE_GPU)
        potencia_mw = pynvml.nvmlDeviceGetPowerUsage(marcador)
        utilizacao = pynvml.nvmlDeviceGetUtilizationRates(marcador)
        memoria = pynvml.nvmlDeviceGetMemoryInfo(marcador)
        nome = pynvml.nvmlDeviceGetName(marcador)
        if isinstance(nome, bytes):
            nome = nome.decode()
        pynvml.nvmlShutdown()
        return {
            "gpu_name": nome,
            "gpu_temp_c": temp,
            "gpu_power_w": round(potencia_mw / 1000.0, 2),
            "gpu_util_pct": utilizacao.gpu,
            "gpu_mem_used_mb": round(memoria.used / (1024 ** 2), 1),
            "gpu_mem_total_mb": round(memoria.total / (1024 ** 2), 1),
        }
    except Exception as e:
        return {
            "gpu_name": None, "gpu_temp_c": None, "gpu_power_w": None,
            "gpu_util_pct": None, "gpu_mem_used_mb": None, "gpu_mem_total_mb": None,
            "gpu_error": str(e),
        }


def _amostrar() -> dict:
    tarefa = _ler_tarefa_atual()
    disco = psutil.disk_usage(str(LOGS_MAQUINA_DIR.parent))
    linha_tab = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_pct": psutil.cpu_percent(interval=None),
        "ram_used_gb": round(psutil.virtual_memory().used / (1024 ** 3), 2),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "disk_free_gb": round(disco.free / (1024 ** 3), 2),
        "current_experiment": tarefa.get("experiment"),
        "current_repetition": tarefa.get("repetition"),
        "current_seed": tarefa.get("seed"),
    }
    linha_tab.update(_estatisticas_gpu())
    return linha_tab


NOMES_CAMPOS = [
    "timestamp_utc", "current_experiment", "current_repetition", "current_seed",
    "cpu_pct", "ram_used_gb", "ram_total_gb", "disk_free_gb",
    "gpu_name", "gpu_temp_c", "gpu_power_w", "gpu_util_pct",
    "gpu_mem_used_mb", "gpu_mem_total_mb", "gpu_error",
]


def _caminho_csv_do_dia() -> Path:
    LOGS_MAQUINA_DIR.mkdir(parents=True, exist_ok=True)
    dia = datetime.now(timezone.utc).strftime("%Y%m%d")
    return LOGS_MAQUINA_DIR / f"system_monitor_{dia}.csv"


def principal():
    analisador = argparse.ArgumentParser(description=__doc__)
    analisador.add_argument("--interval", type=float, default=10.0, help="segundos entre amostras")
    args = analisador.parse_args()

    signal.signal(signal.SIGTERM, _tratar_sinal)
    signal.signal(signal.SIGINT, _tratar_sinal)

    print(f"[system_monitor] iniciado, amostrando a cada {args.interval}s. "
          f"Log em {LOGS_MAQUINA_DIR}/system_monitor_YYYYMMDD.csv")

    psutil.cpu_percent(interval=None)  # PT: a primeira chamada sempre retorna 0.0; descartada / EN: the first call always returns 0.0; discarded

    caminho_atual = None
    fh = None
    escritor = None

    try:
        while not _PARAR:
            caminho = _caminho_csv_do_dia()
            if caminho != caminho_atual:
                if fh:
                    fh.close()
                novo = not caminho.exists()
                fh = open(caminho, "a", newline="", encoding="utf-8")
                escritor = csv.DictWriter(fh, fieldnames=NOMES_CAMPOS)
                if novo:
                    escritor.writeheader()
                caminho_atual = caminho

            linha_tab = _amostrar()
            escritor.writerow(linha_tab)
            fh.flush()
            time.sleep(args.interval)
    finally:
        if fh:
            fh.close()
        print("[system_monitor] encerrado.")


if __name__ == "__main__":
    principal()
