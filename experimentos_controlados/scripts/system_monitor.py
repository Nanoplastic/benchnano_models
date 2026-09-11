"""Logger contínuo de estado da máquina, rodando em paralelo à fila inteira
de repetições controladas (iniciado e encerrado por run_queue.py).

Amostra a cada `--interval` segundos (default 10s): temperatura/potência/
utilização/memória da GPU (via pynvml), uso de CPU e RAM (via psutil),
espaço livre em disco, e o job/repetição em execução naquele instante (lido
de logs_maquina/current_job.json, atualizado por ctrl_common.set_current_job
a cada novo treino disparado). Grava uma linha por amostra em
logs_maquina/system_monitor_YYYYMMDD.csv (um arquivo novo por dia, pra não
crescer um único CSV gigante ao longo de ~60h).

Uso:
    python3 system_monitor.py --interval 10
    (roda até receber SIGTERM/SIGINT -- run_queue.py o inicia como
    subprocesso e encerra no fim da fila ou se detectar que ele morreu)
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
from ctrl_common import CURRENT_JOB_PATH, LOGS_MAQUINA_DIR

_STOP = False


def _handle_signal(signum, frame):
    global _STOP
    _STOP = True


def _read_current_job() -> dict:
    if CURRENT_JOB_PATH.exists():
        try:
            with open(CURRENT_JOB_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _gpu_stats() -> dict:
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode()
        pynvml.nvmlShutdown()
        return {
            "gpu_name": name,
            "gpu_temp_c": temp,
            "gpu_power_w": round(power_mw / 1000.0, 2),
            "gpu_util_pct": util.gpu,
            "gpu_mem_used_mb": round(mem.used / (1024 ** 2), 1),
            "gpu_mem_total_mb": round(mem.total / (1024 ** 2), 1),
        }
    except Exception as e:
        return {
            "gpu_name": None, "gpu_temp_c": None, "gpu_power_w": None,
            "gpu_util_pct": None, "gpu_mem_used_mb": None, "gpu_mem_total_mb": None,
            "gpu_error": str(e),
        }


def _sample() -> dict:
    job = _read_current_job()
    disk = psutil.disk_usage(str(LOGS_MAQUINA_DIR.parent))
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_pct": psutil.cpu_percent(interval=None),
        "ram_used_gb": round(psutil.virtual_memory().used / (1024 ** 3), 2),
        "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        "disk_free_gb": round(disk.free / (1024 ** 3), 2),
        "current_experiment": job.get("experiment"),
        "current_repetition": job.get("repetition"),
        "current_seed": job.get("seed"),
    }
    row.update(_gpu_stats())
    return row


FIELDNAMES = [
    "timestamp_utc", "current_experiment", "current_repetition", "current_seed",
    "cpu_pct", "ram_used_gb", "ram_total_gb", "disk_free_gb",
    "gpu_name", "gpu_temp_c", "gpu_power_w", "gpu_util_pct",
    "gpu_mem_used_mb", "gpu_mem_total_mb", "gpu_error",
]


def _csv_path_for_today() -> Path:
    LOGS_MAQUINA_DIR.mkdir(parents=True, exist_ok=True)
    day = datetime.now(timezone.utc).strftime("%Y%m%d")
    return LOGS_MAQUINA_DIR / f"system_monitor_{day}.csv"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=10.0, help="segundos entre amostras")
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)

    print(f"[system_monitor] iniciado, amostrando a cada {args.interval}s. "
          f"Log em {LOGS_MAQUINA_DIR}/system_monitor_YYYYMMDD.csv")

    psutil.cpu_percent(interval=None)  # primeira chamada sempre retorna 0.0, descarta

    current_path = None
    fh = None
    writer = None

    try:
        while not _STOP:
            path = _csv_path_for_today()
            if path != current_path:
                if fh:
                    fh.close()
                is_new = not path.exists()
                fh = open(path, "a", newline="", encoding="utf-8")
                writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
                if is_new:
                    writer.writeheader()
                current_path = path

            row = _sample()
            writer.writerow(row)
            fh.flush()
            time.sleep(args.interval)
    finally:
        if fh:
            fh.close()
        print("[system_monitor] encerrado.")


if __name__ == "__main__":
    main()
