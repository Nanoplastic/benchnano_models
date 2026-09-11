"""Orquestrador da fila de repetições controladas -- roda os 6 experimentos
citados no artigo, 10 repetições cada, na ordem: YOLO26L principal, SmolVLM
regime de tamanho (LoRA, full), SmolVLM polímero (real-only, real+sintético,
bigvlm), e por fim a avaliação fim-a-fim pareada. Ver plano completo em
/home/mi/.claude/plans/parallel-giggling-ocean.md.

Cada repetição roda como um PROCESSO PYTHON NOVO (subprocess), não dentro
de um loop no mesmo processo -- mesma prática já usada no projeto original
pros folds de CV (44_train_cv_fold.py: "rodar uma vez por fold,
sequencialmente"), pra evitar fragmentação de memória de GPU entre
repetições de treino longas.

É resumível: mantém progresso em experimentos_controlados/.queue_state.json
(via ctrl_common) -- se a máquina cair ou o processo for interrompido,
rodar de novo pula tudo que já terminou.

Inicia e encerra o logger de máquina (system_monitor.py) automaticamente, e
o reinicia se ele cair no meio da fila.

Uso:
    python3 run_queue.py                # fila completa (~60h)
    python3 run_queue.py --dry-run       # só seed=0 de cada experimento,
                                          # pra validar que tudo roda antes
                                          # de comprometer ~60h de GPU
    python3 run_queue.py --from 03_vlm_polymer_real_synth
                                          # retoma a partir de um experimento
                                          # específico (além do resume automático
                                          # por repetição já feito via queue_state)
"""
import argparse
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctrl_common as cc

SCRIPTS_DIR = Path(__file__).resolve().parent

# ordem da fila: (script, nome_do_experimento)
QUEUE = [
    ("01_yolo26_main_10x.py", "01_yolo26_main"),
    ("05_smolvlm_size_lora_10x.py", "05_smolvlm_size_lora"),
    ("06_smolvlm_size_full_10x.py", "06_smolvlm_size_full"),
    ("02_vlm_polymer_real_only_10x.py", "02_vlm_polymer_real_only"),
    ("03_vlm_polymer_real_synth_10x.py", "03_vlm_polymer_real_synth"),
    ("04_vlm_polymer_bigvlm_10x.py", "04_vlm_polymer_bigvlm"),
]

_monitor_proc = None
_SHOULD_STOP = False


def _handle_signal(signum, frame):
    global _SHOULD_STOP
    print(f"\n[run_queue] sinal {signum} recebido -- terminando após a repetição atual...")
    _SHOULD_STOP = True


def _start_monitor():
    global _monitor_proc
    _monitor_proc = subprocess.Popen(
        [sys.executable, str(SCRIPTS_DIR / "system_monitor.py"), "--interval", "10"]
    )
    print(f"[run_queue] system_monitor.py iniciado (pid={_monitor_proc.pid})")


def _ensure_monitor_alive():
    global _monitor_proc
    if _monitor_proc is None or _monitor_proc.poll() is not None:
        print("[run_queue] system_monitor.py não está rodando -- reiniciando.")
        _start_monitor()


def _stop_monitor():
    global _monitor_proc
    if _monitor_proc and _monitor_proc.poll() is None:
        _monitor_proc.terminate()
        try:
            _monitor_proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _monitor_proc.kill()
    print("[run_queue] system_monitor.py encerrado.")


def run_rep_subprocess(script: str, experiment: str, seed: int):
    """Roda uma repetição como subprocesso, gravando a saída (stdout+stderr)
    em logs_maquina/exec_logs/<experimento>/rep_NN.log -- sem deixar de
    exibir em tempo real no terminal (tee manual via Popen) -- e registra
    tempo de início/fim/duração, máquina e uso de recursos em
    logs_maquina/execution_log.jsonl (uma linha por repetição, ver
    ctrl_common.log_execution). É esse log que alimenta as estatísticas de
    tempo de execução do artigo (final_report.py)."""
    cmd = [sys.executable, str(SCRIPTS_DIR / script), "--only-seed", str(seed)]
    print(f"[run_queue] $ {' '.join(cmd)}")

    log_path = cc.EXEC_STDOUT_DIR / experiment / f"rep_{seed:02d}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    start_dt = datetime.now(timezone.utc)
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"# comando: {' '.join(cmd)}\n# inicio (UTC): {start_dt.isoformat()}\n\n")
        logf.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1)
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
        proc.wait()
    end_dt = datetime.now(timezone.utc)
    duration_sec = (end_dt - start_dt).total_seconds()

    summary_path = cc.RESULTADOS_DIR / experiment / f"rep_{seed:02d}" / "summary.json"
    record = {
        "experiment": experiment,
        "repetition_index": seed,
        "seed": seed,
        "script": script,
        "command": cmd,
        "start_utc": start_dt.isoformat(),
        "end_utc": end_dt.isoformat(),
        "duration_sec": round(duration_sec, 1),
        "duration_human": cc.human_duration(duration_sec),
        "exit_code": proc.returncode,
        "status": "ok" if proc.returncode == 0 else "failed",
        "environment": cc.collect_environment(),
        "resource_usage": cc.summarize_resource_window(start_dt, end_dt),
        "stdout_log_path": str(log_path.relative_to(cc.CTRL_ROOT)),
        "summary_json_path": (str(summary_path.relative_to(cc.CTRL_ROOT))
                               if summary_path.exists() else None),
    }
    cc.log_execution(record)
    print(f"[run_queue] {experiment} rep {seed:02d}: {record['status']} em "
          f"{record['duration_human']} (exit={proc.returncode}) -- log em {log_path}")

    if proc.returncode != 0:
        raise RuntimeError(
            f"{script} --only-seed {seed} terminou com código {proc.returncode}. "
            f"Parando a fila -- as repetições já concluídas ficam marcadas em "
            f".queue_state.json, rode run_queue.py de novo depois de investigar. "
            f"Log completo em {log_path}"
        )


def _write_queue_session_record(start_dt, status: str, reps_run: int):
    end_dt = datetime.now(timezone.utc)
    duration_sec = (end_dt - start_dt).total_seconds()
    cc.append_jsonl(cc.QUEUE_SESSIONS_PATH, {
        "host": socket.gethostname(),
        "start_utc": start_dt.isoformat(),
        "end_utc": end_dt.isoformat(),
        "duration_sec": round(duration_sec, 1),
        "duration_human": cc.human_duration(duration_sec),
        "status": status,
        "reps_run_nesta_sessao": reps_run,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                         help="roda só a repetição seed=0 de cada experimento")
    parser.add_argument("--from", dest="from_experiment", default=None,
                         help="pula experimentos antes deste na ordem da fila")
    parser.add_argument("--skip-monitor", action="store_true",
                         help="não inicia o system_monitor.py (só para depuração)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    cc.CTRL_ROOT.mkdir(parents=True, exist_ok=True)
    manifest_path = cc.write_session_manifest()
    print(f"[run_queue] manifesto de ambiente salvo em {manifest_path}")

    if not args.skip_monitor:
        _start_monitor()

    queue = QUEUE
    if args.from_experiment:
        names = [n for _, n in QUEUE]
        if args.from_experiment not in names:
            print(f"[run_queue] ERRO: {args.from_experiment!r} não é um experimento válido. Opções: {names}")
            sys.exit(1)
        idx = names.index(args.from_experiment)
        queue = QUEUE[idx:]

    seeds = [0] if args.dry_run else cc.SEEDS
    if args.dry_run:
        print("[run_queue] --dry-run: só seed=0 de cada experimento.")

    state = cc.load_queue_state()
    session_start = datetime.now(timezone.utc)
    reps_run_session = 0
    session_status = "completo"

    try:
        for script, experiment in queue:
            print(f"\n{'=' * 70}\n[run_queue] EXPERIMENTO: {experiment} ({script})\n{'=' * 70}")
            for seed in seeds:
                if _SHOULD_STOP:
                    session_status = "interrompido_pelo_usuario"
                    print("[run_queue] parando por pedido do usuário (sinal recebido).")
                    return
                if cc.is_rep_done(state, experiment, seed):
                    print(f"[run_queue] {experiment} rep {seed:02d} já concluída, pulando.")
                    continue
                if not args.skip_monitor:
                    _ensure_monitor_alive()
                run_rep_subprocess(script, experiment, seed)
                reps_run_session += 1
                state = cc.load_queue_state()  # recarrega (o subprocesso já marcou como feito)

        if not args.dry_run:
            print(f"\n{'=' * 70}\n[run_queue] AVALIAÇÃO FIM-A-FIM PAREADA (07_end_to_end_paired)\n{'=' * 70}")
            result = subprocess.run([sys.executable, str(SCRIPTS_DIR / "eval_and_pair.py")])
            if result.returncode != 0:
                raise RuntimeError("eval_and_pair.py falhou -- ver saída acima.")

            print(f"\n{'=' * 70}\n[run_queue] RELATÓRIO FINAL\n{'=' * 70}")
            result = subprocess.run([sys.executable, str(SCRIPTS_DIR / "final_report.py")])
            if result.returncode != 0:
                raise RuntimeError("final_report.py falhou -- ver saída acima.")

        print("\n[run_queue] fila concluída." if not args.dry_run else "\n[run_queue] dry-run concluído com sucesso.")
    except BaseException as e:
        session_status = "interrompido_pelo_usuario" if isinstance(e, KeyboardInterrupt) else f"erro: {type(e).__name__}: {e}"
        raise
    finally:
        if not args.skip_monitor:
            _stop_monitor()
        cc.clear_current_job()
        _write_queue_session_record(session_start, session_status, reps_run_session)


if __name__ == "__main__":
    main()
