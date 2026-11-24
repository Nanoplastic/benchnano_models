"""Fila das repetições controladas / Controlled repetitions queue.

PT: Orquestra a fila de repetições controladas: 6 experimentos, 10 repetições cada, na ordem
    YOLO26L principal, SmolVLM de regime de tamanho (LoRA, completo), SmolVLM de polímero
    (só real, real + sintético, VLM maior) e, por fim, a avaliação fim a fim pareada. As
    reavaliações com max_det=2000 e com a regra "eq1" usam o mesmo mecanismo de fila.
    Cada repetição roda num PROCESSO PYTHON NOVO (subprocess), não num laço no mesmo
    processo, para evitar fragmentação da memória da GPU entre treinos longos (como nas
    dobras de 44_train_cv_fold.py).
    É retomável: o progresso fica em experimentos_controlados/.queue_state.json (via
    ctrl_common); se a máquina cair ou o processo for interrompido, rodar de novo pula o que
    já terminou.
    Inicia e encerra o monitor da máquina (system_monitor.py) automaticamente e o reinicia
    se ele cair no meio da fila.
    Uso:
    python3 run_queue.py -- fila completa (~60 h)
    python3 run_queue.py --dry-run -- só a semente 0 de cada experimento, para validar tudo
    antes de comprometer ~60 h de GPU
    python3 run_queue.py --from 03_vlm_polymer_real_synth -- retoma a partir de um
    experimento (além da retomada automática por repetição via queue_state)
EN: Orchestrates the controlled repetitions queue: 6 experiments, 10 repetitions each, in
    the order main YOLO26L, size regime SmolVLM (LoRA, full), polymer SmolVLM (real only,
    real + synthetic, larger VLM) and finally the paired end-to-end evaluation. The
    re-evaluations with max_det=2000 and the "eq1" rule use the same queue mechanism.
    Each repetition runs in a NEW PYTHON PROCESS (subprocess), not in a loop inside one
    process, to avoid GPU memory fragmentation between long training runs (as with the folds
    in 44_train_cv_fold.py).
    It is resumable: progress is kept in experimentos_controlados/.queue_state.json (via
    ctrl_common); if the machine goes down or the process is interrupted, running it again
    skips what is already done.
    Starts and stops the machine monitor (system_monitor.py) automatically and restarts it
    if it dies mid-queue.
    Usage:
    python3 run_queue.py -- full queue (~60 h)
    python3 run_queue.py --dry-run -- seed 0 of each experiment only, to validate everything
    before committing ~60 h of GPU
    python3 run_queue.py --from 03_vlm_polymer_real_synth -- resumes from a given experiment
    (in addition to the automatic per-repetition resume via queue_state)
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

DIR_SCRIPTS = Path(__file__).resolve().parent

# PT: ordem da fila: (script, nome_do_experimento)
# EN: queue order: (script, experiment_name)
FILA = [
    ("01_yolo26_main_10x.py", "01_yolo26_main"),
    ("05_smolvlm_size_lora_10x.py", "05_smolvlm_size_lora"),
    ("06_smolvlm_size_full_10x.py", "06_smolvlm_size_full"),
    ("02_vlm_polymer_real_only_10x.py", "02_vlm_polymer_real_only"),
    ("03_vlm_polymer_real_synth_10x.py", "03_vlm_polymer_real_synth"),
    ("04_vlm_polymer_bigvlm_10x.py", "04_vlm_polymer_bigvlm"),
]

_proc_monitor = None
_DEVE_PARAR = False


def _tratar_sinal(num_sinal, quadro):
    global _DEVE_PARAR
    print(f"\n[run_queue] sinal {num_sinal} recebido -- terminando após a repetição atual...")
    _DEVE_PARAR = True


def _iniciar_monitor():
    global _proc_monitor
    _proc_monitor = subprocess.Popen(
        [sys.executable, str(DIR_SCRIPTS / "system_monitor.py"), "--interval", "10"]
    )
    print(f"[run_queue] system_monitor.py iniciado (pid={_proc_monitor.pid})")


def _garantir_monitor_ativo():
    global _proc_monitor
    if _proc_monitor is None or _proc_monitor.poll() is not None:
        print("[run_queue] system_monitor.py não está rodando -- reiniciando.")
        _iniciar_monitor()


def _parar_monitor():
    global _proc_monitor
    if _proc_monitor and _proc_monitor.poll() is None:
        _proc_monitor.terminate()
        try:
            _proc_monitor.wait(timeout=15)
        except subprocess.TimeoutExpired:
            _proc_monitor.kill()
    print("[run_queue] system_monitor.py encerrado.")


def executar_rep_subprocesso(script: str, experimento: str, semente: int):
    """PT: Roda uma repetição como subprocesso, gravando a saída (stdout + stderr) em
        logs_maquina/exec_logs/<experimento>/rep_NN.log e exibindo-a em tempo real no
        terminal (tee manual via Popen). Registra início/fim/duração, máquina e uso de
        recursos em logs_maquina/execution_log.jsonl (uma linha por repetição, ver
        ctrl_common.registrar_execucao), usado por final_report.py nas estatísticas de tempo
        de execução.
    EN: Runs one repetition as a subprocess, writing its output (stdout + stderr) to
        logs_maquina/exec_logs/<experiment>/rep_NN.log while showing it live in the terminal
        (manual tee via Popen). Records start/end/duration, machine and resource usage in
        logs_maquina/execution_log.jsonl (one line per repetition, see
        ctrl_common.registrar_execucao), used by final_report.py for execution-time
        statistics.
    """
    comando = [sys.executable, str(DIR_SCRIPTS / script), "--only-seed", str(semente)]
    print(f"[run_queue] $ {' '.join(comando)}")

    caminho_log = cc.DIR_STDOUT_EXECUCAO / experimento / f"rep_{semente:02d}.log"
    caminho_log.parent.mkdir(parents=True, exist_ok=True)

    dt_inicio = datetime.now(timezone.utc)
    with open(caminho_log, "w", encoding="utf-8") as arq_log:
        arq_log.write(f"# comando: {' '.join(comando)}\n# inicio (UTC): {dt_inicio.isoformat()}\n\n")
        arq_log.flush()
        proc = subprocess.Popen(comando, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, bufsize=1)
        for linha in proc.stdout:
            sys.stdout.write(linha)
            sys.stdout.flush()
            arq_log.write(linha)
        proc.wait()
    dt_fim = datetime.now(timezone.utc)
    duracao_seg = (dt_fim - dt_inicio).total_seconds()

    caminho_resumo = cc.RESULTADOS_DIR / experimento / f"rep_{semente:02d}" / "summary.json"
    registro = {
        "experiment": experimento,
        "repetition_index": semente,
        "seed": semente,
        "script": script,
        "command": comando,
        "start_utc": dt_inicio.isoformat(),
        "end_utc": dt_fim.isoformat(),
        "duration_sec": round(duracao_seg, 1),
        "duration_human": cc.duracao_legivel(duracao_seg),
        "exit_code": proc.returncode,
        "status": "ok" if proc.returncode == 0 else "failed",
        "environment": cc.coletar_ambiente(),
        "resource_usage": cc.resumir_janela_recursos(dt_inicio, dt_fim),
        "stdout_log_path": str(caminho_log.relative_to(cc.RAIZ_CTRL)),
        "summary_json_path": (str(caminho_resumo.relative_to(cc.RAIZ_CTRL))
                               if caminho_resumo.exists() else None),
    }
    cc.registrar_execucao(registro)
    print(f"[run_queue] {experimento} rep {semente:02d}: {registro['status']} em "
          f"{registro['duration_human']} (exit={proc.returncode}) -- log em {caminho_log}")

    if proc.returncode != 0:
        raise RuntimeError(
            f"{script} --only-seed {semente} terminou com código {proc.returncode}. "
            f"Parando a fila -- as repetições já concluídas ficam marcadas em "
            f".queue_state.json, rode run_queue.py de novo depois de investigar. "
            f"Log completo em {caminho_log}"
        )


def _gravar_registro_sessao_fila(dt_inicio, status: str, reps_executadas: int):
    dt_fim = datetime.now(timezone.utc)
    duracao_seg = (dt_fim - dt_inicio).total_seconds()
    cc.anexar_jsonl(cc.CAMINHO_SESSOES_FILA, {
        "host": socket.gethostname(),
        "start_utc": dt_inicio.isoformat(),
        "end_utc": dt_fim.isoformat(),
        "duration_sec": round(duracao_seg, 1),
        "duration_human": cc.duracao_legivel(duracao_seg),
        "status": status,
        "reps_run_nesta_sessao": reps_executadas,
    })


def principal():
    analisador = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    analisador.add_argument("--dry-run", action="store_true",
                         help="roda só a repetição seed=0 de cada experimento")
    analisador.add_argument("--from", dest="from_experiment", default=None,
                         help="pula experimentos antes deste na ordem da fila")
    analisador.add_argument("--skip-monitor", action="store_true",
                         help="não inicia o system_monitor.py (só para depuração)")
    args = analisador.parse_args()

    signal.signal(signal.SIGINT, _tratar_sinal)
    signal.signal(signal.SIGTERM, _tratar_sinal)

    cc.RAIZ_CTRL.mkdir(parents=True, exist_ok=True)
    caminho_manifesto = cc.gravar_manifesto_sessao()
    print(f"[run_queue] manifesto de ambiente salvo em {caminho_manifesto}")

    if not args.skip_monitor:
        _iniciar_monitor()

    fila = FILA
    if args.from_experiment:
        nomes = [n for _, n in FILA]
        if args.from_experiment not in nomes:
            print(f"[run_queue] ERRO: {args.from_experiment!r} não é um experimento válido. Opções: {nomes}")
            sys.exit(1)
        indice = nomes.index(args.from_experiment)
        fila = FILA[indice:]

    sementes = [0] if args.dry_run else cc.SEMENTES
    if args.dry_run:
        print("[run_queue] --dry-run: só seed=0 de cada experimento.")

    estado = cc.carregar_estado_fila()
    inicio_sessao = datetime.now(timezone.utc)
    reps_executadas_sessao = 0
    status_sessao = "completo"

    try:
        for script, experimento in fila:
            print(f"\n{'=' * 70}\n[run_queue] EXPERIMENTO: {experimento} ({script})\n{'=' * 70}")
            for semente in sementes:
                if _DEVE_PARAR:
                    status_sessao = "interrompido_pelo_usuario"
                    print("[run_queue] parando por pedido do usuário (sinal recebido).")
                    return
                if cc.rep_concluida(estado, experimento, semente):
                    print(f"[run_queue] {experimento} rep {semente:02d} já concluída, pulando.")
                    continue
                if not args.skip_monitor:
                    _garantir_monitor_ativo()
                executar_rep_subprocesso(script, experimento, semente)
                reps_executadas_sessao += 1
                estado = cc.carregar_estado_fila()  # PT: recarrega (o subprocesso já marcou a repetição como feita) / EN: reloads (the subprocess already marked the repetition as done)

        if not args.dry_run:
            print(f"\n{'=' * 70}\n[run_queue] AVALIAÇÃO FIM-A-FIM PAREADA (07_end_to_end_paired)\n{'=' * 70}")
            resultado = subprocess.run([sys.executable, str(DIR_SCRIPTS / "eval_and_pair.py")])
            if resultado.returncode != 0:
                raise RuntimeError("eval_and_pair.py falhou -- ver saída acima.")

            print(f"\n{'=' * 70}\n[run_queue] RELATÓRIO FINAL\n{'=' * 70}")
            resultado = subprocess.run([sys.executable, str(DIR_SCRIPTS / "final_report.py")])
            if resultado.returncode != 0:
                raise RuntimeError("final_report.py falhou -- ver saída acima.")

        print("\n[run_queue] fila concluída." if not args.dry_run else "\n[run_queue] dry-run concluído com sucesso.")
    except BaseException as e:
        status_sessao = "interrompido_pelo_usuario" if isinstance(e, KeyboardInterrupt) else f"erro: {type(e).__name__}: {e}"
        raise
    finally:
        if not args.skip_monitor:
            _parar_monitor()
        cc.limpar_tarefa_atual()
        _gravar_registro_sessao_fila(inicio_sessao, status_sessao, reps_executadas_sessao)


if __name__ == "__main__":
    principal()
