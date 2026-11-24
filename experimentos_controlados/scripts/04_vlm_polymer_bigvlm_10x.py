"""10 repetições controladas do LoRA com VLM maior / 10 controlled larger-VLM LoRA repetitions.

PT: 10 repetições controladas do ajuste fino LoRA do SmolVLM-Instruct (~2B), espelhando
    experimentos/scripts/09_finetune_bigger_vlm.py (mesma receita do 05, trocando só o
    modelo base).
    A primeira repetição também mede a duração por repetição, usada para estimar o tempo
    total da fila.
    Ver 02_vlm_polymer_real_only_10x.py para o desenho completo.
    Rodar (treino longo em GPU; acompanhar no terminal):
    python3 04_vlm_polymer_bigvlm_10x.py [--only-seed N]
EN: 10 controlled repetitions of the LoRA fine-tuning of SmolVLM-Instruct (~2B), mirroring
    experimentos/scripts/09_finetune_bigger_vlm.py (same recipe as 05, changing only the
    base model).
    The first repetition also measures the per-repetition duration, used to estimate the
    total queue time.
    See 02_vlm_polymer_real_only_10x.py for the full design.
    Run (long GPU job; follow it in the terminal):
    python3 04_vlm_polymer_bigvlm_10x.py [--only-seed N]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctrl_common as cc

sys.path.insert(0, str(cc.SCRIPTS_EXPERIMENTOS))
from vlm_common import DIR_CHECKPOINTS, ajuste_fino_lora, carregar_particao_real

EXPERIMENTO = "04_vlm_polymer_bigvlm"
ROTULO_BASE = "mina_real_only_lora_bigvlm"
ID_MODELO = "HuggingFaceTB/SmolVLM-Instruct"


def executar_repeticao(semente: int, estado: dict):
    if cc.rep_concluida(estado, EXPERIMENTO, semente):
        print(f"[{EXPERIMENTO}] rep {semente:02d} já concluída, pulando.")
        return

    rotulo_exp = f"{ROTULO_BASE}_ctrl_seed{semente}"
    print(f"\n=== {EXPERIMENTO} — repetição {semente:02d}/9 (seed={semente}, tag={rotulo_exp}) ===")
    cc.definir_tarefa_atual(EXPERIMENTO, semente)
    cc.fixar_sementes(semente)

    df_treino = carregar_particao_real("train")
    df_val = carregar_particao_real("val")
    df_teste = carregar_particao_real("test")

    t0 = time.monotonic()
    resultado = ajuste_fino_lora(df_treino, df_val, df_teste, rotulo_exp=rotulo_exp, semente=semente, id_modelo=ID_MODELO)
    decorrido_min = (time.monotonic() - t0) / 60.0

    metricas = {"best_val_acc": float(resultado["best_val_acc"]), "test_acc": float(resultado["test_acc"])}
    print(f"[{EXPERIMENTO}] rep {semente:02d}: test_acc={metricas['test_acc']:.3f} ({decorrido_min:.1f} min)")
    cc.salvar_resumo_rep(EXPERIMENTO, semente, semente, metricas, metadados_extras={
        "tag": rotulo_exp,
        "adapter_dir": str(DIR_CHECKPOINTS / rotulo_exp),
        "model_id": ID_MODELO,
        "elapsed_minutes": round(decorrido_min, 1),
        "train_samples": len(df_treino), "val_samples": len(df_val), "test_samples": len(df_teste),
    })
    cc.marcar_rep_concluida(estado, EXPERIMENTO, semente)
    cc.limpar_tarefa_atual()


def principal():
    analisador = argparse.ArgumentParser(description=__doc__)
    analisador.add_argument("--only-seed", type=int, default=None)
    args = analisador.parse_args()

    estado = cc.carregar_estado_fila()
    sementes = [args.only_seed] if args.only_seed is not None else cc.SEMENTES
    for semente in sementes:
        executar_repeticao(semente, estado)


if __name__ == "__main__":
    principal()
