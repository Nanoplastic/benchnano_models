"""10 repetições controladas do LoRA só real / 10 controlled real-only LoRA repetitions.

PT: 10 repetições controladas do ajuste fino LoRA do SmolVLM-500M-Instruct só com dados
    reais, espelhando experimentos/scripts/05_finetune_real_only.py (r=8, alfa=8,
    dropout=0,1, 6 épocas, lr=1e-4), variando a semente completa (random/numpy/torch, via
    ctrl_common.fixar_sementes) com semente explícita 0..9.
    Usa vlm_common.ajuste_fino_lora sem modificação; adaptadores e CSVs de predição ficam na
    pasta compartilhada (experimentos/resultados/checkpoints/, reports/) com rótulos únicos
    "_ctrl_seedN" (mesma convenção de _run2/_run3/_run4), sem sobrescrever nada.
    Rodar (treino longo em GPU; acompanhar no terminal):
    python3 02_vlm_polymer_real_only_10x.py [--only-seed N]
EN: 10 controlled repetitions of the real-data-only LoRA fine-tuning of
    SmolVLM-500M-Instruct, mirroring experimentos/scripts/05_finetune_real_only.py (r=8,
    alpha=8, dropout=0.1, 6 epochs, lr=1e-4), varying the full seed (random/numpy/torch, via
    ctrl_common.fixar_sementes) with explicit seeds 0..9.
    Uses vlm_common.ajuste_fino_lora unchanged; adapters and prediction CSVs go to the
    shared folder (experimentos/resultados/checkpoints/, reports/) with unique "_ctrl_seedN"
    tags (same convention as _run2/_run3/_run4), never overwriting anything.
    Run (long GPU job; follow it in the terminal):
    python3 02_vlm_polymer_real_only_10x.py [--only-seed N]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctrl_common as cc

sys.path.insert(0, str(cc.SCRIPTS_EXPERIMENTOS))
from vlm_common import DIR_CHECKPOINTS, ajuste_fino_lora, carregar_particao_real

EXPERIMENTO = "02_vlm_polymer_real_only"
ROTULO_BASE = "mina_real_only_lora"


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

    resultado = ajuste_fino_lora(df_treino, df_val, df_teste, rotulo_exp=rotulo_exp, semente=semente)

    metricas = {"best_val_acc": float(resultado["best_val_acc"]), "test_acc": float(resultado["test_acc"])}
    print(f"[{EXPERIMENTO}] rep {semente:02d}: test_acc={metricas['test_acc']:.3f}")
    cc.salvar_resumo_rep(EXPERIMENTO, semente, semente, metricas, metadados_extras={
        "tag": rotulo_exp,
        "adapter_dir": str(DIR_CHECKPOINTS / rotulo_exp),
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
