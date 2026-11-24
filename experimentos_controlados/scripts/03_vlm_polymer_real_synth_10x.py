"""10 repetições controladas do LoRA real + sintético / 10 controlled real + synthetic LoRA repetitions.

PT: 10 repetições controladas do ajuste fino LoRA do SmolVLM-500M-Instruct com dados reais +
    sintéticos, espelhando experimentos/scripts/06_finetune_real_plus_synthetic.py (mesma
    receita do 05, com df_treino = treino real + sintético; val/teste só reais, para
    comparação direta com o 02).
    Ver 02_vlm_polymer_real_only_10x.py para o desenho completo (semente completa, rótulos
    "_ctrl_seedN", checkpoints/reports na pasta compartilhada).
    Rodar (treino longo em GPU; acompanhar no terminal):
    python3 03_vlm_polymer_real_synth_10x.py [--only-seed N]
EN: 10 controlled repetitions of the real + synthetic LoRA fine-tuning of
    SmolVLM-500M-Instruct, mirroring experimentos/scripts/06_finetune_real_plus_synthetic.py
    (same recipe as 05, with df_treino = real train + synthetic; val/test real only, for a
    direct comparison with 02).
    See 02_vlm_polymer_real_only_10x.py for the full design (full seed, "_ctrl_seedN" tags,
    checkpoints/reports in the shared folder).
    Run (long GPU job; follow it in the terminal):
    python3 03_vlm_polymer_real_synth_10x.py [--only-seed N]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctrl_common as cc

sys.path.insert(0, str(cc.SCRIPTS_EXPERIMENTOS))
from vlm_common import DIR_CHECKPOINTS, ajuste_fino_lora, carregar_particao_real, carregar_sintetico

EXPERIMENTO = "03_vlm_polymer_real_synth"
ROTULO_BASE = "mina_real_plus_synthetic_lora_v2"


def executar_repeticao(semente: int, estado: dict):
    if cc.rep_concluida(estado, EXPERIMENTO, semente):
        print(f"[{EXPERIMENTO}] rep {semente:02d} já concluída, pulando.")
        return

    rotulo_exp = f"{ROTULO_BASE}_ctrl_seed{semente}"
    print(f"\n=== {EXPERIMENTO} — repetição {semente:02d}/9 (seed={semente}, tag={rotulo_exp}) ===")
    cc.definir_tarefa_atual(EXPERIMENTO, semente)
    cc.fixar_sementes(semente)

    df_treino_real = carregar_particao_real("train")
    df_sintetico = carregar_sintetico()
    df_treino = pd.concat([df_treino_real, df_sintetico], ignore_index=True)
    df_val = carregar_particao_real("val")
    df_teste = carregar_particao_real("test")

    resultado = ajuste_fino_lora(df_treino, df_val, df_teste, rotulo_exp=rotulo_exp, semente=semente)

    metricas = {"best_val_acc": float(resultado["best_val_acc"]), "test_acc": float(resultado["test_acc"])}
    print(f"[{EXPERIMENTO}] rep {semente:02d}: test_acc={metricas['test_acc']:.3f}")
    cc.salvar_resumo_rep(EXPERIMENTO, semente, semente, metricas, metadados_extras={
        "tag": rotulo_exp,
        "adapter_dir": str(DIR_CHECKPOINTS / rotulo_exp),
        "real_train_samples": len(df_treino_real), "synthetic_samples": len(df_sintetico),
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
