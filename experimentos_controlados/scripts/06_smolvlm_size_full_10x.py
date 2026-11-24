"""10 repetições controladas do ajuste completo de regime de tamanho / 10 controlled size regime full fine-tuning repetitions.

PT: 10 repetições controladas do ajuste fino completo (todos os parâmetros, sem LoRA) do
    SmolVLM na classificação de regime de tamanho, espelhando
    experimentos/scripts/18_size_finetune_full.py (mesmo manifesto, condição "caixa de
    referência"). Para a variante de 500M, a documentação do SmolVLM2 recomenda ajuste fino
    completo em vez de LoRA/QLoRA.
    Usa bem mais VRAM que o LoRA (todos os ~500M parâmetros recebem gradiente e estado do
    AdamW); mesma ressalva do script 18 (RTX 4060 Ti, 16 GB).
    Rodar (treino longo em GPU; acompanhar no terminal):
    python3 06_smolvlm_size_full_10x.py [--only-seed N]
EN: 10 controlled repetitions of the full fine-tuning (all parameters, no LoRA) of SmolVLM
    on size regime classification, mirroring experimentos/scripts/18_size_finetune_full.py
    (same manifest, "reference box" condition). For the 500M variant, the SmolVLM2
    documentation recommends full fine-tuning instead of LoRA/QLoRA.
    Uses much more VRAM than LoRA (all ~500M parameters get gradients and AdamW state); same
    caveat as script 18 (RTX 4060 Ti, 16 GB).
    Run (long GPU job; follow it in the terminal):
    python3 06_smolvlm_size_full_10x.py [--only-seed N]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctrl_common as cc

sys.path.insert(0, str(cc.SCRIPTS_EXPERIMENTOS))
from vlm_common import DIR_CHECKPOINTS, RAIZ_RESULTADOS, PROMPT_CLASSIFICACAO_TAMANHO, ROTULOS_TAMANHO, ajuste_fino_completo

EXPERIMENTO = "06_smolvlm_size_full"
ROTULO_BASE = "size_full_ft"
MANIFESTO = RAIZ_RESULTADOS / "15_build_size_regime_manifest" / "data" / "manifest_size_regime.csv"


def executar_repeticao(semente: int, estado: dict):
    if cc.rep_concluida(estado, EXPERIMENTO, semente):
        print(f"[{EXPERIMENTO}] rep {semente:02d} já concluída, pulando.")
        return

    if not MANIFESTO.exists():
        raise FileNotFoundError(f"{MANIFESTO} não existe -- rode 15_build_size_regime_manifest.py primeiro.")

    rotulo_exp = f"{ROTULO_BASE}_ctrl_seed{semente}"
    print(f"\n=== {EXPERIMENTO} — repetição {semente:02d}/9 (seed={semente}, tag={rotulo_exp}) ===")
    cc.definir_tarefa_atual(EXPERIMENTO, semente)
    cc.fixar_sementes(semente)

    df = pd.read_csv(MANIFESTO)
    df_treino = df[df["split"] == "train"].reset_index(drop=True)
    df_val = df[df["split"] == "val"].reset_index(drop=True)
    df_teste = df[df["split"] == "test"].reset_index(drop=True)

    resultado = ajuste_fino_completo(df_treino, df_val, df_teste, rotulo_exp=rotulo_exp, semente=semente,
                            prompt=PROMPT_CLASSIFICACAO_TAMANHO, rotulos=ROTULOS_TAMANHO)

    metricas = {"best_val_acc": float(resultado["best_val_acc"]), "test_acc": float(resultado["test_acc"])}
    print(f"[{EXPERIMENTO}] rep {semente:02d}: test_acc={metricas['test_acc']:.3f}")
    cc.salvar_resumo_rep(EXPERIMENTO, semente, semente, metricas, metadados_extras={
        "tag": rotulo_exp,
        "checkpoint_dir": str(DIR_CHECKPOINTS / rotulo_exp),
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
