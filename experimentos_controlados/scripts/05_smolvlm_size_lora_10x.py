"""10 repetições controladas do LoRA de regime de tamanho / 10 controlled size regime LoRA repetitions.

PT: 10 repetições controladas do ajuste fino LoRA do SmolVLM na classificação de regime de
    tamanho (NANOPLASTIC/MICROPLASTIC), espelhando
    experimentos/scripts/17_size_finetune_lora.py (r=8, alfa=8, mesmo manifesto de regime de
    tamanho, condição "caixa de referência").
    Depois que as 10 repetições deste script e as 10 de 01_yolo26_main_10x.py terminarem,
    rode eval_and_pair.py para a avaliação fim a fim pareada (repetição i do YOLO +
    repetição i deste script), condição "caixa prevista" / fim a fim estrito; ver
    07_end_to_end_paired em resultados/.
    Rodar (treino longo em GPU; acompanhar no terminal):
    python3 05_smolvlm_size_lora_10x.py [--only-seed N]
EN: 10 controlled repetitions of the SmolVLM LoRA fine-tuning on size regime classification
    (NANOPLASTIC/MICROPLASTIC), mirroring experimentos/scripts/17_size_finetune_lora.py
    (r=8, alpha=8, same size regime manifest, "reference box" condition).
    Once the 10 repetitions of this script and the 10 of 01_yolo26_main_10x.py are done, run
    eval_and_pair.py for the paired end-to-end evaluation (YOLO repetition i + repetition i
    of this script), "predicted box" / strict end-to-end condition; see 07_end_to_end_paired
    under resultados/.
    Run (long GPU job; follow it in the terminal):
    python3 05_smolvlm_size_lora_10x.py [--only-seed N]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctrl_common as cc

sys.path.insert(0, str(cc.SCRIPTS_EXPERIMENTOS))
from vlm_common import (
    DIR_CHECKPOINTS, RAIZ_RESULTADOS, PROMPT_CLASSIFICACAO_TAMANHO, ROTULOS_TAMANHO, ajuste_fino_lora,
)

EXPERIMENTO = "05_smolvlm_size_lora"
ROTULO_BASE = "size_lora"
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

    resultado = ajuste_fino_lora(df_treino, df_val, df_teste, rotulo_exp=rotulo_exp, semente=semente,
                            prompt=PROMPT_CLASSIFICACAO_TAMANHO, rotulos=ROTULOS_TAMANHO)

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
