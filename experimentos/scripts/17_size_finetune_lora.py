"""Ajuste fino LoRA para regime de tamanho / LoRA fine-tuning for size regime.

PT: Ajuste fino LoRA do SmolVLM-500M-Instruct na classificação de regime de tamanho
    (NANOPLASTIC/MICROPLASTIC). Mesma receita (r=8, alfa=8, 6 épocas) do ajuste de polímero
    do script 05, trocando prompt e rótulos via vlm_common.ajuste_fino_lora(prompt=...,
    rotulos=...). Condição "caixa de referência".
    Comparado com zero-shot (16) e ajuste fino completo (18) em
    19_compare_size_regime_configs.py.
EN: LoRA fine-tuning of SmolVLM-500M-Instruct on size regime classification
    (NANOPLASTIC/MICROPLASTIC). Same recipe (r=8, alpha=8, 6 epochs) as the polymer run in
    script 05, changing prompt and labels via vlm_common.ajuste_fino_lora(prompt=...,
    rotulos=...). "Reference box" condition.
    Compared with zero-shot (16) and full fine-tuning (18) in
    19_compare_size_regime_configs.py.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import (
    RAIZ_RESULTADOS, PROMPT_CLASSIFICACAO_TAMANHO, ROTULOS_TAMANHO, ajuste_fino_lora,
    salvar_resumo_experimento, configurar_log_experimento,
)

ROTULO_EXP = "size_lora"
NOME_EXP = "17_size_finetune_lora"
DIR_EXP, CAMINHO_LOG, REGISTRADOR, METADADOS_EXECUCAO = configurar_log_experimento(NOME_EXP, __file__, {"tag": ROTULO_EXP})

MANIFESTO = RAIZ_RESULTADOS / "15_build_size_regime_manifest" / "data" / "manifest_size_regime.csv"


def principal():
    if not MANIFESTO.exists():
        raise FileNotFoundError(f"{MANIFESTO} não existe -- rode 15_build_size_regime_manifest.py primeiro.")

    df = pd.read_csv(MANIFESTO)
    df_treino = df[df["split"] == "train"].reset_index(drop=True)
    df_val = df[df["split"] == "val"].reset_index(drop=True)
    df_teste = df[df["split"] == "test"].reset_index(drop=True)

    print(f"LoRA regime de tamanho: train={len(df_treino)}  val={len(df_val)}  test={len(df_teste)}")
    print(df_treino["label"].value_counts())
    REGISTRADOR.info("train=%s val=%s test=%s classes=%s", len(df_treino), len(df_val), len(df_teste),
                df_treino["label"].value_counts().to_dict())

    resultado = ajuste_fino_lora(df_treino, df_val, df_teste, rotulo_exp=ROTULO_EXP,
                            prompt=PROMPT_CLASSIFICACAO_TAMANHO, rotulos=ROTULOS_TAMANHO)

    resumo = {
        "tag": ROTULO_EXP,
        "train_samples": len(df_treino),
        "val_samples": len(df_val),
        "test_samples": len(df_teste),
        "best_val_acc": float(resultado["best_val_acc"]),
        "test_acc": float(resultado["test_acc"]),
    }
    salvar_resumo_experimento(NOME_EXP, METADADOS_EXECUCAO, resumo)
    REGISTRADOR.info("Resumo LoRA regime de tamanho: best_val_acc=%.4f test_acc=%.4f",
                resultado["best_val_acc"], resultado["test_acc"])


if __name__ == "__main__":
    principal()
