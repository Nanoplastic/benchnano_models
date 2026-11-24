"""Ajuste fino completo para regime de tamanho / Full fine-tuning for size regime.

PT: Ajuste fino de todos os parâmetros (sem LoRA) do SmolVLM-500M-Instruct na classificação
    de regime de tamanho (NANOPLASTIC/MICROPLASTIC). Para a variante de 500M, a documentação
    do SmolVLM2 recomenda ajuste fino completo em vez de LoRA/QLoRA.
    Usa bem mais VRAM que o LoRA (todos os ~500M parâmetros recebem gradiente e estado do
    AdamW). Se faltar memória, reduzir acum_grad não ajuda (o lote já é 1); as alternativas
    são otimizador de 8 bits (bitsandbytes) ou LoRA com posto maior, não implementadas aqui.
EN: Full-parameter fine-tuning (no LoRA) of SmolVLM-500M-Instruct on size regime
    classification (NANOPLASTIC/MICROPLASTIC). For the 500M variant, the SmolVLM2
    documentation recommends full fine-tuning instead of LoRA/QLoRA.
    Uses much more VRAM than LoRA (all ~500M parameters get gradients and AdamW state). If
    memory runs out, lowering acum_grad does not help (the batch is already 1); the
    alternatives are an 8-bit optimizer (bitsandbytes) or a higher-rank LoRA, not
    implemented here.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import (
    RAIZ_RESULTADOS, PROMPT_CLASSIFICACAO_TAMANHO, ROTULOS_TAMANHO, ajuste_fino_completo,
    salvar_resumo_experimento, configurar_log_experimento,
)

ROTULO_EXP = "size_full_ft"
NOME_EXP = "18_size_finetune_full"
DIR_EXP, CAMINHO_LOG, REGISTRADOR, METADADOS_EXECUCAO = configurar_log_experimento(NOME_EXP, __file__, {"tag": ROTULO_EXP})

MANIFESTO = RAIZ_RESULTADOS / "15_build_size_regime_manifest" / "data" / "manifest_size_regime.csv"


def principal():
    if not MANIFESTO.exists():
        raise FileNotFoundError(f"{MANIFESTO} não existe -- rode 15_build_size_regime_manifest.py primeiro.")

    df = pd.read_csv(MANIFESTO)
    df_treino = df[df["split"] == "train"].reset_index(drop=True)
    df_val = df[df["split"] == "val"].reset_index(drop=True)
    df_teste = df[df["split"] == "test"].reset_index(drop=True)

    print(f"Full fine-tuning regime de tamanho: train={len(df_treino)}  val={len(df_val)}  test={len(df_teste)}")
    print(df_treino["label"].value_counts())
    REGISTRADOR.info("train=%s val=%s test=%s classes=%s", len(df_treino), len(df_val), len(df_teste),
                df_treino["label"].value_counts().to_dict())

    resultado = ajuste_fino_completo(df_treino, df_val, df_teste, rotulo_exp=ROTULO_EXP,
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
    REGISTRADOR.info("Resumo full fine-tuning regime de tamanho: best_val_acc=%.4f test_acc=%.4f",
                resultado["best_val_acc"], resultado["test_acc"])


if __name__ == "__main__":
    principal()
