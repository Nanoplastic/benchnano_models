"""Ajuste fino completo de tamanho pela regra eq1 / Size full fine-tuning with the eq1 rule.

PT: Mesmo treino de 18_size_finetune_full.py, trocando só o manifesto: usa
    manifest_size_regime_eq1.csv (regime pelo diâmetro equivalente da área do polígono COCO,
    script 57) em vez de manifest_size_regime.csv (regime pela caixa, regra eq2). Mesma
    justificativa de 80_size_finetune_lora_eq1_single.py (novo treino de execução única, não
    reclassificação).
    Não sobrescreve o checkpoint de 18 (rótulo size_full_ft_eq1_single em vez de
    size_full_ft).
    Uso:
    python3 experimentos/scripts/81_size_finetune_full_eq1_single.py
EN: Same training as 18_size_finetune_full.py, changing only the manifest: uses
    manifest_size_regime_eq1.csv (regime from the area-equivalent diameter of the COCO
    polygon, script 57) instead of manifest_size_regime.csv (regime from the box, eq2 rule).
    Same rationale as 80_size_finetune_lora_eq1_single.py (new single-run training, not
    relabelling).
    Does not overwrite the checkpoint of 18 (tag size_full_ft_eq1_single instead of
    size_full_ft).
    Usage:
    python3 experimentos/scripts/81_size_finetune_full_eq1_single.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import (
    RAIZ_RESULTADOS, PROMPT_CLASSIFICACAO_TAMANHO, ROTULOS_TAMANHO, ajuste_fino_completo,
    salvar_resumo_experimento, configurar_log_experimento,
)

ROTULO_EXP = "size_full_ft_eq1_single"
NOME_EXP = "81_size_finetune_full_eq1_single"
DIR_EXP, CAMINHO_LOG, REGISTRADOR, METADADOS_EXECUCAO = configurar_log_experimento(NOME_EXP, __file__, {"tag": ROTULO_EXP})

MANIFESTO = RAIZ_RESULTADOS / "57_build_size_regime_manifest_eq1" / "data" / "manifest_size_regime_eq1.csv"
TAMANHOS_PARTICAO_ESPERADOS = {"train": 1000, "val": 240, "test": 208}


def principal():
    if not MANIFESTO.exists():
        raise FileNotFoundError(f"{MANIFESTO} não existe -- rode 57_build_size_regime_manifest_eq1.py primeiro.")

    df = pd.read_csv(MANIFESTO)
    df_treino = df[df["split"] == "train"].reset_index(drop=True)
    df_val = df[df["split"] == "val"].reset_index(drop=True)
    df_teste = df[df["split"] == "test"].reset_index(drop=True)

    for nome, sub, esperado in [("train", df_treino, TAMANHOS_PARTICAO_ESPERADOS["train"]),
                                 ("val", df_val, TAMANHOS_PARTICAO_ESPERADOS["val"]),
                                 ("test", df_teste, TAMANHOS_PARTICAO_ESPERADOS["test"])]:
        if len(sub) != esperado:
            raise SystemExit(f"sanidade falhou: split '{nome}' tem {len(sub)} linhas, esperado {esperado} "
                              f"(mesma contagem da corrida Eq.2 original) -- abortando antes de treinar.")

    print(f"Full fine-tuning regime de tamanho (Eq.1, corrida unica): train={len(df_treino)}  "
          f"val={len(df_val)}  test={len(df_teste)}")
    print(df_treino["label"].value_counts())
    REGISTRADOR.info("train=%s val=%s test=%s classes=%s", len(df_treino), len(df_val), len(df_teste),
                df_treino["label"].value_counts().to_dict())

    resultado = ajuste_fino_completo(df_treino, df_val, df_teste, rotulo_exp=ROTULO_EXP,
                            prompt=PROMPT_CLASSIFICACAO_TAMANHO, rotulos=ROTULOS_TAMANHO)

    resumo = {
        "tag": ROTULO_EXP,
        "manifest": str(MANIFESTO),
        "train_samples": len(df_treino),
        "val_samples": len(df_val),
        "test_samples": len(df_teste),
        "best_val_acc": float(resultado["best_val_acc"]),
        "test_acc": float(resultado["test_acc"]),
    }
    salvar_resumo_experimento(NOME_EXP, METADADOS_EXECUCAO, resumo)
    REGISTRADOR.info("Resumo full fine-tuning regime de tamanho (Eq.1, corrida unica): best_val_acc=%.4f test_acc=%.4f",
                resultado["best_val_acc"], resultado["test_acc"])


if __name__ == "__main__":
    principal()
