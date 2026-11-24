"""Ajuste fino LoRA com dados reais + sintéticos / LoRA fine-tuning on real + synthetic data.

PT: Mesma receita do script 05, mas o treino usa data/real_crops/train/ + data/synthetic/
    como um único conjunto, embaralhado a cada época. Val e teste são exatamente os mesmos
    do script 05 (só dados reais), para comparação direta.
EN: Same recipe as script 05, but training uses data/real_crops/train/ + data/synthetic/ as
    a single pool, shuffled every epoch. Val and test are exactly the same as in script 05
    (real data only), for a direct comparison.
"""
import pandas as pd

from vlm_common import ajuste_fino_lora, carregar_particao_real, carregar_sintetico

ROTULO_EXP = "mina_real_plus_synthetic_lora_v2"  # PT: v2: fundos de PS com tamanho de bloco adaptativo; rótulo próprio para não sobrescrever saídas da v1 / EN: v2: PS backgrounds with adaptive tile size; separate tag so v1 outputs are not overwritten


def principal():
    df_treino_real = carregar_particao_real("train")
    df_sintetico = carregar_sintetico()
    df_treino = pd.concat([df_treino_real, df_sintetico], ignore_index=True)

    df_val = carregar_particao_real("val")
    df_teste = carregar_particao_real("test")

    print(f"Experimento B (real+sintético): "
          f"real_train={len(df_treino_real)}  synthetic={len(df_sintetico)}  "
          f"train_total={len(df_treino)}  val={len(df_val)}  test={len(df_teste)}")
    print(df_treino["label"].value_counts())

    ajuste_fino_lora(df_treino, df_val, df_teste, rotulo_exp=ROTULO_EXP)


if __name__ == "__main__":
    principal()
