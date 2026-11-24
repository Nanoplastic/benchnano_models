"""Ajuste fino LoRA com VLM maior / LoRA fine-tuning with a larger VLM.

PT: Repete o ajuste fino LoRA só com dados reais (script 05) trocando o modelo base de
    SmolVLM-500M-Instruct para SmolVLM-Instruct (~2B parâmetros, 4x maior), para medir o
    efeito da capacidade do modelo.
    Mesma receita, hiperparâmetros e partição real treino/val/teste do script 05; a única
    variável alterada é o modelo base.
EN: Repeats the real-only LoRA fine-tuning (script 05) switching the base model from
    SmolVLM-500M-Instruct to SmolVLM-Instruct (~2B parameters, 4x larger), to measure the
    effect of model capacity.
    Same recipe, hyperparameters and real train/val/test split as script 05; the only
    changed variable is the base model.
"""
from vlm_common import ajuste_fino_lora, carregar_particao_real

ID_MODELO = "HuggingFaceTB/SmolVLM-Instruct"
ROTULO_EXP = "mina_real_only_lora_bigvlm"


def principal():
    df_treino = carregar_particao_real("train")
    df_val = carregar_particao_real("val")
    df_teste = carregar_particao_real("test")
    print(f"Experimento #4 (VLM maior, só real): modelo={ID_MODELO}  "
          f"train={len(df_treino)}  val={len(df_val)}  test={len(df_teste)}")
    print(df_treino["label"].value_counts())

    ajuste_fino_lora(df_treino, df_val, df_teste, rotulo_exp=ROTULO_EXP, id_modelo=ID_MODELO)


if __name__ == "__main__":
    principal()
