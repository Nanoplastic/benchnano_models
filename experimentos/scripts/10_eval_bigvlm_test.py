"""Avaliação do VLM maior no teste / Larger VLM test evaluation.

PT: Avalia em real_crops/test o melhor adaptador salvo pelo script 09
    (outputs/checkpoints/mina_real_only_lora_bigvlm).
    Roda em processo separado do treino porque o modelo de ~2B em bf16 ocupa quase toda a
    GPU de 6 GB; recarregar um segundo modelo base no mesmo processo para avaliar não cabe
    na memória.
EN: Evaluates on real_crops/test the best adapter saved by script 09
    (outputs/checkpoints/mina_real_only_lora_bigvlm).
    Runs in a process separate from training because the ~2B model in bf16 takes almost the
    whole 6 GB GPU; loading a second base model in the same process for evaluation does not
    fit in memory.
"""
from pathlib import Path

from peft import PeftModel

from vlm_common import DIR_CHECKPOINTS, carregar_modelo_base, carregar_particao_real, avaliar_classificacao

ID_MODELO = "HuggingFaceTB/SmolVLM-Instruct"
ROTULO_EXP = "mina_real_only_lora_bigvlm"


def principal():
    df_teste = carregar_particao_real("test")
    dir_adaptador = DIR_CHECKPOINTS / ROTULO_EXP

    processador, modelo_base = carregar_modelo_base(ID_MODELO)
    modelo = PeftModel.from_pretrained(modelo_base, dir_adaptador)
    modelo.config.use_cache = True

    resultados_teste, acur_teste = avaliar_classificacao(processador, modelo, df_teste, f"{ROTULO_EXP}_real_test")
    print(f"\nAcurácia final REAL TEST ({ROTULO_EXP}, n={len(df_teste)}): {acur_teste:.3f}")


if __name__ == "__main__":
    principal()
