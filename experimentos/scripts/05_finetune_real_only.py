"""Ajuste fino LoRA só com dados reais / LoRA fine-tuning on real data only.

PT: Ajuste fino LoRA do SmolVLM-500M-Instruct usando só dados reais
    (data/real_crops/train/), sem dados sintéticos. O checkpoint é escolhido pela acurácia
    em real_crops/val e a avaliação final é em real_crops/test, nunca visto no treino.
EN: LoRA fine-tuning of SmolVLM-500M-Instruct using only real data (data/real_crops/train/),
    with no synthetic data. The checkpoint is selected by accuracy on real_crops/val and the
    final evaluation is on real_crops/test, never seen during training.
"""
from vlm_common import ajuste_fino_lora, carregar_particao_real, salvar_resumo_experimento, configurar_log_experimento

ROTULO_EXP = "mina_real_only_lora"
NOME_EXP = "05_finetune_real_only"
DIR_EXP, CAMINHO_LOG, REGISTRADOR, METADADOS_EXECUCAO = configurar_log_experimento(NOME_EXP, __file__, {"tag": ROTULO_EXP})


def principal():
    REGISTRADOR.info("iniciando experimento A real-only")
    df_treino = carregar_particao_real("train")
    df_val = carregar_particao_real("val")
    df_teste = carregar_particao_real("test")
    print(f"Experimento A (só real): train={len(df_treino)}  val={len(df_val)}  test={len(df_teste)}")
    print(df_treino["label"].value_counts())
    REGISTRADOR.info("train=%s val=%s test=%s classes=%s", len(df_treino), len(df_val), len(df_teste), df_treino["label"].value_counts().to_dict())

    resultado = ajuste_fino_lora(df_treino, df_val, df_teste, rotulo_exp=ROTULO_EXP)
    resumo = {
        "tag": ROTULO_EXP,
        "train_samples": len(df_treino),
        "val_samples": len(df_val),
        "test_samples": len(df_teste),
        "best_val_acc": float(resultado["best_val_acc"]),
        "test_acc": float(resultado["test_acc"]),
    }
    salvar_resumo_experimento(NOME_EXP, METADADOS_EXECUCAO, resumo)
    REGISTRADOR.info("Resumo experimento A: best_val_acc=%.4f test_acc=%.4f", resultado["best_val_acc"], resultado["test_acc"])


if __name__ == "__main__":
    principal()
