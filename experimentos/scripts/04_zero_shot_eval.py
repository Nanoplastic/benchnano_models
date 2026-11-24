"""Avaliação zero-shot do SmolVLM / SmolVLM zero-shot evaluation.

PT: Avalia o SmolVLM-500M-Instruct sem ajuste fino na partição `test` real
    (data/real_crops/test/). Serve de linha de base e de diagnóstico de colapso numa única
    classe antes do treino.
EN: Evaluates SmolVLM-500M-Instruct without fine-tuning on the real `test` split
    (data/real_crops/test/). Serves as a baseline and as a check for collapse onto a single
    class before training.
"""
from vlm_common import carregar_modelo_base, carregar_particao_real, avaliar_classificacao, salvar_resumo_experimento, configurar_log_experimento

NOME_EXP = "04_zero_shot_eval"
DIR_EXP, CAMINHO_LOG, REGISTRADOR, METADADOS_EXECUCAO = configurar_log_experimento(NOME_EXP, __file__)


def principal():
    REGISTRADOR.info("iniciando baseline zero-shot no split real test")
    processador, modelo = carregar_modelo_base()
    df_teste = carregar_particao_real("test")
    print(f"avaliando zero-shot em {len(df_teste)} recortes reais de teste...")
    REGISTRADOR.info("n_test=%s", len(df_teste))
    resultados, acur = avaliar_classificacao(processador, modelo, df_teste, "zeroshot_real_test")
    contagens_pred = resultados["pred_label"].value_counts().to_dict()
    print(resultados["pred_label"].value_counts())
    print(f"\nacurácia zero-shot (real test, n={len(df_teste)}): {acur:.3f}")

    resumo = {
        "n_test": len(df_teste),
        "acc": float(acur),
        "pred_label_counts": contagens_pred,
        "output_csv": str(DIR_EXP / "reports" / "zeroshot_real_test_predictions.csv"),
    }
    salvar_resumo_experimento(NOME_EXP, METADADOS_EXECUCAO, resumo)
    REGISTRADOR.info("Resumo zero-shot: n_test=%s acc=%.4f pred_counts=%s", len(df_teste), acur, contagens_pred)


if __name__ == "__main__":
    principal()
