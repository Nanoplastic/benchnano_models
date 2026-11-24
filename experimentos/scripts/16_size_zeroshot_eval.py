"""Avaliação zero-shot de regime de tamanho / Size regime zero-shot evaluation.

PT: Linha de base zero-shot do SmolVLM-500M-Instruct na classificação de regime de tamanho
    (NANOPLASTIC/MICROPLASTIC, limiar de 1 µm), condição "caixa de referência" (recorte pela
    bbox de referência do MiNa com margem de 10 px).
    Sem treino. Verifica colapso numa única classe antes dos ajustes finos (scripts 17 e
    18).
EN: Zero-shot baseline of SmolVLM-500M-Instruct on size regime classification
    (NANOPLASTIC/MICROPLASTIC, 1 µm threshold), "reference box" condition (crop from the
    MiNa reference bbox with a 10 px margin).
    No training. Checks for collapse onto a single class before the fine-tuning runs
    (scripts 17 and 18).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import (
    RAIZ_RESULTADOS, PROMPT_CLASSIFICACAO_TAMANHO, ROTULOS_TAMANHO, carregar_modelo_base,
    avaliar_classificacao, salvar_resumo_experimento, configurar_log_experimento,
)

NOME_EXP = "16_size_zeroshot_eval"
DIR_EXP, CAMINHO_LOG, REGISTRADOR, METADADOS_EXECUCAO = configurar_log_experimento(NOME_EXP, __file__)

MANIFESTO = RAIZ_RESULTADOS / "15_build_size_regime_manifest" / "data" / "manifest_size_regime.csv"
DIR_RELATORIOS = DIR_EXP / "reports"


def principal():
    if not MANIFESTO.exists():
        raise FileNotFoundError(f"{MANIFESTO} não existe -- rode 15_build_size_regime_manifest.py primeiro.")

    df = pd.read_csv(MANIFESTO)
    df_teste = df[df["split"] == "test"].reset_index(drop=True)
    REGISTRADOR.info("iniciando baseline zero-shot (regime de tamanho) no split real test")
    REGISTRADOR.info("n_test=%s", len(df_teste))
    print(f"avaliando zero-shot (NANO/MICRO) em {len(df_teste)} recortes reais de teste...")

    processador, modelo = carregar_modelo_base()
    resultados, acur = avaliar_classificacao(
        processador, modelo, df_teste, "size_zeroshot_real_test",
        dir_saida=DIR_RELATORIOS, prompt=PROMPT_CLASSIFICACAO_TAMANHO, rotulos=ROTULOS_TAMANHO,
    )
    contagens_pred = resultados["pred_label"].value_counts().to_dict()
    print(resultados["pred_label"].value_counts())
    print(f"\nacurácia zero-shot regime de tamanho (real test, n={len(df_teste)}): {acur:.3f}")
    print("\nMatriz de confusão (linha=real, coluna=previsto):")
    print(pd.crosstab(resultados["true_label"], resultados["pred_label"]))

    resumo = {
        "n_test": len(df_teste),
        "acc": float(acur),
        "pred_label_counts": contagens_pred,
        "output_csv": str(DIR_RELATORIOS / "size_zeroshot_real_test_predictions.csv"),
    }
    salvar_resumo_experimento(NOME_EXP, METADADOS_EXECUCAO, resumo)
    REGISTRADOR.info("Resumo zero-shot regime de tamanho: n_test=%s acc=%.4f pred_counts=%s", len(df_teste), acur, contagens_pred)


if __name__ == "__main__":
    principal()
