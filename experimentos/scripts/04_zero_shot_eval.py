"""Baseline zero-shot: SmolVLM-500M-Instruct sem nenhum fine-tuning,
avaliado no split `test` real (data/real_crops/test/). Roda primeiro nesta
linha por padrão -- mesmo diagnóstico já usado nos 9 experimentos
anteriores (ver se colapsa numa classe default antes de gastar tempo de
treino)."""
from vlm_common import load_base_model, load_real_split, run_classification_eval, save_experiment_summary, setup_experiment_logging

EXP_NAME = "04_zero_shot_eval"
EXP_DIR, LOG_PATH, LOGGER, RUNTIME_METADATA = setup_experiment_logging(EXP_NAME, __file__)


def main():
    LOGGER.info("iniciando baseline zero-shot no split real test")
    processor, model = load_base_model()
    test_df = load_real_split("test")
    print(f"avaliando zero-shot em {len(test_df)} recortes reais de teste...")
    LOGGER.info("n_test=%s", len(test_df))
    results, acc = run_classification_eval(processor, model, test_df, "zeroshot_real_test")
    pred_counts = results["pred_label"].value_counts().to_dict()
    print(results["pred_label"].value_counts())
    print(f"\nacurácia zero-shot (real test, n={len(test_df)}): {acc:.3f}")

    summary = {
        "n_test": len(test_df),
        "acc": float(acc),
        "pred_label_counts": pred_counts,
        "output_csv": str(EXP_DIR / "reports" / "zeroshot_real_test_predictions.csv"),
    }
    save_experiment_summary(EXP_NAME, RUNTIME_METADATA, summary)
    LOGGER.info("Resumo zero-shot: n_test=%s acc=%.4f pred_counts=%s", len(test_df), acc, pred_counts)


if __name__ == "__main__":
    main()
