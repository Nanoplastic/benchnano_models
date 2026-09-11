"""Experimento A: LoRA fine-tune do SmolVLM-500M-Instruct usando SÓ dado
real (data/real_crops/train/), sem nenhum dado sintético. Checkpoint
selecionado por acurácia no real_crops/val, avaliação final no
real_crops/test (nunca visto durante o treino)."""
from vlm_common import finetune_lora, load_real_split, save_experiment_summary, setup_experiment_logging

TAG = "mina_real_only_lora"
EXP_NAME = "05_finetune_real_only"
EXP_DIR, LOG_PATH, LOGGER, RUNTIME_METADATA = setup_experiment_logging(EXP_NAME, __file__, {"tag": TAG})


def main():
    LOGGER.info("iniciando experimento A real-only")
    train_df = load_real_split("train")
    val_df = load_real_split("val")
    test_df = load_real_split("test")
    print(f"Experimento A (só real): train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    print(train_df["label"].value_counts())
    LOGGER.info("train=%s val=%s test=%s classes=%s", len(train_df), len(val_df), len(test_df), train_df["label"].value_counts().to_dict())

    result = finetune_lora(train_df, val_df, test_df, tag=TAG)
    summary = {
        "tag": TAG,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "best_val_acc": float(result["best_val_acc"]),
        "test_acc": float(result["test_acc"]),
    }
    save_experiment_summary(EXP_NAME, RUNTIME_METADATA, summary)
    LOGGER.info("Resumo experimento A: best_val_acc=%.4f test_acc=%.4f", result["best_val_acc"], result["test_acc"])


if __name__ == "__main__":
    main()
