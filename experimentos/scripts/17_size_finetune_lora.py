"""Fine-tune LoRA do SmolVLM-500M-Instruct na tarefa de regime de tamanho
(NANOPLASTIC/MICROPLASTIC) -- mesma receita (r=8, alpha=8, 6 épocas) do
Experimento A da linha de polímero, só trocando prompt e rótulos via
vlm_common.finetune_lora(prompt=..., labels=...). Condição "reference-box"
(mesmos recortes com bbox de referência do MiNa).

Comparação com zero-shot (16) e full fine-tuning (18) na mesma tarefa --
ver 19_compare_size_regime_configs.py para o teste estatístico pareado.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import (
    RESULTS_ROOT, SIZE_CLASSIFY_PROMPT, SIZE_LABELS, finetune_lora,
    save_experiment_summary, setup_experiment_logging,
)

TAG = "size_lora"
EXP_NAME = "17_size_finetune_lora"
EXP_DIR, LOG_PATH, LOGGER, RUNTIME_METADATA = setup_experiment_logging(EXP_NAME, __file__, {"tag": TAG})

MANIFEST = RESULTS_ROOT / "15_build_size_regime_manifest" / "data" / "manifest_size_regime.csv"


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"{MANIFEST} não existe -- rode 15_build_size_regime_manifest.py primeiro.")

    df = pd.read_csv(MANIFEST)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    print(f"LoRA regime de tamanho: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    print(train_df["label"].value_counts())
    LOGGER.info("train=%s val=%s test=%s classes=%s", len(train_df), len(val_df), len(test_df),
                train_df["label"].value_counts().to_dict())

    result = finetune_lora(train_df, val_df, test_df, tag=TAG,
                            prompt=SIZE_CLASSIFY_PROMPT, labels=SIZE_LABELS)

    summary = {
        "tag": TAG,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "best_val_acc": float(result["best_val_acc"]),
        "test_acc": float(result["test_acc"]),
    }
    save_experiment_summary(EXP_NAME, RUNTIME_METADATA, summary)
    LOGGER.info("Resumo LoRA regime de tamanho: best_val_acc=%.4f test_acc=%.4f",
                result["best_val_acc"], result["test_acc"])


if __name__ == "__main__":
    main()
