"""Copia fiel de 18_size_finetune_full.py, trocando so o manifesto: usa
manifest_size_regime_eq1.csv (regime pelo diametro equivalente da area do
poligono COCO, script 57) em vez de manifest_size_regime.csv (regime pela
caixa, Eq. 2). Mesmo motivo e mesma ressalva de 80_size_finetune_lora_eq1_single.py
(corrida unica nova, nao reclassificacao da corrida antiga).

NAO sobrescreve o checkpoint de 18 (tag diferente: size_full_ft_eq1_single,
em vez de size_full_ft).

Uso:
    python3 experimentos/scripts/81_size_finetune_full_eq1_single.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import (
    RESULTS_ROOT, SIZE_CLASSIFY_PROMPT, SIZE_LABELS, finetune_full,
    save_experiment_summary, setup_experiment_logging,
)

TAG = "size_full_ft_eq1_single"
EXP_NAME = "81_size_finetune_full_eq1_single"
EXP_DIR, LOG_PATH, LOGGER, RUNTIME_METADATA = setup_experiment_logging(EXP_NAME, __file__, {"tag": TAG})

MANIFEST = RESULTS_ROOT / "57_build_size_regime_manifest_eq1" / "data" / "manifest_size_regime_eq1.csv"
EXPECTED_SPLIT_SIZES = {"train": 1000, "val": 240, "test": 208}


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"{MANIFEST} não existe -- rode 57_build_size_regime_manifest_eq1.py primeiro.")

    df = pd.read_csv(MANIFEST)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    for name, sub, expected in [("train", train_df, EXPECTED_SPLIT_SIZES["train"]),
                                 ("val", val_df, EXPECTED_SPLIT_SIZES["val"]),
                                 ("test", test_df, EXPECTED_SPLIT_SIZES["test"])]:
        if len(sub) != expected:
            raise SystemExit(f"sanidade falhou: split '{name}' tem {len(sub)} linhas, esperado {expected} "
                              f"(mesma contagem da corrida Eq.2 original) -- abortando antes de treinar.")

    print(f"Full fine-tuning regime de tamanho (Eq.1, corrida unica): train={len(train_df)}  "
          f"val={len(val_df)}  test={len(test_df)}")
    print(train_df["label"].value_counts())
    LOGGER.info("train=%s val=%s test=%s classes=%s", len(train_df), len(val_df), len(test_df),
                train_df["label"].value_counts().to_dict())

    result = finetune_full(train_df, val_df, test_df, tag=TAG,
                            prompt=SIZE_CLASSIFY_PROMPT, labels=SIZE_LABELS)

    summary = {
        "tag": TAG,
        "manifest": str(MANIFEST),
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "best_val_acc": float(result["best_val_acc"]),
        "test_acc": float(result["test_acc"]),
    }
    save_experiment_summary(EXP_NAME, RUNTIME_METADATA, summary)
    LOGGER.info("Resumo full fine-tuning regime de tamanho (Eq.1, corrida unica): best_val_acc=%.4f test_acc=%.4f",
                result["best_val_acc"], result["test_acc"])


if __name__ == "__main__":
    main()
