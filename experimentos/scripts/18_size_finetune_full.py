"""Full fine-tuning (todos os parâmetros, sem LoRA) do SmolVLM-500M-Instruct
na tarefa de regime de tamanho (NANOPLASTIC/MICROPLASTIC). Testa a
recomendação oficial do blog do SmolVLM2 pra essa variante (500M): "since
the 500M variant is small, it's better to apply full fine-tuning instead of
QLoRA or LoRA" (ver raw/smolvlm_finetuning_guidance.md, achado no
graphify-out) -- nunca testada nos experimentos anteriores deste projeto,
que usaram LoRA em todas as rodadas.

ATENÇÃO: usa bem mais VRAM que o LoRA (todos os ~500M parâmetros recebem
gradiente + estado do otimizador AdamW, não só o adapter de baixo posto).
Se estourar memória na RTX 4060 Ti (16GB), reduzir grad_accum não ajuda
(seq len 1 já é batch=1); a saída é usar 8-bit optimizer (bitsandbytes) ou
LoRA com rank maior como meio-termo -- não implementado aqui, avaliar se
necessário depois de tentar rodar.
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import (
    RESULTS_ROOT, SIZE_CLASSIFY_PROMPT, SIZE_LABELS, finetune_full,
    save_experiment_summary, setup_experiment_logging,
)

TAG = "size_full_ft"
EXP_NAME = "18_size_finetune_full"
EXP_DIR, LOG_PATH, LOGGER, RUNTIME_METADATA = setup_experiment_logging(EXP_NAME, __file__, {"tag": TAG})

MANIFEST = RESULTS_ROOT / "15_build_size_regime_manifest" / "data" / "manifest_size_regime.csv"


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"{MANIFEST} não existe -- rode 15_build_size_regime_manifest.py primeiro.")

    df = pd.read_csv(MANIFEST)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    print(f"Full fine-tuning regime de tamanho: train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    print(train_df["label"].value_counts())
    LOGGER.info("train=%s val=%s test=%s classes=%s", len(train_df), len(val_df), len(test_df),
                train_df["label"].value_counts().to_dict())

    result = finetune_full(train_df, val_df, test_df, tag=TAG,
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
    LOGGER.info("Resumo full fine-tuning regime de tamanho: best_val_acc=%.4f test_acc=%.4f",
                result["best_val_acc"], result["test_acc"])


if __name__ == "__main__":
    main()
