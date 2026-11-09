"""Copia fiel de 17_size_finetune_lora.py, trocando so o manifesto: usa
manifest_size_regime_eq1.csv (regime pelo diametro equivalente da area do
poligono COCO, script 57) em vez de manifest_size_regime.csv (regime pela
caixa, Eq. 2).

Motivo: as Tabelas A4/A5 do apendice e a Fig. 4a (painel esquerdo)/4b usam
uma corrida unica (nao as 10 repeticoes controladas) de zero-shot/LoRA/full
fine-tuning. As 10 repeticoes controladas ja foram retreinadas sob Eq.1
(experimentos_controlados, tag *_eq1_ctrl_seed*), mas essa corrida unica
nunca foi -- reclassificar as previsoes antigas (treinadas sob Eq.2) contra
o gabarito novo sem retreinar subestimaria o modelo, porque ele nunca viu a
fronteira certa durante o treino. Por isso esta e uma corrida de
fine-tuning nova (nao so uma reclassificacao), com a MESMA receita exata de
17 (r=8, alpha=8, 6 epocas, seed=42, mesmos splits train/val/test por
contagem) -- a unica diferenca e o manifesto de origem.

NAO sobrescreve o adapter/checkpoint de 17 (tag diferente: size_lora_eq1_single,
em vez de size_lora) -- a corrida original sob Eq.2 continua intacta em
resultados/checkpoints/size_lora e resultados/reports/size_lora_real_test_predictions.csv.

Uso:
    python3 experimentos/scripts/80_size_finetune_lora_eq1_single.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import (
    RESULTS_ROOT, SIZE_CLASSIFY_PROMPT, SIZE_LABELS, finetune_lora,
    save_experiment_summary, setup_experiment_logging,
)

TAG = "size_lora_eq1_single"
EXP_NAME = "80_size_finetune_lora_eq1_single"
EXP_DIR, LOG_PATH, LOGGER, RUNTIME_METADATA = setup_experiment_logging(EXP_NAME, __file__, {"tag": TAG})

MANIFEST = RESULTS_ROOT / "57_build_size_regime_manifest_eq1" / "data" / "manifest_size_regime_eq1.csv"
EXPECTED_SPLIT_SIZES = {"train": 1000, "val": 240, "test": 208}  # mesmas contagens de 15/17 (Eq.2)


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

    print(f"LoRA regime de tamanho (Eq.1, corrida unica): train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    print(train_df["label"].value_counts())
    LOGGER.info("train=%s val=%s test=%s classes=%s", len(train_df), len(val_df), len(test_df),
                train_df["label"].value_counts().to_dict())

    result = finetune_lora(train_df, val_df, test_df, tag=TAG,
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
    LOGGER.info("Resumo LoRA regime de tamanho (Eq.1, corrida unica): best_val_acc=%.4f test_acc=%.4f",
                result["best_val_acc"], result["test_acc"])


if __name__ == "__main__":
    main()
