"""Experimento B: mesma receita do Experimento A (05), mas o treino agora
usa data/real_crops/train/ + data/synthetic/ (pool único, embaralhado
junto a cada época) -- interpretação literal do pedido do orientador
("gerar mais dados de forma sintética a partir destas classes" para
aumentar o que já temos, não substituir). val e test continuam sendo
EXATAMENTE os mesmos do Experimento A (só real), pra comparação justa."""
import pandas as pd

from vlm_common import finetune_lora, load_real_split, load_synthetic

TAG = "mina_real_plus_synthetic_lora_v2"  # v2: fundo de PS corrigido (Experimento #2 do plano) -- tag separada do v1 pra não sobrescrever outputs/checkpoints_backup_v1/


def main():
    real_train_df = load_real_split("train")
    synthetic_df = load_synthetic()
    train_df = pd.concat([real_train_df, synthetic_df], ignore_index=True)

    val_df = load_real_split("val")
    test_df = load_real_split("test")

    print(f"Experimento B (real+sintético): "
          f"real_train={len(real_train_df)}  synthetic={len(synthetic_df)}  "
          f"train_total={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    print(train_df["label"].value_counts())

    finetune_lora(train_df, val_df, test_df, tag=TAG)


if __name__ == "__main__":
    main()
