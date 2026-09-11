"""Experimento #4 do plano (plano_experimentos_vlm.md): repete o
Experimento A (LoRA fine-tune só com dado real) trocando o modelo base de
SmolVLM-500M-Instruct para SmolVLM-Instruct (~2B, 4x maior), pra checar se
capacidade de modelo importa quando há dado suficiente pra ajustar (no
MiNa, ao contrário da linha `nanoplastic_fewshot` com só 7 imagens, onde um
modelo maior piorou).

Mesma receita/hiperparâmetros do Experimento A, mesmo split real
train/val/test -- única variável trocada é o modelo base, pra comparação
direta com os 72.1% já obtidos com o modelo pequeno.
"""
from vlm_common import finetune_lora, load_real_split

MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"
TAG = "mina_real_only_lora_bigvlm"


def main():
    train_df = load_real_split("train")
    val_df = load_real_split("val")
    test_df = load_real_split("test")
    print(f"Experimento #4 (VLM maior, só real): modelo={MODEL_ID}  "
          f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")
    print(train_df["label"].value_counts())

    finetune_lora(train_df, val_df, test_df, tag=TAG, model_id=MODEL_ID)


if __name__ == "__main__":
    main()
