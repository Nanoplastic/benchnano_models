"""Avaliação final do Experimento #4 (VLM maior) no real_crops/test.

Separado de `09_finetune_bigger_vlm.py` porque o treino já terminou e salvou
o melhor adapter em outputs/checkpoints/mina_real_only_lora_bigvlm (val_acc
0.671), mas o processo original quebrou por falta de VRAM na hora de
recarregar um segundo modelo base pra avaliação (2B em bf16 quase satura os
6GB da GPU sozinho; duas cópias no mesmo processo não cabem). Rodar a
avaliação como processo novo, com a GPU limpa, evita isso sem precisar
retreinar.
"""
from pathlib import Path

from peft import PeftModel

from vlm_common import CHECKPOINTS_DIR, load_base_model, load_real_split, run_classification_eval

MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"
TAG = "mina_real_only_lora_bigvlm"


def main():
    test_df = load_real_split("test")
    adapter_dir = CHECKPOINTS_DIR / TAG

    processor, base_model = load_base_model(MODEL_ID)
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.config.use_cache = True

    test_results, test_acc = run_classification_eval(processor, model, test_df, f"{TAG}_real_test")
    print(f"\nAcurácia final REAL TEST ({TAG}, n={len(test_df)}): {test_acc:.3f}")


if __name__ == "__main__":
    main()
