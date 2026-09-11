"""10 repetições controladas do Experimento B (LoRA do SmolVLM-500M-Instruct
com dado real + sintético) -- espelha
experimentos/scripts/06_finetune_real_plus_synthetic.py (mesma receita do
Experimento A, mas train_df = real_train + synthetic; val/test continuam
SÓ reais, exatamente como no original, para comparação justa com 02).

Ver 02_vlm_polymer_real_only_10x.py para a explicação completa do desenho
(seed completa, tags "_ctrl_repN", checkpoints/reports na pasta
compartilhada de sempre).

Rodar (job longo de GPU -- acompanhar no terminal):
    python3 03_vlm_polymer_real_synth_10x.py [--only-seed N]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctrl_common as cc

sys.path.insert(0, str(cc.EXPERIMENTOS_SCRIPTS))
from vlm_common import CHECKPOINTS_DIR, finetune_lora, load_real_split, load_synthetic

EXPERIMENT = "03_vlm_polymer_real_synth"
BASE_TAG = "mina_real_plus_synthetic_lora_v2"


def run_repetition(seed: int, state: dict):
    if cc.is_rep_done(state, EXPERIMENT, seed):
        print(f"[{EXPERIMENT}] rep {seed:02d} já concluída, pulando.")
        return

    tag = f"{BASE_TAG}_ctrl_seed{seed}"
    print(f"\n=== {EXPERIMENT} — repetição {seed:02d}/9 (seed={seed}, tag={tag}) ===")
    cc.set_current_job(EXPERIMENT, seed)
    cc.seed_everything(seed)

    real_train_df = load_real_split("train")
    synthetic_df = load_synthetic()
    train_df = pd.concat([real_train_df, synthetic_df], ignore_index=True)
    val_df = load_real_split("val")
    test_df = load_real_split("test")

    result = finetune_lora(train_df, val_df, test_df, tag=tag, seed=seed)

    metrics = {"best_val_acc": float(result["best_val_acc"]), "test_acc": float(result["test_acc"])}
    print(f"[{EXPERIMENT}] rep {seed:02d}: test_acc={metrics['test_acc']:.3f}")
    cc.save_rep_summary(EXPERIMENT, seed, seed, metrics, extra_metadata={
        "tag": tag,
        "adapter_dir": str(CHECKPOINTS_DIR / tag),
        "real_train_samples": len(real_train_df), "synthetic_samples": len(synthetic_df),
        "train_samples": len(train_df), "val_samples": len(val_df), "test_samples": len(test_df),
    })
    cc.mark_rep_done(state, EXPERIMENT, seed)
    cc.clear_current_job()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only-seed", type=int, default=None)
    args = parser.parse_args()

    state = cc.load_queue_state()
    seeds = [args.only_seed] if args.only_seed is not None else cc.SEEDS
    for seed in seeds:
        run_repetition(seed, state)


if __name__ == "__main__":
    main()
