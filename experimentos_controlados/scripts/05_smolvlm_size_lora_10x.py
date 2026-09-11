"""10 repetições controladas do fine-tune LoRA do SmolVLM na tarefa de
regime de tamanho (NANOPLASTIC/MICROPLASTIC) -- espelha
experimentos/scripts/17_size_finetune_lora.py (mesma receita r=8 alpha=8,
mesmo manifesto de regime de tamanho, condição reference-box). É o
experimento cujo test_acc (90,9%) é citado no abstract do artigo.

Depois que as 10 repetições daqui E as 10 de 01_yolo26_main_10x.py
terminarem, rode eval_and_pair.py pra gerar a avaliação fim-a-fim pareada
(rep i do YOLO + rep i daqui), que reproduz o número de 90,8%/29,2% do
artigo (condição predicted-box / fim-a-fim estrito) — ver
07_end_to_end_paired em resultados/.

Rodar (job longo de GPU -- acompanhar no terminal):
    python3 05_smolvlm_size_lora_10x.py [--only-seed N]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctrl_common as cc

sys.path.insert(0, str(cc.EXPERIMENTOS_SCRIPTS))
from vlm_common import (
    CHECKPOINTS_DIR, RESULTS_ROOT, SIZE_CLASSIFY_PROMPT, SIZE_LABELS, finetune_lora,
)

EXPERIMENT = "05_smolvlm_size_lora"
BASE_TAG = "size_lora"
MANIFEST = RESULTS_ROOT / "15_build_size_regime_manifest" / "data" / "manifest_size_regime.csv"


def run_repetition(seed: int, state: dict):
    if cc.is_rep_done(state, EXPERIMENT, seed):
        print(f"[{EXPERIMENT}] rep {seed:02d} já concluída, pulando.")
        return

    if not MANIFEST.exists():
        raise FileNotFoundError(f"{MANIFEST} não existe -- rode 15_build_size_regime_manifest.py primeiro.")

    tag = f"{BASE_TAG}_ctrl_seed{seed}"
    print(f"\n=== {EXPERIMENT} — repetição {seed:02d}/9 (seed={seed}, tag={tag}) ===")
    cc.set_current_job(EXPERIMENT, seed)
    cc.seed_everything(seed)

    df = pd.read_csv(MANIFEST)
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)

    result = finetune_lora(train_df, val_df, test_df, tag=tag, seed=seed,
                            prompt=SIZE_CLASSIFY_PROMPT, labels=SIZE_LABELS)

    metrics = {"best_val_acc": float(result["best_val_acc"]), "test_acc": float(result["test_acc"])}
    print(f"[{EXPERIMENT}] rep {seed:02d}: test_acc={metrics['test_acc']:.3f}")
    cc.save_rep_summary(EXPERIMENT, seed, seed, metrics, extra_metadata={
        "tag": tag,
        "adapter_dir": str(CHECKPOINTS_DIR / tag),
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
