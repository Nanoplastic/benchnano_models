"""10 repetições controladas do full fine-tuning (todos os parâmetros, sem
LoRA) do SmolVLM na tarefa de regime de tamanho -- espelha
experimentos/scripts/18_size_finetune_full.py (mesmo manifesto, mesma
condição reference-box). Testa a recomendação oficial do blog do SmolVLM2
pra essa variante (500M): full fine-tuning em vez de LoRA/QLoRA.

ATENÇÃO: usa bem mais VRAM que o LoRA (todos os ~500M parâmetros recebem
gradiente + estado do otimizador AdamW) -- mesma ressalva do script
original sobre a RTX 4060 Ti (16GB).

Rodar (job longo de GPU -- acompanhar no terminal):
    python3 06_smolvlm_size_full_10x.py [--only-seed N]
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctrl_common as cc

sys.path.insert(0, str(cc.EXPERIMENTOS_SCRIPTS))
from vlm_common import CHECKPOINTS_DIR, RESULTS_ROOT, SIZE_CLASSIFY_PROMPT, SIZE_LABELS, finetune_full

EXPERIMENT = "06_smolvlm_size_full"
BASE_TAG = "size_full_ft"
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

    result = finetune_full(train_df, val_df, test_df, tag=tag, seed=seed,
                            prompt=SIZE_CLASSIFY_PROMPT, labels=SIZE_LABELS)

    metrics = {"best_val_acc": float(result["best_val_acc"]), "test_acc": float(result["test_acc"])}
    print(f"[{EXPERIMENT}] rep {seed:02d}: test_acc={metrics['test_acc']:.3f}")
    cc.save_rep_summary(EXPERIMENT, seed, seed, metrics, extra_metadata={
        "tag": tag,
        "checkpoint_dir": str(CHECKPOINTS_DIR / tag),
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
