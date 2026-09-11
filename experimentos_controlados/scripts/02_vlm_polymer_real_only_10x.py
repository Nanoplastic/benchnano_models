"""10 repetições controladas do Experimento A (LoRA do SmolVLM-500M-Instruct
só com dado real) -- espelha experimentos/scripts/05_finetune_real_only.py
(mesma receita: r=8 alpha=8 dropout=0.1, 6 épocas, lr=1e-4), variando a seed
completa (random/numpy/torch, via ctrl_common.seed_everything) entre
repetições. O original só tinha 4 execuções ad hoc (run/run2/run3/run4);
aqui são 10, com seed explícita e documentada (0..9).

Reaproveita vlm_common.finetune_lora sem modificação -- os adapters/CSVs de
predição continuam na mesma pasta compartilhada do projeto
(experimentos/resultados/checkpoints/, reports/), só com tags únicas
"_ctrl_repN" (mesma convenção dos _run2/_run3/_run4 originais, nunca
sobrescreve nada).

Rodar (job longo de GPU -- acompanhar no terminal):
    python3 02_vlm_polymer_real_only_10x.py [--only-seed N]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctrl_common as cc

sys.path.insert(0, str(cc.EXPERIMENTOS_SCRIPTS))
from vlm_common import CHECKPOINTS_DIR, finetune_lora, load_real_split

EXPERIMENT = "02_vlm_polymer_real_only"
BASE_TAG = "mina_real_only_lora"


def run_repetition(seed: int, state: dict):
    if cc.is_rep_done(state, EXPERIMENT, seed):
        print(f"[{EXPERIMENT}] rep {seed:02d} já concluída, pulando.")
        return

    tag = f"{BASE_TAG}_ctrl_seed{seed}"
    print(f"\n=== {EXPERIMENT} — repetição {seed:02d}/9 (seed={seed}, tag={tag}) ===")
    cc.set_current_job(EXPERIMENT, seed)
    cc.seed_everything(seed)

    train_df = load_real_split("train")
    val_df = load_real_split("val")
    test_df = load_real_split("test")

    result = finetune_lora(train_df, val_df, test_df, tag=tag, seed=seed)

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
