"""10 repetições controladas do Experimento #4 (LoRA do SmolVLM-Instruct
~2B, modelo maior) -- espelha experimentos/scripts/09_finetune_bigger_vlm.py
(mesma receita/hiperparâmetros do Experimento A, só troca o modelo base).

Duração por repetição não fica medida no repositório original (o script 09
nunca teve log com timestamp completo) -- a primeira repetição aqui serve
também pra medir isso e ajustar a estimativa de tempo total da fila.

Ver 02_vlm_polymer_real_only_10x.py para a explicação completa do desenho.

Rodar (job longo de GPU -- acompanhar no terminal):
    python3 04_vlm_polymer_bigvlm_10x.py [--only-seed N]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctrl_common as cc

sys.path.insert(0, str(cc.EXPERIMENTOS_SCRIPTS))
from vlm_common import CHECKPOINTS_DIR, finetune_lora, load_real_split

EXPERIMENT = "04_vlm_polymer_bigvlm"
BASE_TAG = "mina_real_only_lora_bigvlm"
MODEL_ID = "HuggingFaceTB/SmolVLM-Instruct"


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

    t0 = time.monotonic()
    result = finetune_lora(train_df, val_df, test_df, tag=tag, seed=seed, model_id=MODEL_ID)
    elapsed_min = (time.monotonic() - t0) / 60.0

    metrics = {"best_val_acc": float(result["best_val_acc"]), "test_acc": float(result["test_acc"])}
    print(f"[{EXPERIMENT}] rep {seed:02d}: test_acc={metrics['test_acc']:.3f} ({elapsed_min:.1f} min)")
    cc.save_rep_summary(EXPERIMENT, seed, seed, metrics, extra_metadata={
        "tag": tag,
        "adapter_dir": str(CHECKPOINTS_DIR / tag),
        "model_id": MODEL_ID,
        "elapsed_minutes": round(elapsed_min, 1),
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
