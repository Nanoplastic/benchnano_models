"""10 repetições controladas do treino do YOLO26L principal -- espelha
experimentos/scripts/13_train_yolo26_corrected.py (mesmos hiperparâmetros,
mesmo dataset/split, mesmo checkpoint pré-treinado de partida), variando só
a seed do Ultralytics (0..9, explícita) entre repetições. Cada repetição
treina do zero a partir de yolo26l.pt (não é fine-tuning incremental).

Saída isolada em experimentos_controlados/ (nunca sobrescreve
experimentos/runs_v2/microplastic_yolo26_v2/, que continua sendo o
checkpoint original citado no artigo):
    runs/01_yolo26_main/rep_00/weights/best.pt ... rep_09/
    resultados/01_yolo26_main/rep_00/summary.json ... rep_09/

Depois de cada treino, avalia no MESMO split de teste travado
(datasets_yolo26_v2/images/test, 15 imagens, nunca visto em treino) com as
mesmas métricas agregadas de 14_yolo_test_eval.py (precision/recall/mAP50/
mAP50:95) -- os números citados no artigo (60,7%/46,7%/28,0%/8,6%) vêm
dessa mesma função, então servem de conferência: a repetição seed=0 deve
bater bit-a-bit com o valor original (Ultralytics é determinístico com seed
fixa).

Rodar (jobs longos de GPU -- acompanhar no terminal):
    python3 01_yolo26_main_10x.py [--only-seed N]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctrl_common as cc
from eval_and_pair import yolo_eval_test_set

from ultralytics import YOLO

EXPERIMENT = "01_yolo26_main"
DATASETMINA_ROOT = cc.DATASETMINA_ROOT
DATA_YAML = str(DATASETMINA_ROOT / "experimentos" / "datasets_yolo26_v2" / "data.yaml")
BASE_WEIGHTS = str(DATASETMINA_ROOT / "experimentos" / "yolo26l.pt")
PROJECT_DIR = str(cc.RUNS_DIR / EXPERIMENT)


def train_kwargs_for(seed: int, run_name: str) -> dict:
    # IDÊNTICO a 13_train_yolo26_corrected.py, só muda seed/project/name.
    return dict(
        data=DATA_YAML,
        task="detect",
        epochs=400,
        imgsz=1024,
        batch=-1,
        optimizer="AdamW",
        lr0=1e-4,
        lrf=0.1,
        cos_lr=True,
        seed=seed,
        device="0",
        project=PROJECT_DIR,
        name=run_name,
        save_period=-1,
        patience=100,
        close_mosaic=15,
        max_det=2000,
        degrees=180.0,
        translate=0.1,
        scale=0.2,
        flipud=0.5,
        fliplr=0.5,
        perspective=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.25,
        mosaic=0.3,
        mixup=0.0,
        erasing=0.0,
        plots=True,
        verbose=True,
        exist_ok=True,
    )


def run_repetition(seed: int, state: dict):
    if cc.is_rep_done(state, EXPERIMENT, seed):
        print(f"[{EXPERIMENT}] rep {seed:02d} já concluída, pulando.")
        return

    run_name = f"rep_{seed:02d}"
    print(f"\n=== {EXPERIMENT} — repetição {seed:02d}/9 (seed={seed}) ===")
    cc.set_current_job(EXPERIMENT, seed)
    cc.seed_everything(seed)

    model = YOLO(BASE_WEIGHTS, task="detect")
    kwargs = train_kwargs_for(seed, run_name)
    model.train(**kwargs)

    best_pt = Path(PROJECT_DIR) / run_name / "weights" / "best.pt"
    print(f"[{EXPERIMENT}] treino concluído: {best_pt}")

    metrics = yolo_eval_test_set(best_pt)
    print(f"[{EXPERIMENT}] avaliação rep {seed:02d}: {metrics['aggregate_metrics']}")

    cc.save_rep_summary(EXPERIMENT, seed, seed, metrics, extra_metadata={
        "best_weights": str(best_pt),
        "train_kwargs": kwargs,
    })
    cc.mark_rep_done(state, EXPERIMENT, seed)
    cc.clear_current_job()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only-seed", type=int, default=None,
                         help="roda só uma repetição específica (0-9), útil pro --dry-run da fila")
    args = parser.parse_args()

    state = cc.load_queue_state()
    seeds = [args.only_seed] if args.only_seed is not None else cc.SEEDS
    for seed in seeds:
        run_repetition(seed, state)


if __name__ == "__main__":
    main()
