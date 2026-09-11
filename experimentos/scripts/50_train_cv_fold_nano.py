"""Treina UM fold da cross-validation NANO-ONLY (mesmos folds/seed de
49_build_nano_cv_folds.py, caixas de treino/val já filtradas pra partículas
<1000nm) -- MESMOS hiperparâmetros de treino-do-zero de 44_train_cv_fold.py,
só muda o data.yaml (aponta pro fold nano) e o diretório de saída
(runs_v2_cv_nano/foldK/).

Rodar uma vez por fold, sequencialmente (49_build_nano_cv_folds.py precisa
ter rodado antes):

    python3 scripts/50_train_cv_fold_nano.py --fold 0
    python3 scripts/50_train_cv_fold_nano.py --fold 1
    python3 scripts/50_train_cv_fold_nano.py --fold 2
    python3 scripts/50_train_cv_fold_nano.py --fold 3
    python3 scripts/50_train_cv_fold_nano.py --fold 4

Cada um treina do zero a partir do checkpoint pré-treinado yolo26l.pt (igual
à CV original, não é fine-tuning).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import save_experiment_summary, setup_experiment_logging

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
CV_DATASETS_DIR = ROOT / "datasets_yolo26_v2_cv_nano"
PROJECT_DIR = str(ROOT / "runs_v2_cv_nano")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True, help="indice do fold (0..4)")
    parser.add_argument("--seed", type=int, default=0,
                         help="seed do ultralytics (default=0, mesmo default usado na CV original "
                              "e nos outros folds). Só passe um valor diferente pra testar "
                              "explicitamente sensibilidade ao seed=0 especifico.")
    args = parser.parse_args()

    fold_dir = CV_DATASETS_DIR / f"fold{args.fold}"
    data_yaml = fold_dir / "data.yaml"
    if not data_yaml.exists():
        print(f"ERRO: {data_yaml} nao existe -- rode 49_build_nano_cv_folds.py primeiro.")
        sys.exit(1)

    run_name = f"fold{args.fold}" if args.seed == 0 else f"fold{args.fold}_seed{args.seed}"
    exp_name = f"50_train_cv_fold_nano{args.fold}" if args.seed == 0 else f"50_train_cv_fold_nano{args.fold}_seed{args.seed}"
    exp_dir, log_path, logger, runtime_metadata = setup_experiment_logging(exp_name, __file__)
    logger.info("=== treino cross-validation nano-only fold=%d ===", args.fold)
    logger.info("data_yaml=%s", data_yaml)

    model = YOLO(str(ROOT / "yolo26l.pt"), task="detect")

    # MESMOS hiperparametros de treino-do-zero de 44_train_cv_fold.py -- so
    # muda data/project/name (aponta pro fold nano).
    train_kwargs = dict(
        data=str(data_yaml),
        task="detect",
        epochs=400,
        imgsz=1024,
        batch=-1,
        optimizer="AdamW",
        lr0=1e-4,
        lrf=0.1,
        cos_lr=True,
        seed=args.seed,
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
    logger.info("hiperparametros=%s", train_kwargs)
    print(f"Fold {args.fold} (nano-only) -- hiperparametros: {train_kwargs}")

    model.train(**train_kwargs)

    best = Path(PROJECT_DIR) / run_name / "weights" / "best.pt"
    logger.info("treino do fold %d (nano) concluido -- best.pt em %s", args.fold, best)

    save_experiment_summary(exp_name, runtime_metadata, {
        "fold": args.fold,
        "data_yaml": str(data_yaml),
        "best_weights": str(best),
        "results_csv": str(Path(PROJECT_DIR) / run_name / "results.csv"),
        "train_kwargs": train_kwargs,
    })
    print(f"\nFold {args.fold} (nano) concluido. Pesos: {best}")
    print(f"Log completo: {log_path}")
    print(f"Curva de treino: {Path(PROJECT_DIR) / run_name / 'results.csv'}")


if __name__ == "__main__":
    main()
