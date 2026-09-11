"""Treina UM fold da cross-validation (item 3 do plano de ação), com os
MESMOS hiperparâmetros de 13_train_yolo26_corrected.py -- só muda o
data.yaml (aponta pro fold) e o diretório de saída (runs_v2_cv/foldK/, não
sobrescreve runs_v2/microplastic_yolo26_v2/ original).

Rodar uma vez por fold, sequencialmente (script 43_build_cv_folds.py precisa
ter rodado antes):

    python3 scripts/44_train_cv_fold.py --fold 0
    python3 scripts/44_train_cv_fold.py --fold 1
    python3 scripts/44_train_cv_fold.py --fold 2
    python3 scripts/44_train_cv_fold.py --fold 3
    python3 scripts/44_train_cv_fold.py --fold 4

Cada um treina do zero a partir do checkpoint pré-treinado yolo26l.pt
(igual ao original, não é fine-tuning), ~80-90 minutos cada nesta GPU
(baseado no treino original: 351 epocas em 5030s).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import save_experiment_summary, setup_experiment_logging

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent
CV_DATASETS_DIR = ROOT / "datasets_yolo26_v2_cv"
PROJECT_DIR = str(ROOT / "runs_v2_cv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True, help="indice do fold (0..4)")
    parser.add_argument("--seed", type=int, default=0,
                         help="seed do ultralytics (default=0, o mesmo default usado no treino "
                              "original e nos outros folds -- treino do ultralytics e' "
                              "deterministico com seed+deterministic fixos, entao rodar de novo "
                              "com seed=0 reproduz o MESMO resultado bit-a-bit. Só passe um valor "
                              "diferente pra testar explicitamente se um fold que treinou mal e' "
                              "sensibilidade real ao split ou artefato do seed=0 especifico.")
    args = parser.parse_args()

    fold_dir = CV_DATASETS_DIR / f"fold{args.fold}"
    data_yaml = fold_dir / "data.yaml"
    if not data_yaml.exists():
        print(f"ERRO: {data_yaml} nao existe -- rode 43_build_cv_folds.py primeiro.")
        sys.exit(1)

    # nome de run diferente quando seed != 0, pra NUNCA sobrescrever o
    # resultado com seed=0 (default, usado pelos outros folds) -- mantem os
    # dois lado a lado pra comparacao/auditoria.
    run_name = f"fold{args.fold}" if args.seed == 0 else f"fold{args.fold}_seed{args.seed}"
    exp_name = f"44_train_cv_fold{args.fold}" if args.seed == 0 else f"44_train_cv_fold{args.fold}_seed{args.seed}"
    exp_dir, log_path, logger, runtime_metadata = setup_experiment_logging(exp_name, __file__)
    logger.info("=== treino cross-validation fold=%d ===", args.fold)
    logger.info("data_yaml=%s", data_yaml)

    model = YOLO(str(ROOT / "yolo26l.pt"), task="detect")

    # MESMOS hiperparametros de 13_train_yolo26_corrected.py -- so muda
    # data/project/name.
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
    print(f"Fold {args.fold} -- hiperparametros: {train_kwargs}")

    model.train(**train_kwargs)

    best = Path(PROJECT_DIR) / run_name / "weights" / "best.pt"
    logger.info("treino do fold %d concluido -- best.pt em %s", args.fold, best)

    save_experiment_summary(exp_name, runtime_metadata, {
        "fold": args.fold,
        "data_yaml": str(data_yaml),
        "best_weights": str(best),
        "results_csv": str(Path(PROJECT_DIR) / run_name / "results.csv"),
        "train_kwargs": train_kwargs,
    })
    print(f"\nFold {args.fold} concluido. Pesos: {best}")
    print(f"Log completo: {log_path}")
    print(f"Curva de treino: {Path(PROJECT_DIR) / run_name / 'results.csv'}")


if __name__ == "__main__":
    main()
