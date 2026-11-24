"""Treino de uma dobra nano da validação cruzada / Training of one nano cross-validation fold.

PT: Treina UMA dobra da validação cruzada só com nanopartículas (mesmas dobras e semente de
    49_build_nano_cv_folds.py, caixas de treino/val já filtradas para partículas < 1000 nm),
    com os mesmos hiperparâmetros de treino do zero de 44_train_cv_fold.py; mudam só o
    data.yaml (aponta para a dobra nano) e o diretório de saída (runs_v2_cv_nano/foldK/).
    Rodar uma vez por dobra, em sequência (depois de 49_build_nano_cv_folds.py):
    python3 scripts/50_train_cv_fold_nano.py --fold 0
    python3 scripts/50_train_cv_fold_nano.py --fold 1
    python3 scripts/50_train_cv_fold_nano.py --fold 2
    python3 scripts/50_train_cv_fold_nano.py --fold 3
    python3 scripts/50_train_cv_fold_nano.py --fold 4
    Cada dobra treina do zero a partir do checkpoint pré-treinado yolo26l.pt (não é ajuste
    fino).
EN: Trains ONE nano-only cross-validation fold (same folds and seed as
    49_build_nano_cv_folds.py, train/val boxes already filtered to < 1000 nm particles),
    with the same from-scratch hyperparameters as 44_train_cv_fold.py; only the data.yaml
    (points to the nano fold) and the output directory (runs_v2_cv_nano/foldK/) change.
    Run once per fold, in sequence (after 49_build_nano_cv_folds.py):
    python3 scripts/50_train_cv_fold_nano.py --fold 0
    python3 scripts/50_train_cv_fold_nano.py --fold 1
    python3 scripts/50_train_cv_fold_nano.py --fold 2
    python3 scripts/50_train_cv_fold_nano.py --fold 3
    python3 scripts/50_train_cv_fold_nano.py --fold 4
    Each fold trains from scratch from the pretrained yolo26l.pt checkpoint (not
    fine-tuning).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import salvar_resumo_experimento, configurar_log_experimento

from ultralytics import YOLO

RAIZ = Path(__file__).resolve().parent.parent
DIR_DATASETS_CV = RAIZ / "datasets_yolo26_v2_cv_nano"
DIR_PROJETO = str(RAIZ / "runs_v2_cv_nano")


def principal():
    analisador = argparse.ArgumentParser(description=__doc__)
    analisador.add_argument("--fold", type=int, required=True, help="indice do fold (0..4)")
    analisador.add_argument("--seed", type=int, default=0,
                         help="seed do ultralytics (default=0, mesmo default usado na CV original "
                              "e nos outros folds). Só passe um valor diferente pra testar "
                              "explicitamente sensibilidade ao seed=0 especifico.")
    args = analisador.parse_args()

    dir_dobra = DIR_DATASETS_CV / f"fold{args.fold}"
    yaml_dados = dir_dobra / "data.yaml"
    if not yaml_dados.exists():
        print(f"ERRO: {yaml_dados} nao existe -- rode 49_build_nano_cv_folds.py primeiro.")
        sys.exit(1)

    nome_execucao = f"fold{args.fold}" if args.seed == 0 else f"fold{args.fold}_seed{args.seed}"
    nome_exp = f"50_train_cv_fold_nano{args.fold}" if args.seed == 0 else f"50_train_cv_fold_nano{args.fold}_seed{args.seed}"
    dir_exp, caminho_log, registrador, metadados_execucao = configurar_log_experimento(nome_exp, __file__)
    registrador.info("=== treino cross-validation nano-only fold=%d ===", args.fold)
    registrador.info("data_yaml=%s", yaml_dados)

    modelo = YOLO(str(RAIZ / "yolo26l.pt"), task="detect")

    # PT: mesmos hiperparâmetros de treino do zero de 44_train_cv_fold.py; mudam só
    #     data/project/name (apontam para a dobra nano)
    # EN: same from-scratch hyperparameters as 44_train_cv_fold.py; only data/project/name
    #     change (they point to the nano fold)
    kwargs_treino = dict(
        data=str(yaml_dados),
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
        project=DIR_PROJETO,
        name=nome_execucao,
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
    registrador.info("hiperparametros=%s", kwargs_treino)
    print(f"Fold {args.fold} (nano-only) -- hiperparametros: {kwargs_treino}")

    modelo.train(**kwargs_treino)

    melhor_item = Path(DIR_PROJETO) / nome_execucao / "weights" / "best.pt"
    registrador.info("treino do fold %d (nano) concluido -- best.pt em %s", args.fold, melhor_item)

    salvar_resumo_experimento(nome_exp, metadados_execucao, {
        "fold": args.fold,
        "data_yaml": str(yaml_dados),
        "best_weights": str(melhor_item),
        "results_csv": str(Path(DIR_PROJETO) / nome_execucao / "results.csv"),
        "train_kwargs": kwargs_treino,
    })
    print(f"\nFold {args.fold} (nano) concluido. Pesos: {melhor_item}")
    print(f"Log completo: {caminho_log}")
    print(f"Curva de treino: {Path(DIR_PROJETO) / nome_execucao / 'results.csv'}")


if __name__ == "__main__":
    principal()
