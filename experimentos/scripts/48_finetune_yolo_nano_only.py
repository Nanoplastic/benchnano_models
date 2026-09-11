"""Fine-tuning do YOLO26L v2 no dataset nano-only (datasets_yolo26_v2_nano/,
gerado por 47_build_nano_only_yolo_dataset.py) -- parte do checkpoint já
treinado (runs_v2/microplastic_yolo26_v2/weights/best.pt), restrito só a
anotações de partículas nano (<1000nm; ver 47), pra virar um detector
especializado em nanoplástico.

Diferenças em relação a 13_train_yolo26_corrected.py (mesma lógica de "taxa
menor pra não desestabilizar pesos já convergidos" usada no fine-tuning
tiled, 29_finetune_yolo_tiled.py):
  - lr0=1e-5 (10x menor que o treino original)
  - epochs=100 / patience=30 (orçamento de fine-tuning, não treino do zero)
  - imgsz=1024, IGUAL ao treino original (aqui não há tiling/recorte de
    imagem como em 29 -- só as anotações foram filtradas, a escala da
    imagem continua a mesma do checkpoint base, então não faz sentido mudar
    imgsz)

Não sobrescreve nada: salva em runs_v2_nano/microplastic_yolo26_nano_ft/
(pasta nova), nunca em runs_v2/microplastic_yolo26_v2/ (checkpoint original
intocado). A partição de teste travada nunca entra aqui (47 só copia
datasets_yolo26_v2_nano/images/test verbatim pra referência -- o data.yaml
usado no .train() só aponta pra train/val).

Job de GPU longo -- rode direto no terminal:
    python3 scripts/48_finetune_yolo_nano_only.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import save_experiment_summary, setup_experiment_logging

from ultralytics import YOLO

EXP_NAME = "48_finetune_yolo_nano_only"
EXP_DIR, LOG_PATH, LOGGER, RUNTIME_METADATA = setup_experiment_logging(EXP_NAME, __file__)

ROOT = Path(__file__).resolve().parent.parent
BASE_CHECKPOINT = ROOT / "runs_v2" / "microplastic_yolo26_v2" / "weights" / "best.pt"
DATA_YAML = str(ROOT / "datasets_yolo26_v2_nano" / "data.yaml")
PROJECT_DIR = str(ROOT / "runs_v2_nano")
RUN_NAME = "microplastic_yolo26_nano_ft"


def main():
    if not BASE_CHECKPOINT.exists():
        msg = f"Checkpoint base nao existe em {BASE_CHECKPOINT}. Rode 13_train_yolo26_corrected.py primeiro."
        LOGGER.error(msg)
        print(msg)
        sys.exit(1)
    if not Path(DATA_YAML).exists():
        msg = f"{DATA_YAML} nao existe. Rode 47_build_nano_only_yolo_dataset.py primeiro."
        LOGGER.error(msg)
        print(msg)
        sys.exit(1)

    LOGGER.info("=== fine-tuning YOLO26L v2 no dataset nano-only ===")
    LOGGER.info("base_checkpoint=%s data_yaml=%s", BASE_CHECKPOINT, DATA_YAML)

    model = YOLO(str(BASE_CHECKPOINT), task="detect")

    train_kwargs = dict(
        data=DATA_YAML,
        task="detect",
        epochs=100,
        imgsz=1024,
        batch=-1,
        optimizer="AdamW",
        lr0=1e-5,
        lrf=0.1,
        cos_lr=True,
        device="0",
        project=PROJECT_DIR,
        name=RUN_NAME,
        save_period=-1,
        patience=30,
        close_mosaic=10,
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
    LOGGER.info("hiperparametros=%s", train_kwargs)
    print(f"Hiperparametros: {train_kwargs}")

    model.train(**train_kwargs)

    best = Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt"
    LOGGER.info("fine-tuning concluido -- best.pt em %s", best)
    print(f"\nFine-tuning concluido. Checkpoint: {best}")

    save_experiment_summary(EXP_NAME, RUNTIME_METADATA, {
        "base_checkpoint": str(BASE_CHECKPOINT),
        "data_yaml": DATA_YAML,
        "project_dir": PROJECT_DIR,
        "run_name": RUN_NAME,
        "best_weights": str(best),
        "train_kwargs": {k: v for k, v in train_kwargs.items()},
    })
    print(f"Resumo salvo em {EXP_DIR / 'summary.json'}")


if __name__ == "__main__":
    main()
