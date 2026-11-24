"""Ajuste fino do YOLO26L só com nanopartículas / Nano-only YOLO26L fine-tuning.

PT: Ajuste fino do YOLO26L v2 no dataset nano (datasets_yolo26_v2_nano/, gerado por
    47_build_nano_only_yolo_dataset.py), partindo do checkpoint treinado
    (runs_v2/microplastic_yolo26_v2/weights/best.pt), para obter um detector especializado
    em nanoplástico.
    Diferenças em relação a 13_train_yolo26_corrected.py (taxa menor para não desestabilizar
    pesos já convergidos):
    - lr0=1e-5 (10x menor que no treino original)
    - epochs=100 / patience=30 (orçamento de ajuste fino, não de treino do zero)
    - imgsz=1024, igual ao treino original (só as anotações foram filtradas; a escala da
    imagem é a mesma do checkpoint base)
    Salva em runs_v2_nano/microplastic_yolo26_nano_ft/, sem tocar em
    runs_v2/microplastic_yolo26_v2/. A partição de teste não entra no treino (o data.yaml
    usado em .train() aponta só para train/val).
    Treino longo em GPU; rodar direto no terminal:
    python3 scripts/48_finetune_yolo_nano_only.py
EN: Fine-tunes YOLO26L v2 on the nano dataset (datasets_yolo26_v2_nano/, built by
    47_build_nano_only_yolo_dataset.py), starting from the trained checkpoint
    (runs_v2/microplastic_yolo26_v2/weights/best.pt), to obtain a nanoplastic-specialized
    detector.
    Differences from 13_train_yolo26_corrected.py (lower rate so already converged weights
    are not destabilized):
    - lr0=1e-5 (10x lower than in the original training)
    - epochs=100 / patience=30 (fine-tuning budget, not training from scratch)
    - imgsz=1024, same as the original training (only annotations were filtered; image scale
    matches the base checkpoint)
    Saves to runs_v2_nano/microplastic_yolo26_nano_ft/, leaving
    runs_v2/microplastic_yolo26_v2/ untouched. The test split is not used for training (the
    data.yaml passed to .train() points only to train/val).
    Long GPU job; run directly in a terminal:
    python3 scripts/48_finetune_yolo_nano_only.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import salvar_resumo_experimento, configurar_log_experimento

from ultralytics import YOLO

NOME_EXP = "48_finetune_yolo_nano_only"
DIR_EXP, CAMINHO_LOG, REGISTRADOR, METADADOS_EXECUCAO = configurar_log_experimento(NOME_EXP, __file__)

RAIZ = Path(__file__).resolve().parent.parent
CHECKPOINT_BASE = RAIZ / "runs_v2" / "microplastic_yolo26_v2" / "weights" / "best.pt"
YAML_DADOS = str(RAIZ / "datasets_yolo26_v2_nano" / "data.yaml")
DIR_PROJETO = str(RAIZ / "runs_v2_nano")
NOME_EXECUCAO = "microplastic_yolo26_nano_ft"


def principal():
    if not CHECKPOINT_BASE.exists():
        msg = f"Checkpoint base nao existe em {CHECKPOINT_BASE}. Rode 13_train_yolo26_corrected.py primeiro."
        REGISTRADOR.error(msg)
        print(msg)
        sys.exit(1)
    if not Path(YAML_DADOS).exists():
        msg = f"{YAML_DADOS} nao existe. Rode 47_build_nano_only_yolo_dataset.py primeiro."
        REGISTRADOR.error(msg)
        print(msg)
        sys.exit(1)

    REGISTRADOR.info("=== fine-tuning YOLO26L v2 no dataset nano-only ===")
    REGISTRADOR.info("base_checkpoint=%s data_yaml=%s", CHECKPOINT_BASE, YAML_DADOS)

    modelo = YOLO(str(CHECKPOINT_BASE), task="detect")

    kwargs_treino = dict(
        data=YAML_DADOS,
        task="detect",
        epochs=100,
        imgsz=1024,
        batch=-1,
        optimizer="AdamW",
        lr0=1e-5,
        lrf=0.1,
        cos_lr=True,
        device="0",
        project=DIR_PROJETO,
        name=NOME_EXECUCAO,
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
    REGISTRADOR.info("hiperparametros=%s", kwargs_treino)
    print(f"Hiperparametros: {kwargs_treino}")

    modelo.train(**kwargs_treino)

    melhor_item = Path(DIR_PROJETO) / NOME_EXECUCAO / "weights" / "best.pt"
    REGISTRADOR.info("fine-tuning concluido -- best.pt em %s", melhor_item)
    print(f"\nFine-tuning concluido. Checkpoint: {melhor_item}")

    salvar_resumo_experimento(NOME_EXP, METADADOS_EXECUCAO, {
        "base_checkpoint": str(CHECKPOINT_BASE),
        "data_yaml": YAML_DADOS,
        "project_dir": DIR_PROJETO,
        "run_name": NOME_EXECUCAO,
        "best_weights": str(melhor_item),
        "train_kwargs": {k: v for k, v in kwargs_treino.items()},
    })
    print(f"Resumo salvo em {DIR_EXP / 'summary.json'}")


if __name__ == "__main__":
    principal()
