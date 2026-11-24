"""Treino do detector YOLO26L (v2) / YOLO26L detector training (v2).

PT: Treina o YOLO26L de detecção de partículas com duas garantias:
    1. Sem vazamento: `datasets_yolo26_v2/` contém em treino/val só as imagens marcadas
    `train`/`val` em resultados/00_split_images/data/image_splits.csv. As 15 imagens
    marcadas `test` ficam em datasets_yolo26_v2/images/test e só são usadas por
    14_yolo_test_eval.py.
    2. Otimizador explícito: `optimizer="AdamW"`. Com o padrão do Ultralytics
    (`optimizer='auto'`), o `lr0` informado é ignorado e o otimizador e a taxa de
    aprendizado são escolhidos automaticamente (AdamW com lr=0,00125); com o otimizador
    explícito, lr0=1e-4 é respeitado.
    Os demais hiperparâmetros (aumento de dados, paciência etc.) seguem a receita de treino
    do detector de referência.
EN: Trains the YOLO26L particle detector with two guarantees:
    1. No leakage: `datasets_yolo26_v2/` holds in train/val only the images marked
    `train`/`val` in resultados/00_split_images/data/image_splits.csv. The 15 images marked
    `test` are kept in datasets_yolo26_v2/images/test and are only used by
    14_yolo_test_eval.py.
    2. Explicit optimizer: `optimizer="AdamW"`. With the Ultralytics default
    (`optimizer='auto'`), the given `lr0` is ignored and the optimizer and learning rate are
    chosen automatically (AdamW with lr=0.00125); with an explicit optimizer, lr0=1e-4 is
    honoured.
    The remaining hyperparameters (augmentation, patience, etc.) follow the reference
    detector training recipe.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import salvar_resumo_experimento, configurar_log_experimento

from ultralytics import YOLO

NOME_EXP = "13_train_yolo26_corrected"
DIR_EXP, CAMINHO_LOG, REGISTRADOR, METADADOS_EXECUCAO = configurar_log_experimento(NOME_EXP, __file__)

RAIZ = Path(__file__).resolve().parent.parent
YAML_DADOS = str(RAIZ / "datasets_yolo26_v2" / "data.yaml")
DIR_PROJETO = str(RAIZ / "runs_v2")
NOME_EXECUCAO = "microplastic_yolo26_v2"


def principal():
    REGISTRADOR.info("=== retreino YOLO26L v2 (split sem vazamento + optimizer=AdamW explicito) ===")
    REGISTRADOR.info("data_yaml=%s", YAML_DADOS)

    modelo = YOLO(str(RAIZ / "yolo26l.pt"), task="detect")

    kwargs_treino = dict(
        data=YAML_DADOS,
        task="detect",
        epochs=400,  # PT: teto de épocas; com lr0=1e-4 as perdas ainda caem após 150 épocas, e patience=100 interrompe ao estabilizar / EN: epoch cap; with lr0=1e-4 the losses still decrease after 150 epochs, and patience=100 stops once they plateau
        imgsz=1024,
        batch=-1,
        optimizer="AdamW",  # PT: explícito: com 'auto' o Ultralytics ignora lr0 / EN: explicit: with 'auto' Ultralytics ignores lr0
        lr0=1e-4,
        lrf=0.1,
        cos_lr=True,
        device="0",
        project=DIR_PROJETO,
        name=NOME_EXECUCAO,
        save_period=-1,  # PT: salva só best.pt/last.pt (sem checkpoint por época) / EN: saves only best.pt/last.pt (no per-epoch checkpoint)
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
    REGISTRADOR.info("hiperparametros=%s", kwargs_treino)
    print(f"Hiperparametros: {kwargs_treino}")

    resultados = modelo.train(**kwargs_treino)

    melhor_item = Path(DIR_PROJETO) / NOME_EXECUCAO / "weights" / "best.pt"
    REGISTRADOR.info("treino concluido -- best.pt em %s", melhor_item)

    salvar_resumo_experimento(NOME_EXP, METADADOS_EXECUCAO, {
        "data_yaml": YAML_DADOS,
        "best_weights": str(melhor_item),
        "results_csv": str(Path(DIR_PROJETO) / NOME_EXECUCAO / "results.csv"),
        "train_kwargs": kwargs_treino,
    })
    print(f"\nTreino concluido. Pesos finais: {melhor_item}")
    print(f"Log completo (com todos os hiperparametros e metadados do ambiente): {CAMINHO_LOG}")
    print(f"Curva de treino / metricas por epoca: {Path(DIR_PROJETO) / NOME_EXECUCAO / 'results.csv'}")


if __name__ == "__main__":
    principal()
