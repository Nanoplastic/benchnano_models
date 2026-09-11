"""Retreino do YOLO26L (v2) corrigindo dois problemas achados na run do
colega (microplastic_yolo26/, commit 59e0c39):

1. VAZAMENTO: o `scripts_thiago/main.py` original gerava train/val com split
   aleatório 80/20 por classe, sem nunca reservar uma partição de teste. Ao
   cruzar a lista de imagens dele com o split por imagem-fonte deste projeto
   (00_split_images.py), 90 das 102 imagens do nosso split inteiro (88%) já
   estavam no train+val dele -- incluindo 13 das 15 imagens do nosso `test`.
   Qualquer avaliação anterior nessas imagens não media generalização.

   Correção: `datasets_yolo26_v2/` foi reconstruído do zero usando SÓ as
   imagens marcadas `train`/`val` em resultados/00_split_images/data/
   image_splits.csv; as 15 imagens marcadas `test` nunca entram aqui --
   ficam em datasets_yolo26_v2/images/test, usadas só por
   14_yolo_test_eval.py, uma vez.

2. OTIMIZADOR: `scripts_thiago/main.py` nunca define `optimizer=` no
   `model.train(...)`, então o Ultralytics usa o default `optimizer='auto'`,
   que IGNORA silenciosamente o `lr0=1e-4` documentado no Methods do artigo e
   escolhe sozinho `AdamW(lr=0.00125, ...)` -- confirmado reproduzindo a
   mesma run aqui e vendo o aviso do próprio Ultralytics. Ou seja, o `main.tex`
   afirma uma taxa de aprendizado que nunca foi de fato usada.

   Correção: `optimizer="AdamW"` explícito abaixo, para que lr0=1e-4 seja
   realmente respeitado.

Resto dos hiperparâmetros é idêntico ao `scripts_thiago/main.py` original
(mesmos valores de augmentação, épocas, paciência etc.), para manter a
receita o mais próxima possível da run original, mudando só o que estava
errado.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import save_experiment_summary, setup_experiment_logging

from ultralytics import YOLO

EXP_NAME = "13_train_yolo26_corrected"
EXP_DIR, LOG_PATH, LOGGER, RUNTIME_METADATA = setup_experiment_logging(EXP_NAME, __file__)

ROOT = Path(__file__).resolve().parent.parent
DATA_YAML = str(ROOT / "datasets_yolo26_v2" / "data.yaml")
PROJECT_DIR = str(ROOT / "runs_v2")
RUN_NAME = "microplastic_yolo26_v2"


def main():
    LOGGER.info("=== retreino YOLO26L v2 (split sem vazamento + optimizer=AdamW explicito) ===")
    LOGGER.info("data_yaml=%s", DATA_YAML)

    model = YOLO(str(ROOT / "yolo26l.pt"), task="detect")

    train_kwargs = dict(
        data=DATA_YAML,
        task="detect",
        epochs=400,  # 150 nao foi suficiente p/ convergir com lr0=1e-4 (box_loss/cls_loss ainda
                     # caindo na epoca 150, ver 1a rodada) -- patience=100 abaixo para de verdade
                     # quando platear, entao 400 e so um teto generoso, nao necessariamente vai
                     # rodar todas
        imgsz=1024,
        batch=-1,
        optimizer="AdamW",  # CORRIGIDO -- era 'auto' (default) no script original, que ignora lr0
        lr0=1e-4,
        lrf=0.1,
        cos_lr=True,
        device="0",
        project=PROJECT_DIR,
        name=RUN_NAME,
        save_period=-1,  # so salva best.pt/last.pt; original usava 1 (150 checkpoints, ~7GB à toa)
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
    LOGGER.info("hiperparametros=%s", train_kwargs)
    print(f"Hiperparametros: {train_kwargs}")

    results = model.train(**train_kwargs)

    best = Path(PROJECT_DIR) / RUN_NAME / "weights" / "best.pt"
    LOGGER.info("treino concluido -- best.pt em %s", best)

    save_experiment_summary(EXP_NAME, RUNTIME_METADATA, {
        "data_yaml": DATA_YAML,
        "best_weights": str(best),
        "results_csv": str(Path(PROJECT_DIR) / RUN_NAME / "results.csv"),
        "train_kwargs": train_kwargs,
    })
    print(f"\nTreino concluido. Pesos finais: {best}")
    print(f"Log completo (com todos os hiperparametros e metadados do ambiente): {LOG_PATH}")
    print(f"Curva de treino / metricas por epoca: {Path(PROJECT_DIR) / RUN_NAME / 'results.csv'}")


if __name__ == "__main__":
    main()
