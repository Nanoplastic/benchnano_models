"""Estende o bootstrap por imagem (mesma unidade de reamostragem e mesma
semente de 14_yolo_test_eval.py / 11_compare_experiments_statistical.py) para
precisao, mAP@0.5 e mAP@0.5:0.95 -- as metricas "Agregado" da Tabela de
resultados do YOLO26L, que ate agora so tinham CI para o recall casado por
tamanho (script 14).

Diferente do bootstrap de recall (que reusa contagens de casamento por IoU ja
computadas), precisao e mAP sao produzidas pelo validador interno do
Ultralytics (curva precisao-recall por classe, integrada sobre o range de
confianca). Reimplementar essa curva fora do Ultralytics arriscaria divergir
do ponto estimado ja publicado (0,607 / 0,280 / 0,086). Em vez disso, este
script chama model.val() uma vez POR REPLICA, sobre uma lista de imagens
reamostrada com reposicao (arquivo .txt, formato padrao do Ultralytics que
aceita entradas repetidas -- validado manualmente: uma imagem listada 3x em
uma reamostragem de 4 entradas produziu Instances=913, exatamente 3*141+490,
as contagens reais das duas imagens envolvidas).

Custo computacional: cada chamada a model.val() sobre 15 imagens leva ~1s
(dataloader + inferencia), entao usamos N_BOOTSTRAP=2000 em vez de 10000 --
reportado explicitamente como tal no texto, nao arredondado pra 10000.
"""
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import save_experiment_summary, setup_experiment_logging

from ultralytics import YOLO

EXP_NAME = "25_bootstrap_yolo_detection_metrics"
EXP_DIR, LOG_PATH, LOGGER, RUNTIME_METADATA = setup_experiment_logging(EXP_NAME, __file__)

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "datasets_yolo26_v2"
MODEL_PATH = ROOT / "runs_v2" / "microplastic_yolo26_v2" / "weights" / "best.pt"
CLASS_NAMES = ["PE", "PET", "PP", "PS"]

IMG_SIZE = 1024
CONF_THRESHOLD = 0.15
N_BOOTSTRAP = 2000
SEED = 42

BOOTSTRAP_DIR = TEST_DIR / "_bootstrap_lists"


def main():
    if not MODEL_PATH.exists():
        print(f"Checkpoint nao existe em {MODEL_PATH}")
        sys.exit(1)

    LOGGER.info("=== bootstrap por imagem para precisao/mAP50/mAP50-95 (N=%d) ===", N_BOOTSTRAP)
    model = YOLO(str(MODEL_PATH))

    img_paths = sorted((TEST_DIR / "images/test").glob("*.png"))
    n_images = len(img_paths)
    LOGGER.info("n_imagens=%d", n_images)

    BOOTSTRAP_DIR.mkdir(exist_ok=True)
    list_path = BOOTSTRAP_DIR / "resample.txt"
    yaml_path = BOOTSTRAP_DIR / "resample.yaml"
    yaml_path.write_text(
        f"train: {list_path.resolve()}\n"
        f"val: {list_path.resolve()}\n"
        f"nc: {len(CLASS_NAMES)}\n"
        "names:\n" + "\n".join(f"  - {c}" for c in CLASS_NAMES) + "\n"
    )

    rng = np.random.default_rng(SEED)
    precisions = np.empty(N_BOOTSTRAP)
    recalls = np.empty(N_BOOTSTRAP)
    map50s = np.empty(N_BOOTSTRAP)
    map5095s = np.empty(N_BOOTSTRAP)

    t_start = time.time()
    for b in range(N_BOOTSTRAP):
        idx = rng.integers(0, n_images, size=n_images)
        sampled = [img_paths[i] for i in idx]
        list_path.write_text("\n".join(str(p.resolve()) for p in sampled))

        # cache do Ultralytics invalidada a cada replica (lista de imagens muda)
        cache_file = (TEST_DIR / "labels" / "test.cache")
        if cache_file.exists():
            cache_file.unlink()

        metrics = model.val(data=str(yaml_path), split="val", imgsz=IMG_SIZE,
                             conf=CONF_THRESHOLD, iou=0.7, plots=False, verbose=False)
        precisions[b] = float(metrics.box.mp)
        recalls[b] = float(metrics.box.mr)
        map50s[b] = float(metrics.box.map50)
        map5095s[b] = float(metrics.box.map)

        if (b + 1) % 50 == 0:
            elapsed = time.time() - t_start
            LOGGER.info("replica %d/%d concluida (%.1fs decorridos, %.2fs/replica)",
                        b + 1, N_BOOTSTRAP, elapsed, elapsed / (b + 1))

    results = {}
    for name, arr in [("precision", precisions), ("recall", recalls),
                       ("map50", map50s), ("map50_95", map5095s)]:
        ci_low, ci_high = np.percentile(arr, [2.5, 97.5])
        results[name] = {"mean_over_replicas": float(arr.mean()), "ci95_low": float(ci_low), "ci95_high": float(ci_high)}
        print(f"{name:10s}: media_replicas={arr.mean():.4f}  IC95%=[{ci_low:.4f}, {ci_high:.4f}]")
        LOGGER.info("%s: media_replicas=%.4f ic95_low=%.4f ic95_high=%.4f", name, arr.mean(), ci_low, ci_high)

    # restaura o estado do diretorio de teste (remove lista/yaml/cache temporarios,
    # forca regeneracao do cache oficial na proxima avaliacao "de verdade")
    shutil.rmtree(BOOTSTRAP_DIR, ignore_errors=True)
    cache_file = (TEST_DIR / "labels" / "test.cache")
    if cache_file.exists():
        cache_file.unlink()

    save_experiment_summary(EXP_NAME, RUNTIME_METADATA, {
        "model_path": str(MODEL_PATH),
        "n_test_images": n_images,
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "conf_threshold": CONF_THRESHOLD,
        "point_estimates_from_14_yolo_test_eval": {
            "precision": 0.6069422606876318, "recall": 0.46682537100157057,
            "map50": 0.28024896067965005, "map50_95": 0.08606379406840536,
        },
        "bootstrap_ci95": results,
    })
    print(f"\nLog completo: {LOG_PATH}")


if __name__ == "__main__":
    main()
