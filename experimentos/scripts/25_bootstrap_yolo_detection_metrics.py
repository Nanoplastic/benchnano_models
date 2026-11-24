"""Bootstrap das métricas de detecção do YOLO / YOLO detection metrics bootstrap.

PT: Estende o bootstrap por imagem (mesma unidade de reamostragem e mesma semente de
    14_yolo_test_eval.py e 11_compare_experiments_statistical.py) para precisão, mAP@0.5 e
    mAP@0.5:0.95 agregados do YOLO26L.
    Precisão e mAP vêm do validador interno do Ultralytics (curva precisão-recall por classe
    integrada sobre a faixa de confiança). Para não divergir da estimativa pontual do
    Ultralytics, o script chama model.val() uma vez por réplica sobre uma lista de imagens
    reamostrada com reposição (arquivo .txt; o Ultralytics aceita entradas repetidas: uma
    imagem listada 3x numa reamostragem de 4 entradas produziu Instances=913 = 3*141+490, as
    contagens reais das duas imagens).
    Custo: cada model.val() sobre 15 imagens leva ~1 s, por isso N_BOOTSTRAP=2000 (não
    10000).
EN: Extends the per-image bootstrap (same resampling unit and seed as 14_yolo_test_eval.py
    and 11_compare_experiments_statistical.py) to aggregate YOLO26L precision, mAP@0.5 and
    mAP@0.5:0.95.
    Precision and mAP come from the internal Ultralytics validator (per-class
    precision-recall curve integrated over the confidence range). To stay consistent with
    the Ultralytics point estimate, the script calls model.val() once per replicate over a
    list of images resampled with replacement (a .txt file; Ultralytics accepts repeated
    entries: one image listed 3x in a 4-entry resample produced Instances=913 = 3*141+490,
    the actual counts of the two images).
    Cost: each model.val() over 15 images takes ~1 s, hence N_BOOTSTRAP=2000 (not 10000).
"""
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import salvar_resumo_experimento, configurar_log_experimento

from ultralytics import YOLO

NOME_EXP = "25_bootstrap_yolo_detection_metrics"
DIR_EXP, CAMINHO_LOG, REGISTRADOR, METADADOS_EXECUCAO = configurar_log_experimento(NOME_EXP, __file__)

RAIZ = Path(__file__).resolve().parent.parent
DIR_TESTE = RAIZ / "datasets_yolo26_v2"
CAMINHO_MODELO = RAIZ / "runs_v2" / "microplastic_yolo26_v2" / "weights" / "best.pt"
NOMES_CLASSES = ["PE", "PET", "PP", "PS"]

TAMANHO_IMG = 1024
LIMIAR_CONF = 0.15
N_BOOTSTRAP = 2000
SEMENTE = 42

DIR_BOOTSTRAP = DIR_TESTE / "_bootstrap_lists"


def principal():
    if not CAMINHO_MODELO.exists():
        print(f"Checkpoint nao existe em {CAMINHO_MODELO}")
        sys.exit(1)

    REGISTRADOR.info("=== bootstrap por imagem para precisao/mAP50/mAP50-95 (N=%d) ===", N_BOOTSTRAP)
    modelo = YOLO(str(CAMINHO_MODELO))

    caminhos_img = sorted((DIR_TESTE / "images/test").glob("*.png"))
    n_imagens = len(caminhos_img)
    REGISTRADOR.info("n_imagens=%d", n_imagens)

    DIR_BOOTSTRAP.mkdir(exist_ok=True)
    caminho_lista = DIR_BOOTSTRAP / "resample.txt"
    caminho_yaml = DIR_BOOTSTRAP / "resample.yaml"
    caminho_yaml.write_text(
        f"train: {caminho_lista.resolve()}\n"
        f"val: {caminho_lista.resolve()}\n"
        f"nc: {len(NOMES_CLASSES)}\n"
        "names:\n" + "\n".join(f"  - {c}" for c in NOMES_CLASSES) + "\n"
    )

    gerador = np.random.default_rng(SEMENTE)
    precisoes = np.empty(N_BOOTSTRAP)
    recalls = np.empty(N_BOOTSTRAP)
    map50s = np.empty(N_BOOTSTRAP)
    map5095s = np.empty(N_BOOTSTRAP)

    t_inicio = time.time()
    for b in range(N_BOOTSTRAP):
        indice = gerador.integers(0, n_imagens, size=n_imagens)
        amostrados = [caminhos_img[i] for i in indice]
        caminho_lista.write_text("\n".join(str(p.resolve()) for p in amostrados))

        # PT: o cache do Ultralytics é invalidado a cada réplica (a lista de imagens muda)
        # EN: the Ultralytics cache is invalidated on every replicate (the image list
        #     changes)
        arquivo_cache = (DIR_TESTE / "labels" / "test.cache")
        if arquivo_cache.exists():
            arquivo_cache.unlink()

        metricas = modelo.val(data=str(caminho_yaml), split="val", imgsz=TAMANHO_IMG,
                             conf=LIMIAR_CONF, iou=0.7, plots=False, verbose=False)
        precisoes[b] = float(metricas.box.mp)
        recalls[b] = float(metricas.box.mr)
        map50s[b] = float(metricas.box.map50)
        map5095s[b] = float(metricas.box.map)

        if (b + 1) % 50 == 0:
            decorrido = time.time() - t_inicio
            REGISTRADOR.info("replica %d/%d concluida (%.1fs decorridos, %.2fs/replica)",
                        b + 1, N_BOOTSTRAP, decorrido, decorrido / (b + 1))

    resultados = {}
    for nome, matriz in [("precision", precisoes), ("recall", recalls),
                       ("map50", map50s), ("map50_95", map5095s)]:
        ic_inf, ic_sup = np.percentile(matriz, [2.5, 97.5])
        resultados[nome] = {"mean_over_replicas": float(matriz.mean()), "ci95_low": float(ic_inf), "ci95_high": float(ic_sup)}
        print(f"{nome:10s}: media_replicas={matriz.mean():.4f}  IC95%=[{ic_inf:.4f}, {ic_sup:.4f}]")
        REGISTRADOR.info("%s: media_replicas=%.4f ic95_low=%.4f ic95_high=%.4f", nome, matriz.mean(), ic_inf, ic_sup)

    # PT: restaura o diretório de teste (remove lista/yaml/cache temporários e força a
    #     regeneração do cache na próxima avaliação completa)
    # EN: restores the test directory (removes the temporary list/yaml/cache and forces
    #     cache regeneration on the next full evaluation)
    shutil.rmtree(DIR_BOOTSTRAP, ignore_errors=True)
    arquivo_cache = (DIR_TESTE / "labels" / "test.cache")
    if arquivo_cache.exists():
        arquivo_cache.unlink()

    salvar_resumo_experimento(NOME_EXP, METADADOS_EXECUCAO, {
        "model_path": str(CAMINHO_MODELO),
        "n_test_images": n_imagens,
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEMENTE,
        "conf_threshold": LIMIAR_CONF,
        "point_estimates_from_14_yolo_test_eval": {
            "precision": 0.6069422606876318, "recall": 0.46682537100157057,
            "map50": 0.28024896067965005, "map50_95": 0.08606379406840536,
        },
        "bootstrap_ci95": resultados,
    })
    print(f"\nLog completo: {CAMINHO_LOG}")


if __name__ == "__main__":
    principal()
