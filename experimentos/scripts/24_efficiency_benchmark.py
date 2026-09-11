"""Benchmark de latencia e memoria para a Secao de Eficiencia Computacional
(sec:results_efficiency / sec:efficiency). Mede, separadamente:

1. Latencia do YOLO26L por imagem de MEV completa (15 imagens do teste
   travado, datasets_yolo26_v2/images/test), com REPS repeticoes por imagem
   apos aquecimento.
2. Latencia do SmolVLM (base + adaptador LoRA size_lora, mesmo checkpoint do
   pipeline fim-a-fim em 20_predicted_box_and_end_to_end.py) por particula
   candidata, sobre uma amostra de recortes reais extraidos do teste.
3. Latencia do pipeline completo por imagem (YOLO + SmolVLM em todas as
   deteccoes acima do limiar de confianca), medida direto (nao estimada por
   soma), e pico de memoria do acelerador por fase.

Sincronizacao CUDA (torch.cuda.synchronize()) e feita nos limites de medicao
porque a execucao do PyTorch/CUDA e assincrona. reset_peak_memory_stats()
zera o contador antes de cada fase.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import (
    CHECKPOINTS_DIR, DEVICE, MODEL_ID, SIZE_CLASSIFY_PROMPT,
    classify_image, load_base_model, save_experiment_summary,
    setup_experiment_logging,
)

from peft import PeftModel
from ultralytics import YOLO

EXP_NAME = "24_efficiency_benchmark"
EXP_DIR, LOG_PATH, LOGGER, RUNTIME_METADATA = setup_experiment_logging(EXP_NAME, __file__)

ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = ROOT / "datasets_yolo26_v2"
YOLO_MODEL_PATH = ROOT / "runs_v2" / "microplastic_yolo26_v2" / "weights" / "best.pt"
SIZE_LORA_ADAPTER = CHECKPOINTS_DIR / "size_lora"

IMG_SIZE = 1024
CONF_THRESHOLD = 0.15
CROP_MARGIN_PX = 10
N_WARMUP = 3
N_REPS_YOLO = 5
N_REPS_SMOLVLM = 5
N_SMOLVLM_CROPS = 30  # particulas candidatas distintas usadas p/ medir latencia por particula
SEED = 42


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main():
    rng = np.random.default_rng(SEED)
    img_paths = sorted((TEST_DIR / "images/test").glob("*.png"))
    LOGGER.info("n_imagens_teste=%d", len(img_paths))

    # ------------------------------------------------------------------
    # Fase 1: YOLO26L, latencia por imagem
    # ------------------------------------------------------------------
    yolo_model = YOLO(str(YOLO_MODEL_PATH))

    # aquecimento: N_WARMUP inferencias descartadas (compilacao CUDA/cuDNN,
    # alocacao de cache) antes de medir
    for img_path in img_paths[:N_WARMUP]:
        yolo_model.predict(source=str(img_path), imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)
    sync()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    yolo_times_ms = []
    all_detections = []  # (img_path, xyxy_norm, cls_id) para reuso na Fase 3
    for img_path in img_paths:
        reps = []
        dets = None
        for r in range(N_REPS_YOLO):
            sync()
            t0 = time.perf_counter()
            pred = yolo_model.predict(source=str(img_path), imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]
            sync()
            t1 = time.perf_counter()
            reps.append((t1 - t0) * 1000.0)
            if r == 0:
                dets = pred
        yolo_times_ms.extend(reps)
        n_candidates = len(dets.boxes) if dets.boxes is not None else 0
        all_detections.append((img_path, dets, n_candidates))
        LOGGER.info("yolo imagem=%s reps_ms=%s n_candidatos=%d", img_path.name, [f"{x:.1f}" for x in reps], n_candidates)

    yolo_peak_mem_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else None
    yolo_times_ms = np.array(yolo_times_ms)
    avg_candidates_per_image = float(np.mean([n for _, _, n in all_detections]))

    print(f"YOLO26L por imagem: media={yolo_times_ms.mean():.1f}ms desvio_padrao={yolo_times_ms.std(ddof=1):.1f}ms "
          f"(n={len(yolo_times_ms)} medicoes = {len(img_paths)} imagens x {N_REPS_YOLO} repeticoes)")
    print(f"Candidatos medios por imagem (conf>={CONF_THRESHOLD}): {avg_candidates_per_image:.1f}")
    if yolo_peak_mem_mb is not None:
        print(f"Pico de memoria do acelerador (YOLO26L): {yolo_peak_mem_mb:.0f} MB")

    del yolo_model

    # ------------------------------------------------------------------
    # Fase 2: SmolVLM, latencia por particula candidata
    # ------------------------------------------------------------------
    processor, base_model = load_base_model(MODEL_ID)
    vlm_model = PeftModel.from_pretrained(base_model, str(SIZE_LORA_ADAPTER)).to(DEVICE)
    vlm_model.eval()

    # monta ate N_SMOLVLM_CROPS recortes reais a partir das deteccoes da Fase 1
    crops = []
    for img_path, dets, n_candidates in all_detections:
        if n_candidates == 0:
            continue
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            W, H = im.size
            xyxy = dets.boxes.xyxy.cpu().numpy()
            for box in xyxy:
                x1, y1, x2, y2 = box
                x1 = max(0, int(x1) - CROP_MARGIN_PX)
                y1 = max(0, int(y1) - CROP_MARGIN_PX)
                x2 = min(W, int(x2) + CROP_MARGIN_PX)
                y2 = min(H, int(y2) + CROP_MARGIN_PX)
                crops.append(im.crop((x1, y1, x2, y2)).copy())
                if len(crops) >= N_SMOLVLM_CROPS:
                    break
        if len(crops) >= N_SMOLVLM_CROPS:
            break
    LOGGER.info("n_recortes_smolvlm=%d", len(crops))

    for crop in crops[:N_WARMUP]:
        classify_image(processor, vlm_model, crop, prompt=SIZE_CLASSIFY_PROMPT)
    sync()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    smolvlm_times_ms = []
    for crop in crops:
        for _r in range(N_REPS_SMOLVLM):
            sync()
            t0 = time.perf_counter()
            classify_image(processor, vlm_model, crop, prompt=SIZE_CLASSIFY_PROMPT)
            sync()
            t1 = time.perf_counter()
            smolvlm_times_ms.append((t1 - t0) * 1000.0)

    smolvlm_peak_mem_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else None
    smolvlm_times_ms = np.array(smolvlm_times_ms)
    print(f"\nSmolVLM por particula: media={smolvlm_times_ms.mean():.1f}ms desvio_padrao={smolvlm_times_ms.std(ddof=1):.1f}ms "
          f"(n={len(smolvlm_times_ms)} medicoes = {len(crops)} particulas x {N_REPS_SMOLVLM} repeticoes)")
    if smolvlm_peak_mem_mb is not None:
        print(f"Pico de memoria do acelerador (SmolVLM): {smolvlm_peak_mem_mb:.0f} MB")

    # ------------------------------------------------------------------
    # Fase 3: latencia do pipeline completo por imagem, POR COMPOSICAO --
    # nao por medicao direta. Uma medicao direta rodaria o SmolVLM uma vez
    # por candidato de cada imagem amostrada (ate ~300 chamadas/imagem, ver
    # n_candidatos da Fase 1); com N_REPS_YOLO=5 repeticoes isso passaria de
    # 3000 chamadas ao SmolVLM soh nesta fase, tornando o benchmark
    # impraticavel. Em vez disso, a latencia do pipeline por imagem eh
    # composta a partir das duas fases ja medidas (mesmo raciocinio do texto
    # do artigo: "o tempo total depende do numero de candidatos do detector
    # em cada imagem"): latencia_pipeline = latencia_YOLO_media +
    # candidatos_da_imagem * latencia_SmolVLM_media_por_particula. A
    # incerteza eh propagada assumindo independencia entre as duas fases
    # (variancia da soma = soma das variancias, com var(k*X) = k^2*var(X)).
    # A memoria de pico do pipeline completo eh o maximo entre as duas fases
    # medidas (elas nao rodam simultaneamente no pipeline real: SmolVLM soh
    # processa os candidatos DEPOIS que o YOLO termina a imagem).
    # ------------------------------------------------------------------
    yolo_mean = float(yolo_times_ms.mean())
    yolo_var_of_mean = float(yolo_times_ms.var(ddof=1) / len(yolo_times_ms))
    smolvlm_mean = float(smolvlm_times_ms.mean())
    smolvlm_var_of_mean = float(smolvlm_times_ms.var(ddof=1) / len(smolvlm_times_ms))

    per_image_pipeline_ms = []
    for _img_path, _dets, n_candidates in all_detections:
        per_image_pipeline_ms.append(yolo_mean + n_candidates * smolvlm_mean)
    per_image_pipeline_ms = np.array(per_image_pipeline_ms)

    pipeline_mean = float(per_image_pipeline_ms.mean())
    # propagacao de incerteza pra UMA imagem com o numero medio de candidatos
    k = avg_candidates_per_image
    pipeline_std_composed = float(np.sqrt(yolo_var_of_mean + (k ** 2) * smolvlm_var_of_mean))
    pipeline_peak_mem_mb = max(
        m for m in (yolo_peak_mem_mb, smolvlm_peak_mem_mb) if m is not None
    ) if torch.cuda.is_available() else None

    print(f"\nPipeline completo por imagem (composto: YOLO + candidatos x SmolVLM): "
          f"media={pipeline_mean:.1f}ms (min={per_image_pipeline_ms.min():.1f}ms, max={per_image_pipeline_ms.max():.1f}ms "
          f"entre as {len(img_paths)} imagens do teste, variando com o numero de candidatos por imagem)")
    if pipeline_peak_mem_mb is not None:
        print(f"Pico de memoria do acelerador (pipeline completo, maximo entre as fases): {pipeline_peak_mem_mb:.0f} MB")

    save_experiment_summary(EXP_NAME, RUNTIME_METADATA, {
        "n_test_images": len(img_paths),
        "n_reps_yolo_per_image": N_REPS_YOLO,
        "n_reps_smolvlm_per_particle": N_REPS_SMOLVLM,
        "n_warmup": N_WARMUP,
        "conf_threshold": CONF_THRESHOLD,
        "crop_margin_px": CROP_MARGIN_PX,
        "yolo_latency_ms": {
            "mean": float(yolo_times_ms.mean()), "std": float(yolo_times_ms.std(ddof=1)),
            "n_measurements": int(len(yolo_times_ms)),
        },
        "avg_candidates_per_image": avg_candidates_per_image,
        "yolo_peak_memory_mb": yolo_peak_mem_mb,
        "smolvlm_latency_ms": {
            "mean": float(smolvlm_times_ms.mean()), "std": float(smolvlm_times_ms.std(ddof=1)),
            "n_measurements": int(len(smolvlm_times_ms)), "n_particles": len(crops),
        },
        "smolvlm_peak_memory_mb": smolvlm_peak_mem_mb,
        "full_pipeline_latency_ms_per_image": {
            "method": "composed = yolo_mean + n_candidates_da_imagem * smolvlm_mean, por imagem",
            "mean": pipeline_mean,
            "std_at_avg_candidates": pipeline_std_composed,
            "min": float(per_image_pipeline_ms.min()),
            "max": float(per_image_pipeline_ms.max()),
            "n_images": len(img_paths),
        },
        "full_pipeline_peak_memory_mb": pipeline_peak_mem_mb,
    })
    print(f"\nLog completo: {LOG_PATH}")


if __name__ == "__main__":
    main()
