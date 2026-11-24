"""Benchmark de latência e memória / Latency and memory benchmark.

PT: Mede separadamente:
    1. Latência do YOLO26L por micrografia completa (15 imagens de
    datasets_yolo26_v2/images/test), com N_REPS_YOLO repetições por imagem após aquecimento.
    2. Latência do SmolVLM (base + adaptador LoRA de tamanho, o mesmo de
    20_predicted_box_and_end_to_end.py) por partícula candidata, numa amostra de recortes
    reais do teste.
    3. Latência do pipeline completo por imagem (YOLO + SmolVLM em todas as detecções acima
    do limiar de confiança), composta a partir das fases 1 e 2, e pico de memória do
    acelerador por fase.
    torch.cuda.synchronize() é chamado nos limites de cada medição porque a execução
    PyTorch/CUDA é assíncrona; reset_peak_memory_stats() zera o contador antes de cada fase.
EN: Measures separately:
    1. YOLO26L latency per full micrograph (15 images in datasets_yolo26_v2/images/test),
    with N_REPS_YOLO repetitions per image after warm-up.
    2. SmolVLM latency (base + size LoRA adapter, the same as in
    20_predicted_box_and_end_to_end.py) per candidate particle, on a sample of real test
    crops.
    3. Full pipeline latency per image (YOLO + SmolVLM on every detection above the
    confidence threshold), composed from phases 1 and 2, and peak accelerator memory per
    phase.
    torch.cuda.synchronize() is called at each measurement boundary because PyTorch/CUDA
    execution is asynchronous; reset_peak_memory_stats() resets the counter before each
    phase.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import (
    DIR_CHECKPOINTS, DISPOSITIVO, ID_MODELO, PROMPT_CLASSIFICACAO_TAMANHO,
    classificar_imagem, carregar_modelo_base, salvar_resumo_experimento,
    configurar_log_experimento,
)

from peft import PeftModel
from ultralytics import YOLO

NOME_EXP = "24_efficiency_benchmark"
DIR_EXP, CAMINHO_LOG, REGISTRADOR, METADADOS_EXECUCAO = configurar_log_experimento(NOME_EXP, __file__)

RAIZ = Path(__file__).resolve().parent.parent
DIR_TESTE = RAIZ / "datasets_yolo26_v2"
CAMINHO_MODELO_YOLO = RAIZ / "runs_v2" / "microplastic_yolo26_v2" / "weights" / "best.pt"
ADAPTADOR_LORA_TAMANHO = DIR_CHECKPOINTS / "size_lora"

TAMANHO_IMG = 1024
LIMIAR_CONF = 0.15
MARGEM_RECORTE_PX = 10
N_AQUECIMENTO = 3
N_REPS_YOLO = 5
N_REPS_SMOLVLM = 5
N_RECORTES_SMOLVLM = 30  # PT: partículas candidatas distintas usadas para medir a latência por partícula / EN: distinct candidate particles used to measure per-particle latency
SEMENTE = 42


def sincronizar():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def principal():
    gerador = np.random.default_rng(SEMENTE)
    caminhos_img = sorted((DIR_TESTE / "images/test").glob("*.png"))
    REGISTRADOR.info("n_imagens_teste=%d", len(caminhos_img))

    # PT: Fase 1: YOLO26L, latência por imagem
    # EN: Phase 1: YOLO26L, latency per image
    modelo_yolo = YOLO(str(CAMINHO_MODELO_YOLO))

    # PT: aquecimento: N_AQUECIMENTO inferências descartadas (compilação CUDA/cuDNN,
    #     alocação de cache) antes de medir
    # EN: warm-up: N_AQUECIMENTO discarded inferences (CUDA/cuDNN compilation, cache
    #     allocation) before measuring
    for caminho_img in caminhos_img[:N_AQUECIMENTO]:
        modelo_yolo.predict(source=str(caminho_img), imgsz=TAMANHO_IMG, conf=LIMIAR_CONF, verbose=False)
    sincronizar()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    tempos_yolo_ms = []
    todas_deteccoes = []  # PT: (caminho_img, xyxy_norm, id_cls) para reuso na Fase 3 / EN: (img_path, xyxy_norm, cls_id) reused in Phase 3
    for caminho_img in caminhos_img:
        reps = []
        deteccoes = None
        for r in range(N_REPS_YOLO):
            sincronizar()
            t0 = time.perf_counter()
            pred = modelo_yolo.predict(source=str(caminho_img), imgsz=TAMANHO_IMG, conf=LIMIAR_CONF, verbose=False)[0]
            sincronizar()
            t1 = time.perf_counter()
            reps.append((t1 - t0) * 1000.0)
            if r == 0:
                deteccoes = pred
        tempos_yolo_ms.extend(reps)
        n_candidatos = len(deteccoes.boxes) if deteccoes.boxes is not None else 0
        todas_deteccoes.append((caminho_img, deteccoes, n_candidatos))
        REGISTRADOR.info("yolo imagem=%s reps_ms=%s n_candidatos=%d", caminho_img.name, [f"{x:.1f}" for x in reps], n_candidatos)

    pico_mem_yolo_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else None
    tempos_yolo_ms = np.array(tempos_yolo_ms)
    media_candidatos_por_imagem = float(np.mean([n for _, _, n in todas_deteccoes]))

    print(f"YOLO26L por imagem: media={tempos_yolo_ms.mean():.1f}ms desvio_padrao={tempos_yolo_ms.std(ddof=1):.1f}ms "
          f"(n={len(tempos_yolo_ms)} medicoes = {len(caminhos_img)} imagens x {N_REPS_YOLO} repeticoes)")
    print(f"Candidatos medios por imagem (conf>={LIMIAR_CONF}): {media_candidatos_por_imagem:.1f}")
    if pico_mem_yolo_mb is not None:
        print(f"Pico de memoria do acelerador (YOLO26L): {pico_mem_yolo_mb:.0f} MB")

    del modelo_yolo

    # PT: Fase 2: SmolVLM, latência por partícula candidata
    # EN: Phase 2: SmolVLM, latency per candidate particle
    processador, modelo_base = carregar_modelo_base(ID_MODELO)
    modelo_vlm = PeftModel.from_pretrained(modelo_base, str(ADAPTADOR_LORA_TAMANHO)).to(DISPOSITIVO)
    modelo_vlm.eval()

    # PT: monta até N_RECORTES_SMOLVLM recortes reais a partir das detecções da Fase 1
    # EN: builds up to N_RECORTES_SMOLVLM real crops from the Phase 1 detections
    recortes = []
    for caminho_img, deteccoes, n_candidatos in todas_deteccoes:
        if n_candidatos == 0:
            continue
        with Image.open(caminho_img) as im:
            im = im.convert("RGB")
            W, H = im.size
            xyxy = deteccoes.boxes.xyxy.cpu().numpy()
            for caixa in xyxy:
                x1, y1, x2, y2 = caixa
                x1 = max(0, int(x1) - MARGEM_RECORTE_PX)
                y1 = max(0, int(y1) - MARGEM_RECORTE_PX)
                x2 = min(W, int(x2) + MARGEM_RECORTE_PX)
                y2 = min(H, int(y2) + MARGEM_RECORTE_PX)
                recortes.append(im.crop((x1, y1, x2, y2)).copy())
                if len(recortes) >= N_RECORTES_SMOLVLM:
                    break
        if len(recortes) >= N_RECORTES_SMOLVLM:
            break
    REGISTRADOR.info("n_recortes_smolvlm=%d", len(recortes))

    for recorte in recortes[:N_AQUECIMENTO]:
        classificar_imagem(processador, modelo_vlm, recorte, prompt=PROMPT_CLASSIFICACAO_TAMANHO)
    sincronizar()

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    tempos_smolvlm_ms = []
    for recorte in recortes:
        for _r in range(N_REPS_SMOLVLM):
            sincronizar()
            t0 = time.perf_counter()
            classificar_imagem(processador, modelo_vlm, recorte, prompt=PROMPT_CLASSIFICACAO_TAMANHO)
            sincronizar()
            t1 = time.perf_counter()
            tempos_smolvlm_ms.append((t1 - t0) * 1000.0)

    pico_mem_smolvlm_mb = (torch.cuda.max_memory_allocated() / (1024 ** 2)) if torch.cuda.is_available() else None
    tempos_smolvlm_ms = np.array(tempos_smolvlm_ms)
    print(f"\nSmolVLM por particula: media={tempos_smolvlm_ms.mean():.1f}ms desvio_padrao={tempos_smolvlm_ms.std(ddof=1):.1f}ms "
          f"(n={len(tempos_smolvlm_ms)} medicoes = {len(recortes)} particulas x {N_REPS_SMOLVLM} repeticoes)")
    if pico_mem_smolvlm_mb is not None:
        print(f"Pico de memoria do acelerador (SmolVLM): {pico_mem_smolvlm_mb:.0f} MB")

    # PT: Fase 3: latência do pipeline completo por imagem, por composição. A medição direta
    #     rodaria o SmolVLM uma vez por candidato de cada imagem (até ~300 chamadas por
    #     imagem, ver n_candidatos da Fase 1); com N_REPS_YOLO=5 isso passaria de 3000
    #     chamadas nesta fase. A latência é composta a partir das duas fases medidas, já que
    #     o tempo total depende do número de candidatos do detector em cada imagem:
    #     latencia_pipeline = latencia_YOLO_media + candidatos_da_imagem *
    #     latencia_SmolVLM_media_por_particula.
    #     A incerteza é propagada assumindo independência entre as fases (variância da soma
    #     = soma das variâncias, com var(k*X) = k^2*var(X)). O pico de memória do pipeline é
    #     o máximo entre as duas fases, que não rodam simultaneamente (o SmolVLM só processa
    #     os candidatos depois que o YOLO termina a imagem).
    # EN: Phase 3: full pipeline latency per image, by composition. A direct measurement
    #     would run SmolVLM once per candidate in each image (up to ~300 calls per image,
    #     see n_candidatos from Phase 1); with N_REPS_YOLO=5 that would exceed 3000 calls in
    #     this phase. Latency is composed from the two measured phases, since total time
    #     depends on the number of detector candidates in each image:
    #     pipeline_latency = mean_YOLO_latency + image_candidates *
    #     mean_SmolVLM_latency_per_particle.
    #     Uncertainty is propagated assuming the phases are independent (variance of a sum =
    #     sum of variances, with var(k*X) = k^2*var(X)). Pipeline peak memory is the maximum
    #     of the two phases, which do not run at the same time (SmolVLM only processes the
    #     candidates after YOLO finishes the image).
    media_yolo = float(tempos_yolo_ms.mean())
    var_media_yolo = float(tempos_yolo_ms.var(ddof=1) / len(tempos_yolo_ms))
    media_smolvlm = float(tempos_smolvlm_ms.mean())
    var_media_smolvlm = float(tempos_smolvlm_ms.var(ddof=1) / len(tempos_smolvlm_ms))

    pipeline_ms_por_imagem = []
    for _caminho_img, _deteccoes, n_candidatos in todas_deteccoes:
        pipeline_ms_por_imagem.append(media_yolo + n_candidatos * media_smolvlm)
    pipeline_ms_por_imagem = np.array(pipeline_ms_por_imagem)

    media_pipeline = float(pipeline_ms_por_imagem.mean())
    # PT: propagação de incerteza para UMA imagem com o número médio de candidatos
    # EN: uncertainty propagation for ONE image with the mean number of candidates
    k = media_candidatos_por_imagem
    dp_composto_pipeline = float(np.sqrt(var_media_yolo + (k ** 2) * var_media_smolvlm))
    pico_mem_pipeline_mb = max(
        m for m in (pico_mem_yolo_mb, pico_mem_smolvlm_mb) if m is not None
    ) if torch.cuda.is_available() else None

    print(f"\nPipeline completo por imagem (composto: YOLO + candidatos x SmolVLM): "
          f"media={media_pipeline:.1f}ms (min={pipeline_ms_por_imagem.min():.1f}ms, max={pipeline_ms_por_imagem.max():.1f}ms "
          f"entre as {len(caminhos_img)} imagens do teste, variando com o numero de candidatos por imagem)")
    if pico_mem_pipeline_mb is not None:
        print(f"Pico de memoria do acelerador (pipeline completo, maximo entre as fases): {pico_mem_pipeline_mb:.0f} MB")

    salvar_resumo_experimento(NOME_EXP, METADADOS_EXECUCAO, {
        "n_test_images": len(caminhos_img),
        "n_reps_yolo_per_image": N_REPS_YOLO,
        "n_reps_smolvlm_per_particle": N_REPS_SMOLVLM,
        "n_warmup": N_AQUECIMENTO,
        "conf_threshold": LIMIAR_CONF,
        "crop_margin_px": MARGEM_RECORTE_PX,
        "yolo_latency_ms": {
            "mean": float(tempos_yolo_ms.mean()), "std": float(tempos_yolo_ms.std(ddof=1)),
            "n_measurements": int(len(tempos_yolo_ms)),
        },
        "avg_candidates_per_image": media_candidatos_por_imagem,
        "yolo_peak_memory_mb": pico_mem_yolo_mb,
        "smolvlm_latency_ms": {
            "mean": float(tempos_smolvlm_ms.mean()), "std": float(tempos_smolvlm_ms.std(ddof=1)),
            "n_measurements": int(len(tempos_smolvlm_ms)), "n_particles": len(recortes),
        },
        "smolvlm_peak_memory_mb": pico_mem_smolvlm_mb,
        "full_pipeline_latency_ms_per_image": {
            "method": "composed = yolo_mean + n_candidates_da_imagem * smolvlm_mean, por imagem",
            "mean": media_pipeline,
            "std_at_avg_candidates": dp_composto_pipeline,
            "min": float(pipeline_ms_por_imagem.min()),
            "max": float(pipeline_ms_por_imagem.max()),
            "n_images": len(caminhos_img),
        },
        "full_pipeline_peak_memory_mb": pico_mem_pipeline_mb,
    })
    print(f"\nLog completo: {CAMINHO_LOG}")


if __name__ == "__main__":
    principal()
