"""Benchmark de eficiência com max_det=2000 / Efficiency benchmark with max_det=2000.

PT: Mesma medição de 24_efficiency_benchmark.py com uma mudança de inferência: max_det. Roda
    duas vezes: primeiro com max_det=300 (checagem: deve reproduzir
    avg_candidates_per_image=141,13, yolo_mean=27,38 ms, pipeline_mean=70054 ms,
    pipeline_max=148880 ms), depois com max_det=2000.
    O predict() da Fase 1 de 24 não passava max_det (padrão 300 do Ultralytics), o que
    limitava avg_candidates_per_image e, por composição, o tempo estimado do pipeline
    (full_pipeline_latency_ms_per_image = n_candidatos * media_smolvlm por imagem).
    As latências por unidade (ms por imagem no YOLO, ms por partícula no SmolVLM) não
    dependem de max_det; só mudam avg_candidates_per_image e os tempos de pipeline
    derivados. A Fase 2 também é refeita porque os recortes amostrados dependem das
    detecções da Fase 1, que mudam com max_det.
    Grava em resultados/76_efficiency_benchmark_maxdet2000/, sem sobrescrever
    24_efficiency_benchmark.
    Uso:
    python3 -u 76_efficiency_benchmark_maxdet2000.py
EN: Same measurement as 24_efficiency_benchmark.py with one inference change: max_det. Runs
    twice: first with max_det=300 (check: must reproduce avg_candidates_per_image=141.13,
    yolo_mean=27.38 ms, pipeline_mean=70054 ms, pipeline_max=148880 ms), then with
    max_det=2000.
    Phase 1 predict() in 24 did not pass max_det (Ultralytics default 300), which capped
    avg_candidates_per_image and, by composition, the estimated pipeline time
    (full_pipeline_latency_ms_per_image = n_candidates * smolvlm_mean per image).
    Per-unit latencies (ms per image for YOLO, ms per particle for SmolVLM) do not depend on
    max_det; only avg_candidates_per_image and the derived pipeline times change. Phase 2 is
    also rerun because the sampled crops depend on the Phase 1 detections, which change with
    max_det.
    Writes to resultados/76_efficiency_benchmark_maxdet2000/, without overwriting
    24_efficiency_benchmark.
    Usage:
    python3 -u 76_efficiency_benchmark_maxdet2000.py
"""
import json
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

NOME_EXP = "76_efficiency_benchmark_maxdet2000"
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

RESUMO_REFERENCIA = (Path(__file__).resolve().parent.parent / "resultados"
                     / "24_efficiency_benchmark" / "summary.json")
MAX_DETS_TESTADOS = (300, 2000)
TOL_CHECAGEM_REL = 0.002   # PT: tolerância relativa de 0,2% (checagem com max_det=300) / EN: 0.2% relative tolerance (check with max_det=300)


def sincronizar():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def executar_uma_vez(max_det: int):
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
        modelo_yolo.predict(source=str(caminho_img), imgsz=TAMANHO_IMG, conf=LIMIAR_CONF,
                             max_det=max_det, verbose=False)
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
            pred = modelo_yolo.predict(source=str(caminho_img), imgsz=TAMANHO_IMG, conf=LIMIAR_CONF,
                                      max_det=max_det, verbose=False)[0]
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
    #     rodaria o SmolVLM uma vez por candidato de cada imagem (centenas de chamadas por
    #     imagem, ver n_candidatos da Fase 1), o que tornaria o benchmark impraticável. A
    #     latência é composta a partir das duas fases medidas, já que o tempo total depende
    #     do número de candidatos do detector em cada imagem:
    #     latencia_pipeline = latencia_YOLO_media + candidatos_da_imagem *
    #     latencia_SmolVLM_media_por_particula.
    #     A incerteza é propagada assumindo independência entre as fases (variância da soma
    #     = soma das variâncias, com var(k*X) = k^2*var(X)). O pico de memória do pipeline é
    #     o máximo entre as duas fases, que não rodam simultaneamente (o SmolVLM só processa
    #     os candidatos depois que o YOLO termina a imagem).
    # EN: Phase 3: full pipeline latency per image, by composition. A direct measurement
    #     would run SmolVLM once per candidate in each image (hundreds of calls per image,
    #     see n_candidatos from Phase 1), making the benchmark impractical. Latency is
    #     composed from the two measured phases, since total time depends on the number of
    #     detector candidates in each image:
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

    return {
        "max_det": max_det,
        "n_test_images": len(caminhos_img),
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
            "mean": media_pipeline,
            "std_at_avg_candidates": dp_composto_pipeline,
            "min": float(pipeline_ms_por_imagem.min()),
            "max": float(pipeline_ms_por_imagem.max()),
            "n_images": len(caminhos_img),
        },
        "full_pipeline_peak_memory_mb": pico_mem_pipeline_mb,
    }


def principal():
    if not RESUMO_REFERENCIA.exists():
        raise SystemExit(f"nao existe: {RESUMO_REFERENCIA}")
    referencia = json.loads(RESUMO_REFERENCIA.read_text())["metrics"]

    resultados = {}
    for md in MAX_DETS_TESTADOS:
        print(f"\n{'='*72}\nmax_det={md}\n{'='*72}")
        REGISTRADOR.info("iniciando run_once(max_det=%d)", md)
        resultados[md] = executar_uma_vez(md)

    r300 = resultados[300]
    print(f"\n{'='*72}\nSANIDADE -- max_det=300 reproduz 24_efficiency_benchmark?\n{'='*72}")
    # PT: tolerância estreita (0,2%) para métricas determinísticas (contagem de candidatos e
    #     pico de memória dependem só do checkpoint e do max_det); tolerância larga (10%)
    #     para latência em ms, que varia entre execuções por clock, temperatura e outros
    #     processos na máquina
    # EN: tight tolerance (0.2%) for deterministic metrics (candidate count and peak memory
    #     depend only on the checkpoint and max_det); loose tolerance (10%) for latency in
    #     ms, which varies between runs with clock, temperature and other processes on the
    #     machine
    checagens = [
        ("yolo_latency_ms.mean", r300["yolo_latency_ms"]["mean"], referencia["yolo_latency_ms"]["mean"], 0.10),
        ("avg_candidates_per_image", r300["avg_candidates_per_image"], referencia["avg_candidates_per_image"], TOL_CHECAGEM_REL),
        ("yolo_peak_memory_mb", r300["yolo_peak_memory_mb"], referencia["yolo_peak_memory_mb"], TOL_CHECAGEM_REL),
        ("full_pipeline.mean", r300["full_pipeline_latency_ms_per_image"]["mean"],
         referencia["full_pipeline_latency_ms_per_image"]["mean"], 0.10),
        ("full_pipeline.max", r300["full_pipeline_latency_ms_per_image"]["max"],
         referencia["full_pipeline_latency_ms_per_image"]["max"], 0.10),
    ]
    ok = True
    for nome, obtido, pub, tol in checagens:
        rel = abs(obtido - pub) / pub if pub else abs(obtido - pub)
        sinalizador = "" if rel <= tol else "  <-- DIVERGE"
        if rel > tol:
            ok = False
        print(f"   {nome:28s} referência={pub:14.3f}  obtido={obtido:14.3f}  "
              f"dif_rel={rel*100:.3f}%  (tol={tol*100:.0f}%){sinalizador}")
    print("   ->", "SIM" if ok else "NAO")
    if not ok:
        print("\n   !! nao reproduz -- o resultado de max_det=2000 NAO deve ser usado.")
        print("      (yolo_latency_ms e smolvlm_latency_ms podem variar um pouco por ruido de GPU;")
        print("      o teste real de sanidade e avg_candidates_per_image e os tempos de pipeline.)")
        return

    r2000 = resultados[2000]
    print(f"\n{'='*72}\nmax_det=300 (sanidade)  vs  max_det=2000\n{'='*72}")
    print(f"   avg_candidates_per_image   {r300['avg_candidates_per_image']:8.1f}  ->  "
          f"{r2000['avg_candidates_per_image']:8.1f}")
    print(f"   pipeline mean (s)          {r300['full_pipeline_latency_ms_per_image']['mean']/1000:8.1f}  ->  "
          f"{r2000['full_pipeline_latency_ms_per_image']['mean']/1000:8.1f}")
    print(f"   pipeline max (s)           {r300['full_pipeline_latency_ms_per_image']['max']/1000:8.1f}  ->  "
          f"{r2000['full_pipeline_latency_ms_per_image']['max']/1000:8.1f}")
    print(f"   smolvlm_peak_memory_mb     {r300['smolvlm_peak_memory_mb']:8.1f}  ->  "
          f"{r2000['smolvlm_peak_memory_mb']:8.1f}")

    salvar_resumo_experimento(NOME_EXP, METADADOS_EXECUCAO, {
        "note": "reavaliacao do benchmark de latencia e memoria com max_det explicito. "
                "yolo_latency_ms e smolvlm_latency_ms sao custos por chamada e nao "
                "dependem de max_det; avg_candidates_per_image e os tempos de pipeline "
                "compostos mudam porque dependem da contagem de deteccoes.",
        "published_reference": referencia,
        "sanity_reproduces_published": ok,
        "maxdet_300": r300,
        "maxdet_2000": r2000,
    })
    print(f"\nLog completo: {CAMINHO_LOG}")
    print(f"\nLog completo: {CAMINHO_LOG}")


if __name__ == "__main__":
    principal()
