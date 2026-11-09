"""Copia fiel de 24_efficiency_benchmark.py com UMA mudanca de inferencia:
max_det. Roda DUAS vezes -- primeiro max_det=300 (sanidade, deve reproduzir
avg_candidates_per_image=141.13, yolo_mean=27.38ms, pipeline_mean=70054ms,
pipeline_max=148880ms), depois max_det=2000.

Os Metodos declaram "at most 2,000 detections per image", mas o predict() da
Fase 1 nao passava max_det (padrao 300 do Ultralytics) -- mesmo problema ja
corrigido no detector principal. Isso limitava avg_candidates_per_image e,
por composicao, o tempo de pipeline estimado (full_pipeline_latency_ms_per_image),
que e derivado de n_candidates * smolvlm_mean por imagem.

As latencias POR UNIDADE (ms por imagem do YOLO, ms por particula do SmolVLM)
nao dependem de max_det -- sao custos por chamada, nao por contagem de
candidatos. So mudam: avg_candidates_per_image e os tempos de pipeline
compostos a partir dela. Reroda a Fase 2 tambem (nao so reusa o yolo/smolvlm_mean
publicados) porque os recortes amostrados para a Fase 2 dependem de quais
deteccoes a Fase 1 produziu, que mudam com max_det.

Grava em resultados/76_efficiency_benchmark_maxdet2000/, nao sobrescreve
24_efficiency_benchmark.

Uso:
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
    CHECKPOINTS_DIR, DEVICE, MODEL_ID, SIZE_CLASSIFY_PROMPT,
    classify_image, load_base_model, save_experiment_summary,
    setup_experiment_logging,
)

from peft import PeftModel
from ultralytics import YOLO

EXP_NAME = "76_efficiency_benchmark_maxdet2000"
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

PUBLISHED_SUMMARY = (Path(__file__).resolve().parent.parent / "resultados"
                     / "24_efficiency_benchmark" / "summary.json")
MAX_DETS = (300, 2000)
SANITY_TOL_REL = 0.002   # 0.2% de tolerancia relativa (sanidade, max_det=300)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def run_once(max_det: int):
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
        yolo_model.predict(source=str(img_path), imgsz=IMG_SIZE, conf=CONF_THRESHOLD,
                             max_det=max_det, verbose=False)
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
            pred = yolo_model.predict(source=str(img_path), imgsz=IMG_SIZE, conf=CONF_THRESHOLD,
                                      max_det=max_det, verbose=False)[0]
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

    return {
        "max_det": max_det,
        "n_test_images": len(img_paths),
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
            "mean": pipeline_mean,
            "std_at_avg_candidates": pipeline_std_composed,
            "min": float(per_image_pipeline_ms.min()),
            "max": float(per_image_pipeline_ms.max()),
            "n_images": len(img_paths),
        },
        "full_pipeline_peak_memory_mb": pipeline_peak_mem_mb,
    }


def main():
    if not PUBLISHED_SUMMARY.exists():
        raise SystemExit(f"nao existe: {PUBLISHED_SUMMARY}")
    published = json.loads(PUBLISHED_SUMMARY.read_text())["metrics"]

    results = {}
    for md in MAX_DETS:
        print(f"\n{'='*72}\nmax_det={md}\n{'='*72}")
        LOGGER.info("iniciando run_once(max_det=%d)", md)
        results[md] = run_once(md)

    r300 = results[300]
    print(f"\n{'='*72}\nSANIDADE -- max_det=300 reproduz 24_efficiency_benchmark?\n{'='*72}")
    # tolerancia apertada (0.2%) para metricas DETERMINISTICAS -- contagem de
    # candidatos e pico de memoria dependem so do checkpoint e do max_det, nao
    # de ruido de cronometragem. Tolerancia larga (10%) para LATENCIA em ms --
    # tempo de parede de GPU varia entre execucoes por clock/termica/outros
    # processos na maquina, mesmo com o mesmo checkpoint e os mesmos dados.
    checks = [
        ("yolo_latency_ms.mean", r300["yolo_latency_ms"]["mean"], published["yolo_latency_ms"]["mean"], 0.10),
        ("avg_candidates_per_image", r300["avg_candidates_per_image"], published["avg_candidates_per_image"], SANITY_TOL_REL),
        ("yolo_peak_memory_mb", r300["yolo_peak_memory_mb"], published["yolo_peak_memory_mb"], SANITY_TOL_REL),
        ("full_pipeline.mean", r300["full_pipeline_latency_ms_per_image"]["mean"],
         published["full_pipeline_latency_ms_per_image"]["mean"], 0.10),
        ("full_pipeline.max", r300["full_pipeline_latency_ms_per_image"]["max"],
         published["full_pipeline_latency_ms_per_image"]["max"], 0.10),
    ]
    ok = True
    for name, obtained, pub, tol in checks:
        rel = abs(obtained - pub) / pub if pub else abs(obtained - pub)
        flag = "" if rel <= tol else "  <-- DIVERGE"
        if rel > tol:
            ok = False
        print(f"   {name:28s} publicado={pub:14.3f}  obtido={obtained:14.3f}  "
              f"dif_rel={rel*100:.3f}%  (tol={tol*100:.0f}%){flag}")
    print("   ->", "SIM" if ok else "NAO")
    if not ok:
        print("\n   !! nao reproduz -- o resultado de max_det=2000 NAO deve ser usado.")
        print("      (yolo_latency_ms e smolvlm_latency_ms podem variar um pouco por ruido de GPU;")
        print("      o teste real de sanidade e avg_candidates_per_image e os tempos de pipeline.)")
        return

    r2000 = results[2000]
    print(f"\n{'='*72}\nmax_det=300 (sanidade)  vs  max_det=2000\n{'='*72}")
    print(f"   avg_candidates_per_image   {r300['avg_candidates_per_image']:8.1f}  ->  "
          f"{r2000['avg_candidates_per_image']:8.1f}")
    print(f"   pipeline mean (s)          {r300['full_pipeline_latency_ms_per_image']['mean']/1000:8.1f}  ->  "
          f"{r2000['full_pipeline_latency_ms_per_image']['mean']/1000:8.1f}")
    print(f"   pipeline max (s)           {r300['full_pipeline_latency_ms_per_image']['max']/1000:8.1f}  ->  "
          f"{r2000['full_pipeline_latency_ms_per_image']['max']/1000:8.1f}")
    print(f"   smolvlm_peak_memory_mb     {r300['smolvlm_peak_memory_mb']:8.1f}  ->  "
          f"{r2000['smolvlm_peak_memory_mb']:8.1f}")

    save_experiment_summary(EXP_NAME, RUNTIME_METADATA, {
        "note": "reavaliacao da Secao de Custo Computacional com max_det explicito. "
                "yolo_latency_ms e smolvlm_latency_ms sao custos por chamada e nao "
                "dependem de max_det; avg_candidates_per_image e os tempos de pipeline "
                "compostos mudam porque dependem da contagem de deteccoes.",
        "published_reference": published,
        "sanity_reproduces_published": ok,
        "maxdet_300": r300,
        "maxdet_2000": r2000,
    })
    print(f"\nLog completo: {LOG_PATH}")
    print(f"\nLog completo: {LOG_PATH}")


if __name__ == "__main__":
    main()
