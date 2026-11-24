"""Registro persistente por partícula / Persistent per-particle record.

PT: Camada 3 (representação digital): registro persistente por partícula, D_i, que liga a
    evidência de origem ao estado analítico atual e preserva estados anteriores após
    reprocessamento.
    1. Esquema único (`RegistroParticula`) com 10 grupos de campos: imagem de origem,
    metadados físicos, caixa prevista, rótulo + confiança do detector, diâmetro de
    referência, regime do SmolVLM, regime de referência, status de avaliação e versões de
    modelo/processamento.
    2. Persistência só de acréscimo (JSON Lines): cada execução (reprocessamento) grava
    novas linhas com um `run_id` próprio; nenhuma linha anterior é sobrescrita ou removida.
    `carregar_historico` lê tudo; `estado_recente` mantém só o registro mais recente por
    partícula.
    `particle_index` é a posição estável da partícula no arquivo de rótulos de referência da
    imagem (base 0). Não é o id de anotação original do MiNa/COCO (o pareamento por IoU
    contra os JSONs COCO brutos falha em imagens densas, onde a contagem de anotações não
    coincide com o número de partículas da partição de teste); é estável entre execuções
    porque o arquivo de rótulos de teste é estático.
    Este módulo não roda o YOLO26L nem o SmolVLM: quem monta os registros e chama
    `anexar_registros` é o script de pipeline (ver
    `55_predicted_box_and_end_to_end_records.py`).
EN: Layer 3 (digital representation): a persistent per-particle record, D_i, linking the
    source evidence to the current analytical state and keeping previous states after
    reprocessing.
    1. A single schema (`RegistroParticula`) with 10 field groups: source image, physical
    metadata, predicted box, detector label + confidence, reference diameter, SmolVLM
    regime, reference regime, evaluation status and model/processing versions.
    2. Append-only persistence (JSON Lines): each run (reprocessing) writes new lines with
    its own `run_id`; no previous line is overwritten or removed. `carregar_historico` reads
    everything; `estado_recente` keeps only the latest record per particle.
    `particle_index` is the particle's stable position in the image's reference label file
    (0-based). It is not the original MiNa/COCO annotation id (IoU matching against the raw
    COCO JSONs fails in dense images, where the annotation count does not match the number
    of particles in the test split); it is stable across runs because the test label file is
    static.
    This module does not run YOLO26L or SmolVLM: the pipeline script builds the records and
    calls `anexar_registros` (see `55_predicted_box_and_end_to_end_records.py`).
"""
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from vlm_common import RAIZ_RESULTADOS, coletar_metadados_execucao

DIR_REGISTROS_PARTICULAS = RAIZ_RESULTADOS / "particle_records"


@dataclass
class RegistroParticula:
    # PT: -- identidade da partícula, ligada à Camada 0 (evidência de origem) --
    # EN: -- particle identity, linked to Layer 0 (source evidence) --
    source_image: str          # PT: I_i: nome do arquivo da imagem de origem / EN: I_i: source image file name
    particle_index: int        # PT: posição estável no arquivo de rótulos de referência (base 0) / EN: stable position in the reference label file (0-based)

    # PT: -- metadados físicos e de aquisição, m_i (Camada 0) --
    # EN: -- physical and acquisition metadata, m_i (Layer 0) --
    magnification: float | None
    box_ref_xyxy: tuple[float, float, float, float] | None

    # PT: -- referência (ground truth), usada só para avaliação --
    # EN: -- reference (ground truth), used only for evaluation --
    polymer_label_ref: str
    size_nm_ref: float | None      # PT: d_i: diâmetro de referência, quando disponível / EN: d_i: reference diameter, when available
    size_regime_ref: str | None    # PT: s_i^ref: regime de tamanho de referência / EN: s_i^ref: reference size regime

    # PT: -- Camada 1: saída do detector --
    # EN: -- Layer 1: detector output --
    detected: bool                                 # PT: D_i: detecção correta / EN: D_i: correct detection
    box_pred_xyxy: tuple[float, float, float, float] | None  # PT: b_i: caixa prevista / EN: b_i: predicted box
    polymer_label_pred: str | None                 # PT: p_i^YOLO: polímero previsto pelo detector / EN: p_i^YOLO: polymer predicted by the detector
    detector_confidence: float | None              # PT: c_i^YOLO: confiança do detector / EN: c_i^YOLO: detector confidence
    polymer_correct: bool                          # PT: P_i: polímero correto / EN: P_i: correct polymer

    # PT: -- Camada 2: saída do SmolVLM --
    # EN: -- Layer 2: SmolVLM output --
    size_regime_pred: str | None                   # PT: s_i^VLM: regime de tamanho previsto pelo SmolVLM / EN: s_i^VLM: size regime predicted by SmolVLM
    size_correct: bool | None                      # PT: S_i: regime de tamanho correto / EN: S_i: correct size regime

    # PT: -- avaliação fim a fim --
    # EN: -- end-to-end evaluation --
    end_to_end_correct: bool                       # PT: E_i: acerto fim a fim / EN: E_i: end-to-end correctness
    evaluation_status: str                         # PT: q_i, ver status_avaliacao() / EN: q_i, see status_avaliacao()

    # PT: -- proveniência desta execução (reprocessamento), v_i --
    # EN: -- provenance of this run (reprocessing), v_i --
    run_id: str
    created_at_utc: str
    model_versions: dict = field(default_factory=dict)


def status_avaliacao(detected: bool, polymer_correct: bool, size_correct: bool | None) -> str:
    """PT: Converte (D_i, P_i, S_i) num status legível (q_i). Cada partícula falha em
        exatamente um estágio ou passa em todos, o que distingue "partícula perdida" de
        "erro de classificação" sem reler as três colunas booleanas.
    EN: Converts (D_i, P_i, S_i) into a readable status (q_i). Each particle fails at
        exactly one stage or passes all of them, which distinguishes "missed particle" from
        "classification error" without re-reading the three boolean columns.
    """
    if not detected:
        return "missed_detection"
    if not polymer_correct:
        return "wrong_polymer_label"
    if size_correct is not True:
        return "wrong_size_regime"
    return "correct"


def gerar_id_execucao(model_versions: dict, rotulo_exp: str | None = None) -> str:
    """PT: Identificador único da execução (reprocessamento): carimbo de tempo UTC (resolução
        de segundo) + commit do git, para que duas execuções no mesmo segundo com
        checkpoints diferentes continuem distinguíveis pelo commit registrado em
        `model_versions`.
    EN: Unique run (reprocessing) identifier: UTC timestamp (one-second resolution) + git
        commit, so that two runs in the same second with different checkpoints remain
        distinguishable by the commit recorded in `model_versions`.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    commit = model_versions.get("git_commit") or "nogit"
    partes = [ts, commit]
    if rotulo_exp:
        partes.append(rotulo_exp)
    return "_".join(partes)


def coletar_versoes_modelos(extra: dict | None = None) -> dict:
    """PT: Invólucro fino sobre `vlm_common.coletar_metadados_execucao`: reaproveita o registro
        de versões de bibliotecas e commit usado nos `summary.json` dos experimentos.
    EN: Thin wrapper over `vlm_common.coletar_metadados_execucao`: reuses the library
        version and commit record used in the experiments' `summary.json` files.
    """
    return coletar_metadados_execucao(extra)


def anexar_registros(registros: list[RegistroParticula], caminho_saida: Path) -> Path:
    """PT: Acrescenta registros a `caminho_saida` em JSON Lines. Nunca abre o arquivo em modo
        de escrita com truncamento, garantindo que uma execução nova não apague o estado de
        execuções anteriores. `caminho_saida` deve ficar sob `DIR_REGISTROS_PARTICULAS`, mas
        qualquer caminho é aceito (útil em testes).
    EN: Appends records to `caminho_saida` as JSON Lines. The file is never opened in
        truncating write mode, so a new run never erases the state of previous runs.
        `caminho_saida` should live under `DIR_REGISTROS_PARTICULAS`, but any path is
        accepted (useful for tests).
    """
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    with open(caminho_saida, "a", encoding="utf-8") as marcador:
        for registro in registros:
            marcador.write(json.dumps(asdict(registro), ensure_ascii=False) + "\n")
    return caminho_saida


def carregar_historico(caminho: Path) -> pd.DataFrame:
    """PT: Lê todo o histórico acumulado (todas as execuções registradas) num único DataFrame,
        uma linha por partícula por execução. Uma partícula reprocessada N vezes aparece N
        vezes, uma por `run_id`.
    EN: Reads the whole accumulated history (every recorded run) into a single DataFrame,
        one row per particle per run. A particle reprocessed N times appears N times, once
        per `run_id`.
    """
    if not caminho.exists():
        return pd.DataFrame()
    linhas_tab = [json.loads(linha) for linha in open(caminho, encoding="utf-8") if linha.strip()]
    return pd.DataFrame(linhas_tab)


def estado_recente(df: pd.DataFrame) -> pd.DataFrame:
    """PT: Filtra `carregar_historico(...)` mantendo só o registro mais recente por partícula
        (source_image, particle_index), isto é, o estado analítico atual. O histórico
        completo continua em disco; só a visão retornada é filtrada.
    EN: Filters `carregar_historico(...)` keeping only the latest record per particle
        (source_image, particle_index), i.e. the current analytical state. The full history
        stays on disk; only the returned view is filtered.
    """
    if df.empty:
        return df
    return (
        df.sort_values("created_at_utc")
        .groupby(["source_image", "particle_index"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
