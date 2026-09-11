"""Camada 3 do artigo (`sec:digital_representation`): registro persistente por
partícula, $\\mathcal{D}_i$, ligando a evidência de origem ao estado analítico
atual e preservando estados anteriores após reprocessamento.

Contexto: uma revisão da seção "Arquitetura em camadas" encontrou essa camada
descrita no artigo mas não implementada no código -- cada script gravava um
subconjunto diferente de campos, sem identificador de partícula estável entre
estágios, e sempre sobrescrevendo o arquivo de saída anterior (sem histórico).
Este módulo cobre exatamente essas duas lacunas:

1. Schema único (`ParticleRecord`) com os 10 campos descritos no artigo:
   imagem de origem, metadados físicos, caixa predita, rótulo+confiança do
   detector, diâmetro de referência, regime do SmolVLM, regime de
   referência, status de avaliação e versões de modelo/processamento.
2. Persistência append-only (JSON Lines): cada execução ("reprocessamento")
   grava novas linhas com um `run_id` próprio; nenhuma linha de uma execução
   anterior é sobrescrita ou removida. `load_history` lê tudo; `latest_state`
   filtra só o registro mais recente por partícula, quando só o estado atual
   interessa.

Não faz parte deste módulo: rodar o YOLO26L ou o SmolVLM. Quem monta os
`ParticleRecord` e decide quando chamar `append_records` é o script de
pipeline (ver `55_predicted_box_and_end_to_end_records.py`).
"""
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from vlm_common import RESULTS_ROOT, collect_runtime_metadata

PARTICLE_RECORDS_DIR = RESULTS_ROOT / "particle_records"


@dataclass
class ParticleRecord:
    # -- identidade da partícula, ligada à Camada 0 (evidência de origem) --
    source_image: str          # I_i: nome do arquivo de imagem de origem
    particle_index: int        # posição estável da partícula no arquivo de
                                # rótulo de referência daquela imagem (0-based).
                                # NÃO é o id de anotação original do MiNa/COCO
                                # -- uma tentativa de recuperar esse id via
                                # casamento por IoU contra os JSONs COCO
                                # brutos falhou (contagem de anotações não
                                # bate com o número de partículas do split de
                                # teste em imagens densas, mesmo tipo de
                                # colisão de nome de arquivo já documentado
                                # no artigo). Este índice é estável entre
                                # execuções porque o arquivo de rótulo de
                                # teste é estático; não é rastreável até o
                                # dataset MiNa original fora deste split.

    # -- metadados físicos e de aquisição, m_i (Camada 0) --
    magnification: float | None
    box_ref_xyxy: tuple[float, float, float, float] | None

    # -- referência (ground truth), usada só para avaliação --
    polymer_label_ref: str
    size_nm_ref: float | None      # d_i: diâmetro de referência, quando disponível
    size_regime_ref: str | None    # s_i^ref

    # -- Camada 1: saída do detector --
    detected: bool                                 # D_i
    box_pred_xyxy: tuple[float, float, float, float] | None  # b_i
    polymer_label_pred: str | None                 # p_i^YOLO
    detector_confidence: float | None              # c_i^YOLO
    polymer_correct: bool                          # P_i

    # -- Camada 2: saída do SmolVLM --
    size_regime_pred: str | None                   # s_i^VLM
    size_correct: bool | None                      # S_i

    # -- avaliação fim a fim --
    end_to_end_correct: bool                       # E_i
    evaluation_status: str                         # q_i, ver `evaluation_status()`

    # -- proveniência desta execução (reprocessamento), v_i --
    run_id: str
    created_at_utc: str
    model_versions: dict = field(default_factory=dict)


def evaluation_status(detected: bool, polymer_correct: bool, size_correct: bool | None) -> str:
    """Traduz (D_i, P_i, S_i) num status legível -- q_i do artigo. Cada
    partícula falha em exatamente um estágio, ou passa em todos; isso é o
    que permite distinguir "partícula perdida" de "erro de classificação"
    sem precisar reler as três colunas booleanas toda vez."""
    if not detected:
        return "missed_detection"
    if not polymer_correct:
        return "wrong_polymer_label"
    if size_correct is not True:
        return "wrong_size_regime"
    return "correct"


def make_run_id(model_versions: dict, tag: str | None = None) -> str:
    """Identificador único desta execução (reprocessamento). Combina
    timestamp (granularidade de segundo já evita colisão em uma única
    chamada de script) com o commit do git, pra que duas execuções no mesmo
    segundo com checkpoints diferentes ainda fiquem distinguíveis pelo
    commit registrado em `model_versions`."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    commit = model_versions.get("git_commit") or "nogit"
    parts = [ts, commit]
    if tag:
        parts.append(tag)
    return "_".join(parts)


def collect_model_versions(extra: dict | None = None) -> dict:
    """Wrapper fino sobre `vlm_common.collect_runtime_metadata` -- reaproveita
    o mesmo registro de versões de biblioteca e commit já usado em todo
    `summary.json` de experimento, em vez de duplicar essa lógica aqui."""
    return collect_runtime_metadata(extra)


def append_records(records: list[ParticleRecord], out_path: Path) -> Path:
    """Acrescenta registros a `out_path` em formato JSON Lines. Nunca abre em
    modo de escrita truncada -- é isso que garante que uma execução nova não
    apaga o estado analítico de execuções anteriores. `out_path` deve viver
    sob `PARTICLE_RECORDS_DIR`, mas a função aceita qualquer caminho pra
    facilitar testes."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    return out_path


def load_history(path: Path) -> pd.DataFrame:
    """Lê todo o histórico acumulado (todas as execuções já registradas) como
    um único DataFrame, uma linha por partícula por execução. Uma partícula
    reprocessada N vezes aparece N vezes, uma por `run_id`."""
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]
    return pd.DataFrame(rows)


def latest_state(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra `load_history(...)` para manter só o registro mais recente por
    partícula (source_image, particle_index) -- o "estado analítico atual" do
    artigo. O histórico completo continua disponível via `load_history`;
    esta função não descarta nada em disco, só na visão retornada."""
    if df.empty:
        return df
    return (
        df.sort_values("created_at_utc")
        .groupby(["source_image", "particle_index"], as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
