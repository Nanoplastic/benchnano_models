"""Relatório final das repetições controladas / Controlled repetitions final report.

PT: Agrega as 10 repetições de cada experimento controlado (+ a avaliação fim a fim pareada
    e as reavaliações com max_det=2000 / regra eq1) em média, desvio-padrão e IC 95%
    (distribuição t, n=10)  por métrica.
    Saída em experimentos_controlados/relatorio_final/:
    estatisticas_agregadas.json -- tudo, estruturado
    estatisticas_agregadas.csv -- uma linha por (experimento, métrica)
    tempos_execucao.json / tempos_execucao.csv -- tempo de execução por experimento
    Rodar depois que todos os experimentos e a avaliação pareada tiverem as 10 repetições
    (run_queue.py chama este script automaticamente no final):
    python3 final_report.py
EN: Aggregates the 10 repetitions of each controlled experiment (+ the paired end-to-end
    evaluation and the max_det=2000 / eq1-rule re-evaluations) into mean, standard deviation
    and 95% CI (t distribution, n=10)  per metric.
    Output in experimentos_controlados/relatorio_final/:
    estatisticas_agregadas.json -- everything, structured
    estatisticas_agregadas.csv -- one row per (experiment, metric)
    tempos_execucao.json / tempos_execucao.csv -- execution time per experiment
    Run once every experiment and the paired evaluation have all 10 repetitions
    (run_queue.py calls this script automatically at the end):
    python3 final_report.py
"""
import csv
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctrl_common as cc

# PT: Métricas relatadas por experimento e como extraí-las de metrics{}.
#     max_det: os scripts de avaliação originais não passam max_det ao preditor do
#     Ultralytics, que usa então o padrão 300 em vez do max_det=2000 do treino (gravado no
#     checkpoint; ver experimentos/resultados/61_reeval_detector_maxdet/summary.json, campo
#     "note"). Isso subestima recall/precisão/mAP do detector em micrografias densas. Os
#     valores finais são "01_yolo26_main_maxdet2000" (detector com max_det=2000),
#     "07_smolvlm_size_lora_eq1" e "08_smolvlm_size_full_eq1" (regime de tamanho pela regra
#     eq1) e "09_end_to_end_paired_maxdet2000" (fim a fim pareado com max_det=2000). As
#     entradas "01_yolo26_main", "05_smolvlm_size_lora", "06_smolvlm_size_full" e
#     "07_end_to_end_paired" são mantidas para rastreabilidade, marcadas como SUPERADAS.
# EN: Metrics reported per experiment and how to extract them from metrics{}.
#     max_det: the original evaluation scripts do not pass max_det to the Ultralytics
#     predictor, which then uses the default 300 instead of the training max_det=2000
#     (stored in the checkpoint; see
#     experimentos/resultados/61_reeval_detector_maxdet/summary.json, "note" field). This
#     underestimates detector recall/precision/mAP on dense micrographs. The final values
#     are "01_yolo26_main_maxdet2000" (detector with max_det=2000),
#     "07_smolvlm_size_lora_eq1" and "08_smolvlm_size_full_eq1" (size regime with the eq1
#     rule) and "09_end_to_end_paired_maxdet2000" (paired end-to-end with max_det=2000). The
#     entries "01_yolo26_main", "05_smolvlm_size_lora", "06_smolvlm_size_full" and
#     "07_end_to_end_paired" are kept for traceability, marked as SUPERSEDED.
ESPEC_METRICAS = {
    "01_yolo26_main": [
        ("precision", lambda m: m["aggregate_metrics"]["precision"]),
        ("recall", lambda m: m["aggregate_metrics"]["recall"]),
        ("map50", lambda m: m["aggregate_metrics"]["map50"]),
        ("map50_95", lambda m: m["aggregate_metrics"]["map50_95"]),
    ],
    "01_yolo26_main_maxdet2000": [
        ("precision", lambda m: m["aggregate_metrics"]["precision"]),
        ("recall", lambda m: m["aggregate_metrics"]["recall"]),
        ("map50", lambda m: m["aggregate_metrics"]["map50"]),
        ("map50_95", lambda m: m["aggregate_metrics"]["map50_95"]),
    ],
    "02_vlm_polymer_real_only": [("test_acc", lambda m: m["test_acc"])],
    "03_vlm_polymer_real_synth": [("test_acc", lambda m: m["test_acc"])],
    "04_vlm_polymer_bigvlm": [("test_acc", lambda m: m["test_acc"])],
    "05_smolvlm_size_lora": [("test_acc", lambda m: m["test_acc"])],
    "06_smolvlm_size_full": [("test_acc", lambda m: m["test_acc"])],
    "07_smolvlm_size_lora_eq1": [("test_acc", lambda m: m["test_acc"])],
    "08_smolvlm_size_full_eq1": [("test_acc", lambda m: m["test_acc"])],
    "07_end_to_end_paired": [
        ("D_rate", lambda m: m["D_rate"]),
        ("P_rate_given_D", lambda m: m["P_rate_given_D"]),
        ("smolvlm_accuracy_predicted_box", lambda m: m["smolvlm_accuracy_predicted_box"]),
        ("E_rate_strict_end_to_end", lambda m: m["E_rate_strict_end_to_end"]),
    ],
    "09_end_to_end_paired_maxdet2000": [
        ("D_rate", lambda m: m["D_rate"]),
        ("P_rate_given_D", lambda m: m["P_rate_given_D"]),
        ("smolvlm_accuracy_predicted_box", lambda m: m["smolvlm_accuracy_predicted_box"]),
        ("E_rate_strict_end_to_end", lambda m: m["E_rate_strict_end_to_end"]),
    ],
}

ROTULOS = {
    "01_yolo26_main": "YOLO26L principal (histórico, max_det=300 -- SUPERADO, ver 01_yolo26_main_maxdet2000)",
    "01_yolo26_main_maxdet2000": "YOLO26L principal (max_det=2000, corrigido -- valor final)",
    "02_vlm_polymer_real_only": "SmolVLM LoRA polímero (só real)",
    "03_vlm_polymer_real_synth": "SmolVLM LoRA polímero (real+sintético)",
    "04_vlm_polymer_bigvlm": "SmolVLM ~2B polímero (só real)",
    "05_smolvlm_size_lora": "SmolVLM LoRA regime de tamanho (histórico -- SUPERADO, ver 07_smolvlm_size_lora_eq1)",
    "06_smolvlm_size_full": "SmolVLM full fine-tune regime de tamanho (histórico -- SUPERADO, ver 08_smolvlm_size_full_eq1)",
    "07_smolvlm_size_lora_eq1": "SmolVLM LoRA regime de tamanho (regra eq1 -- valor final)",
    "08_smolvlm_size_full_eq1": "SmolVLM full fine-tune regime de tamanho (regra eq1 -- valor final)",
    "07_end_to_end_paired": "Fim-a-fim pareado (YOLO+SmolVLM regime) (histórico, max_det=300 -- SUPERADO, ver 09_end_to_end_paired_maxdet2000)",
    "09_end_to_end_paired_maxdet2000": "Fim-a-fim pareado (YOLO+SmolVLM regime) (max_det=2000 -- valor final)",
}

# PT: Fonte especial para "01_yolo26_main_maxdet2000": não é uma pasta
#     experimentos_controlados/resultados/<exp>/rep_NN/ como as demais; as 10 sementes da
#     reavaliação ficam num único summary.json em
#     experimentos/resultados/61_reeval_detector_maxdet/.
# EN: Special source for "01_yolo26_main_maxdet2000": it is not an
#     experimentos_controlados/resultados/<exp>/rep_NN/ folder like the others; the 10
#     re-evaluation seeds live in a single summary.json under
#     experimentos/resultados/61_reeval_detector_maxdet/.
RESUMO_DETECTOR_MAXDET2000 = (
    cc.RAIZ_PROJETO_ORIGEM / "experimentos" / "resultados"
    / "61_reeval_detector_maxdet" / "summary.json"
)


def carregar_reps_detector_maxdet2000() -> list:
    if not RESUMO_DETECTOR_MAXDET2000.exists():
        return []
    with open(RESUMO_DETECTOR_MAXDET2000, encoding="utf-8") as f:
        dados = json.load(f)
    por_semente = dados["metrics"]["controlled_seeds_maxdet2000"]
    return [
        {"repetition_index": int(semente), "metrics": metricas_semente}
        for semente, metricas_semente in por_semente.items()
    ]


def agregar_experimento(experimento: str) -> dict:
    if experimento == "01_yolo26_main_maxdet2000":
        reps = carregar_reps_detector_maxdet2000()
    else:
        reps = cc.carregar_todos_resumos_rep(experimento)
    if not reps:
        return {"experiment": experimento, "n_reps_found": 0, "metrics": {}}

    reps = sorted(reps, key=lambda r: r["repetition_index"])
    saida = {"experiment": experimento, "label": ROTULOS.get(experimento, experimento),
           "n_reps_found": len(reps), "metrics": {}}

    for nome_metrica, extrator in ESPEC_METRICAS.get(experimento, []):
        valores = []
        for r in reps:
            try:
                v = extrator(r["metrics"])
            except (KeyError, TypeError):
                continue
            if v is not None:
                valores.append(v)
        if valores:
            saida["metrics"][nome_metrica] = cc.media_dp_ic95(valores)
    return saida


def para_linhas_csv(todos_resultados: list) -> list:
    linhas_tab = []
    for resultado_exp in todos_resultados:
        for nome_metrica, estatisticas in resultado_exp["metrics"].items():
            linhas_tab.append({
                "experiment": resultado_exp["experiment"],
                "label": resultado_exp["label"],
                "metric": nome_metrica,
                "n": estatisticas["n"],
                "mean": estatisticas["mean"],
                "std": estatisticas["std"],
                "ci95_low": estatisticas["ci95_low"],
                "ci95_high": estatisticas["ci95_high"],
            })
    return linhas_tab


def extrair_tempos_treino_yolo() -> dict | None:
    """PT: As 10 repetições de 01_yolo26_main rodaram antes do execution_log.jsonl existir, mas
        o Ultralytics grava o tempo acumulado de treino por época em
        runs/01_yolo26_main/rep_NN/results.csv (coluna `time`, em segundos); a última linha
        de cada repetição dá o tempo total. Usado como alternativa em agregar_tempos() só
        para esse experimento.
    EN: The 10 repetitions of 01_yolo26_main ran before execution_log.jsonl existed, but
        Ultralytics records the cumulative training time per epoch in
        runs/01_yolo26_main/rep_NN/results.csv (`time` column, in seconds); the last row of
        each repetition gives the total time. Used as a fallback in agregar_tempos() for
        that experiment only.
    """
    dir_execucoes_yolo = cc.DIR_EXECUCOES / "01_yolo26_main"
    if not dir_execucoes_yolo.exists():
        return None

    por_rep = []
    for dir_rep in sorted(dir_execucoes_yolo.glob("rep_*")):
        caminho_csv = dir_rep / "results.csv"
        if not caminho_csv.exists():
            continue
        with open(caminho_csv, encoding="utf-8") as f:
            linhas_tab = list(csv.DictReader(f))
        if not linhas_tab:
            continue
        ultimo = linhas_tab[-1]
        try:
            por_rep.append({
                "repeticao": dir_rep.name,
                "n_epocas": int(float(ultimo["epoch"])) + 1,
                "duration_sec": float(ultimo["time"]),
            })
        except (KeyError, ValueError):
            continue

    if not por_rep:
        return None

    duracoes = [r["duration_sec"] for r in por_rep]
    total = sum(duracoes)
    return {
        "fonte": "ultralytics results.csv (coluna 'time') -- rodou antes do execution_log.jsonl existir",
        "n_execucoes_logadas": len(por_rep),
        "n_ok": len(por_rep),
        "n_falhas": 0,
        "tempo_total_sec": round(total, 1),
        "tempo_total_human": cc.duracao_legivel(total),
        "tempo_medio_sec": round(total / len(duracoes), 1),
        "tempo_medio_human": cc.duracao_legivel(total / len(duracoes)),
        "por_repeticao": por_rep,
    }


def extrair_tempos_treino_vlm(experimento: str) -> dict | None:
    """PT: Os experimentos 02-06 (SmolVLM, via
        vlm_common.ajuste_fino_lora/ajuste_fino_completo) não têm relógio interno de treino,
        mas cada época salva um CSV de validação (<rotulo>_epochN_val_predictions.csv) com
        horário de modificação preciso, e o teste final salva outro CSV logo depois do
        treino. Reconstrução aproximada: diferença entre arquivos consecutivos = duração de
        uma época; duração total = (último arquivo - primeiro) + 1 época média (estimativa
        da 1ª época, que não tem marcador de início). Usado como alternativa em
        agregar_tempos() só para repetições sem registro no execution_log.jsonl.
    EN: Experiments 02-06 (SmolVLM, via vlm_common.ajuste_fino_lora/ajuste_fino_completo)
        have no internal training clock, but each epoch saves a validation CSV
        (<tag>_epochN_val_predictions.csv) with a precise modification time, and the final
        test saves another CSV right after training. Approximate reconstruction: difference
        between consecutive files = one epoch's duration; total duration = (last file -
        first) + 1 mean epoch (estimate for the 1st epoch, which has no start marker). Used
        as a fallback in agregar_tempos() only for repetitions with no entry in
        execution_log.jsonl.
    """
    reps = cc.carregar_todos_resumos_rep(experimento)
    if not reps:
        return None

    por_rep = []
    for r in reps:
        rotulo_exp = r.get("metadata", {}).get("tag")
        if not rotulo_exp:
            continue
        arquivos_epoca = sorted(
            cc.DIR_RELATORIOS_EXPERIMENTOS.glob(f"{rotulo_exp}_epoch*_val_predictions.csv"),
            key=lambda p: int(re.search(r"_epoch(\d+)_", p.name).group(1))
        )
        if len(arquivos_epoca) < 2:
            continue
        tempos_modif = [p.stat().st_mtime for p in arquivos_epoca]
        deltas = [b - a for a, b in zip(tempos_modif, tempos_modif[1:])]
        media_epoca = sum(deltas) / len(deltas)

        arquivo_teste = cc.DIR_RELATORIOS_EXPERIMENTOS / f"{rotulo_exp}_real_test_predictions.csv"
        mtime_fim = arquivo_teste.stat().st_mtime if arquivo_teste.exists() else tempos_modif[-1]

        duracao = (mtime_fim - tempos_modif[0]) + media_epoca
        por_rep.append({
            "repeticao": r["repetition_index"],
            "tag": rotulo_exp,
            "n_epocas_com_csv": len(arquivos_epoca),
            "duration_sec": round(duracao, 1),
        })

    if not por_rep:
        return None

    duracoes = [r["duration_sec"] for r in por_rep]
    total = sum(duracoes)
    return {
        "fonte": "aproximado, via horário de modificação dos CSVs de validação por época "
                 "(<tag>_epochN_val_predictions.csv) -- vlm_common.py não registra tempo de "
                 "treino internamente; rodou antes do execution_log.jsonl existir",
        "n_execucoes_logadas": len(por_rep),
        "n_ok": len(por_rep),
        "n_falhas": 0,
        "tempo_total_sec": round(total, 1),
        "tempo_total_human": cc.duracao_legivel(total),
        "tempo_medio_sec": round(total / len(duracoes), 1),
        "tempo_medio_human": cc.duracao_legivel(total / len(duracoes)),
        "por_repeticao": por_rep,
    }


def agregar_tempos() -> dict:
    """PT: Agrega logs_maquina/execution_log.jsonl (uma linha por repetição rodada via
        run_queue.py) em tempo total/médio por experimento. Só cobre repetições
        registradas nesse log.
    EN: Aggregates logs_maquina/execution_log.jsonl (one line per repetition run via
        run_queue.py) into total/mean time per experiment. Only covers repetitions recorded in
        that log.
    """
    registros = cc.carregar_log_execucao()
    por_exp: dict = {}
    for r in registros:
        por_exp.setdefault(r["experiment"], []).append(r)

    saida: dict = {}
    total_todos = 0.0
    for exp, regs in por_exp.items():
        ok = [r for r in regs if r.get("status") == "ok"]
        duracoes = [r["duration_sec"] for r in ok]
        total = sum(duracoes)
        total_todos += total
        saida[exp] = {
            "n_execucoes_logadas": len(regs),
            "n_ok": len(ok),
            "n_falhas": len(regs) - len(ok),
            "tempo_total_sec": round(total, 1),
            "tempo_total_human": cc.duracao_legivel(total),
            "tempo_medio_sec": round(total / len(duracoes), 1) if duracoes else None,
            "tempo_medio_human": cc.duracao_legivel(total / len(duracoes)) if duracoes else None,
        }

    if saida.get("01_yolo26_main", {}).get("n_ok", 0) == 0:
        tempos_yolo = extrair_tempos_treino_yolo()
        if tempos_yolo:
            total_todos += tempos_yolo["tempo_total_sec"]
            saida["01_yolo26_main"] = tempos_yolo

    for exp in ("02_vlm_polymer_real_only", "03_vlm_polymer_real_synth",
                "04_vlm_polymer_bigvlm", "05_smolvlm_size_lora", "06_smolvlm_size_full"):
        if saida.get(exp, {}).get("n_ok", 0) == 0:
            tempos_vlm = extrair_tempos_treino_vlm(exp)
            if tempos_vlm:
                total_todos += tempos_vlm["tempo_total_sec"]
                saida[exp] = tempos_vlm

    sessoes = cc.carregar_jsonl(cc.CAMINHO_SESSOES_FILA)
    saida["__sessoes_da_fila__"] = sessoes
    saida["__tempo_total_geral_sec__"] = round(total_todos, 1)
    saida["__tempo_total_geral_human__"] = cc.duracao_legivel(total_todos)
    return saida


def principal():
    cc.RELATORIO_FINAL_DIR.mkdir(parents=True, exist_ok=True)

    todos_resultados = [agregar_experimento(exp) for exp in list(ESPEC_METRICAS.keys())]

    for r in todos_resultados:
        status = f"{r['n_reps_found']}/10 repetições encontradas" if r["n_reps_found"] else "NENHUMA repetição encontrada"
        print(f"{r['experiment']:35s} {status}")
        for nome_metrica, estatisticas in r["metrics"].items():
            print(f"    {nome_metrica:32s} média={estatisticas['mean']:.4f}  dp={estatisticas['std']:.4f}  "
                  f"IC95%=[{estatisticas['ci95_low']:.4f}, {estatisticas['ci95_high']:.4f}]  (n={estatisticas['n']})")

    caminho_json = cc.RELATORIO_FINAL_DIR / "estatisticas_agregadas.json"
    with open(caminho_json, "w", encoding="utf-8") as f:
        json.dump(todos_resultados, f, indent=2, ensure_ascii=False)

    caminho_csv = cc.RELATORIO_FINAL_DIR / "estatisticas_agregadas.csv"
    pd.DataFrame(para_linhas_csv(todos_resultados)).to_csv(caminho_csv, index=False)


    tempos = agregar_tempos()
    print(f"\nTempo total (todos experimentos, execuções logadas): {tempos['__tempo_total_geral_human__']}")
    for exp, t in tempos.items():
        if exp.startswith("__"):
            continue
        print(f"  {exp:35s} total={t['tempo_total_human']:>10s}  médio={t['tempo_medio_human'] or '-':>10s}  "
              f"(n_ok={t['n_ok']}, falhas={t['n_falhas']})")

    caminho_json_tempos = cc.RELATORIO_FINAL_DIR / "tempos_execucao.json"
    with open(caminho_json_tempos, "w", encoding="utf-8") as f:
        json.dump(tempos, f, indent=2, ensure_ascii=False)

    linhas_tempos = [
        {"experiment": exp, **{k: v for k, v in t.items() if k != "por_repeticao"}}
        for exp, t in tempos.items() if not exp.startswith("__")
    ]
    caminho_csv_tempos = cc.RELATORIO_FINAL_DIR / "tempos_execucao.csv"
    pd.DataFrame(linhas_tempos).to_csv(caminho_csv_tempos, index=False)

    print(f"\nSalvo em:\n  {caminho_json}\n  {caminho_csv}\n"
          f"  {caminho_json_tempos}\n  {caminho_csv_tempos}")

    incompleto = [r["experiment"] for r in todos_resultados if r["n_reps_found"] < cc.N_REPS]
    if incompleto:
        print(f"\nAVISO: experimentos com menos de 10 repetições ainda: {incompleto}")


if __name__ == "__main__":
    principal()
