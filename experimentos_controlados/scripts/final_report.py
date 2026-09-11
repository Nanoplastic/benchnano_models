"""Agrega as 10 repetições de cada um dos 6 experimentos (+ a avaliação
fim-a-fim pareada) em estatísticas prontas pro artigo: média, desvio-padrão
e IC95% (distribuição t, n=10) por métrica -- mesma disciplina de relato de
Jain (cap. 13.2), já citada em main_V5.tex.

Saída em experimentos_controlados/relatorio_final/:
    estatisticas_agregadas.json  -- tudo, estruturado
    estatisticas_agregadas.csv   -- uma linha por (experimento, métrica)
    tabela_latex.tex             -- bloco de tabela pronto pra colar no artigo

Rodar depois que todos os experimentos + a avaliação pareada tiverem as 10
repetições completas (run_queue.py chama isto automaticamente no final):
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

# quais métricas relatar por experimento, e como extraí-las de metrics{}
METRIC_SPECS = {
    "01_yolo26_main": [
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
    "07_end_to_end_paired": [
        ("D_rate", lambda m: m["D_rate"]),
        ("P_rate_given_D", lambda m: m["P_rate_given_D"]),
        ("smolvlm_accuracy_predicted_box", lambda m: m["smolvlm_accuracy_predicted_box"]),
        ("E_rate_strict_end_to_end", lambda m: m["E_rate_strict_end_to_end"]),
    ],
}

LABELS = {
    "01_yolo26_main": "YOLO26L principal",
    "02_vlm_polymer_real_only": "SmolVLM LoRA polímero (só real)",
    "03_vlm_polymer_real_synth": "SmolVLM LoRA polímero (real+sintético)",
    "04_vlm_polymer_bigvlm": "SmolVLM ~2B polímero (só real)",
    "05_smolvlm_size_lora": "SmolVLM LoRA regime de tamanho",
    "06_smolvlm_size_full": "SmolVLM full fine-tune regime de tamanho",
    "07_end_to_end_paired": "Fim-a-fim pareado (YOLO+SmolVLM regime)",
}


def aggregate_experiment(experiment: str) -> dict:
    reps = cc.load_all_rep_summaries(experiment)
    if not reps:
        return {"experiment": experiment, "n_reps_found": 0, "metrics": {}}

    reps = sorted(reps, key=lambda r: r["repetition_index"])
    out = {"experiment": experiment, "label": LABELS.get(experiment, experiment),
           "n_reps_found": len(reps), "metrics": {}}

    for metric_name, extractor in METRIC_SPECS.get(experiment, []):
        values = []
        for r in reps:
            try:
                v = extractor(r["metrics"])
            except (KeyError, TypeError):
                continue
            if v is not None:
                values.append(v)
        if values:
            out["metrics"][metric_name] = cc.mean_std_ci95(values)
    return out


def to_csv_rows(all_results: list) -> list:
    rows = []
    for exp_result in all_results:
        for metric_name, stats in exp_result["metrics"].items():
            rows.append({
                "experiment": exp_result["experiment"],
                "label": exp_result["label"],
                "metric": metric_name,
                "n": stats["n"],
                "mean": stats["mean"],
                "std": stats["std"],
                "ci95_low": stats["ci95_low"],
                "ci95_high": stats["ci95_high"],
            })
    return rows


def to_latex(all_results: list) -> str:
    lines = [
        r"% Gerado automaticamente por experimentos_controlados/scripts/final_report.py",
        r"% Repetições controladas (n=10, seeds 0-9), média +/- desvio-padrao, IC95% via distribuicao t",
        r"\begin{table}[H]",
        r"    \centering",
        r"    \caption{Resultados agregados de 10 repetições controladas por experimento "
        r"(média $\pm$ desvio-padrão, IC 95\%).}",
        r"    \label{tab:controlled_repetitions}",
        r"    \begin{tabular}{llccc}",
        r"        \hline",
        r"        \textbf{Experimento} & \textbf{Métrica} & \textbf{n} & \textbf{Média} $\pm$ \textbf{DP} & \textbf{IC 95\%} \\",
        r"        \hline",
    ]
    for exp_result in all_results:
        for metric_name, stats in exp_result["metrics"].items():
            label = exp_result["label"].replace("%", r"\%")
            lines.append(
                f"        {label} & {metric_name} & {stats['n']} & "
                f"{stats['mean']*100:.1f}\\% $\\pm$ {stats['std']*100:.1f}\\% & "
                f"[{stats['ci95_low']*100:.1f}\\%, {stats['ci95_high']*100:.1f}\\%] \\\\"
            )
    lines += [r"        \hline", r"    \end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def extract_yolo_training_times() -> dict | None:
    """01_yolo26_main rodou as 10 repetições antes do execution_log.jsonl
    existir, mas o tempo NÃO se perdeu: o ultralytics já grava o tempo
    acumulado de treino por época em runs/01_yolo26_main/rep_NN/results.csv
    (coluna `time`, em segundos) -- a última linha de cada rep dá o tempo
    total de treino daquela repetição. Usado como fallback em
    aggregate_timing() só para esse experimento."""
    yolo_runs_dir = cc.RUNS_DIR / "01_yolo26_main"
    if not yolo_runs_dir.exists():
        return None

    por_rep = []
    for rep_dir in sorted(yolo_runs_dir.glob("rep_*")):
        csv_path = rep_dir / "results.csv"
        if not csv_path.exists():
            continue
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        last = rows[-1]
        try:
            por_rep.append({
                "repeticao": rep_dir.name,
                "n_epocas": int(float(last["epoch"])) + 1,
                "duration_sec": float(last["time"]),
            })
        except (KeyError, ValueError):
            continue

    if not por_rep:
        return None

    durations = [r["duration_sec"] for r in por_rep]
    total = sum(durations)
    return {
        "fonte": "ultralytics results.csv (coluna 'time') -- rodou antes do execution_log.jsonl existir",
        "n_execucoes_logadas": len(por_rep),
        "n_ok": len(por_rep),
        "n_falhas": 0,
        "tempo_total_sec": round(total, 1),
        "tempo_total_human": cc.human_duration(total),
        "tempo_medio_sec": round(total / len(durations), 1),
        "tempo_medio_human": cc.human_duration(total / len(durations)),
        "por_repeticao": por_rep,
    }


def extract_vlm_training_times(experiment: str) -> dict | None:
    """02-06 (SmolVLM, via vlm_common.finetune_lora/finetune_full) não têm
    relógio interno de treino -- só print(), que se perde no terminal. Mas
    cada época salva um CSV de validação (<tag>_epochN_val_predictions.csv)
    com horário de modificação preciso, e o teste final salva outro CSV
    logo depois do treino acabar. Reconstrução aproximada: delta entre
    arquivos consecutivos = duração de uma época; duração total = (último
    arquivo - primeiro) + 1 época média (proxy pra duração da 1a época, que
    não tem marcador de início). Usado como fallback em aggregate_timing()
    só pra reps que já rodaram antes do execution_log.jsonl existir."""
    reps = cc.load_all_rep_summaries(experiment)
    if not reps:
        return None

    por_rep = []
    for r in reps:
        tag = r.get("metadata", {}).get("tag")
        if not tag:
            continue
        epoch_files = sorted(
            cc.EXPERIMENTOS_REPORTS_DIR.glob(f"{tag}_epoch*_val_predictions.csv"),
            key=lambda p: int(re.search(r"_epoch(\d+)_", p.name).group(1))
        )
        if len(epoch_files) < 2:
            continue
        mtimes = [p.stat().st_mtime for p in epoch_files]
        deltas = [b - a for a, b in zip(mtimes, mtimes[1:])]
        avg_epoch = sum(deltas) / len(deltas)

        test_file = cc.EXPERIMENTOS_REPORTS_DIR / f"{tag}_real_test_predictions.csv"
        end_mtime = test_file.stat().st_mtime if test_file.exists() else mtimes[-1]

        duration = (end_mtime - mtimes[0]) + avg_epoch
        por_rep.append({
            "repeticao": r["repetition_index"],
            "tag": tag,
            "n_epocas_com_csv": len(epoch_files),
            "duration_sec": round(duration, 1),
        })

    if not por_rep:
        return None

    durations = [r["duration_sec"] for r in por_rep]
    total = sum(durations)
    return {
        "fonte": "aproximado, via horário de modificação dos CSVs de validação por época "
                 "(<tag>_epochN_val_predictions.csv) -- vlm_common.py não registra tempo de "
                 "treino internamente; rodou antes do execution_log.jsonl existir",
        "n_execucoes_logadas": len(por_rep),
        "n_ok": len(por_rep),
        "n_falhas": 0,
        "tempo_total_sec": round(total, 1),
        "tempo_total_human": cc.human_duration(total),
        "tempo_medio_sec": round(total / len(durations), 1),
        "tempo_medio_human": cc.human_duration(total / len(durations)),
        "por_repeticao": por_rep,
    }


def aggregate_timing() -> dict:
    """Agrega logs_maquina/execution_log.jsonl (uma linha por repetição
    rodada via run_queue.py) em tempo total/médio por experimento -- pro
    artigo relatar tempo de execução junto das métricas (item pedido pelos
    checklists de reprodutibilidade de ML: infraestrutura + tempo de
    execução). Só cobre repetições rodadas DEPOIS que esse log foi
    implementado -- reps anteriores não têm registro de tempo aqui."""
    records = cc.load_execution_log()
    by_exp: dict = {}
    for r in records:
        by_exp.setdefault(r["experiment"], []).append(r)

    out: dict = {}
    total_all = 0.0
    for exp, recs in by_exp.items():
        ok = [r for r in recs if r.get("status") == "ok"]
        durations = [r["duration_sec"] for r in ok]
        total = sum(durations)
        total_all += total
        out[exp] = {
            "n_execucoes_logadas": len(recs),
            "n_ok": len(ok),
            "n_falhas": len(recs) - len(ok),
            "tempo_total_sec": round(total, 1),
            "tempo_total_human": cc.human_duration(total),
            "tempo_medio_sec": round(total / len(durations), 1) if durations else None,
            "tempo_medio_human": cc.human_duration(total / len(durations)) if durations else None,
        }

    if out.get("01_yolo26_main", {}).get("n_ok", 0) == 0:
        yolo_times = extract_yolo_training_times()
        if yolo_times:
            total_all += yolo_times["tempo_total_sec"]
            out["01_yolo26_main"] = yolo_times

    for exp in ("02_vlm_polymer_real_only", "03_vlm_polymer_real_synth",
                "04_vlm_polymer_bigvlm", "05_smolvlm_size_lora", "06_smolvlm_size_full"):
        if out.get(exp, {}).get("n_ok", 0) == 0:
            vlm_times = extract_vlm_training_times(exp)
            if vlm_times:
                total_all += vlm_times["tempo_total_sec"]
                out[exp] = vlm_times

    sessions = cc.load_jsonl(cc.QUEUE_SESSIONS_PATH)
    out["__sessoes_da_fila__"] = sessions
    out["__tempo_total_geral_sec__"] = round(total_all, 1)
    out["__tempo_total_geral_human__"] = cc.human_duration(total_all)
    return out


def main():
    cc.RELATORIO_FINAL_DIR.mkdir(parents=True, exist_ok=True)

    all_results = [aggregate_experiment(exp) for exp in list(METRIC_SPECS.keys())]

    for r in all_results:
        status = f"{r['n_reps_found']}/10 repetições encontradas" if r["n_reps_found"] else "NENHUMA repetição encontrada"
        print(f"{r['experiment']:35s} {status}")
        for metric_name, stats in r["metrics"].items():
            print(f"    {metric_name:32s} média={stats['mean']:.4f}  dp={stats['std']:.4f}  "
                  f"IC95%=[{stats['ci95_low']:.4f}, {stats['ci95_high']:.4f}]  (n={stats['n']})")

    json_path = cc.RELATORIO_FINAL_DIR / "estatisticas_agregadas.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    csv_path = cc.RELATORIO_FINAL_DIR / "estatisticas_agregadas.csv"
    pd.DataFrame(to_csv_rows(all_results)).to_csv(csv_path, index=False)

    latex_path = cc.RELATORIO_FINAL_DIR / "tabela_latex.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(to_latex(all_results))

    timing = aggregate_timing()
    print(f"\nTempo total (todos experimentos, execuções logadas): {timing['__tempo_total_geral_human__']}")
    for exp, t in timing.items():
        if exp.startswith("__"):
            continue
        print(f"  {exp:35s} total={t['tempo_total_human']:>10s}  médio={t['tempo_medio_human'] or '-':>10s}  "
              f"(n_ok={t['n_ok']}, falhas={t['n_falhas']})")

    timing_json_path = cc.RELATORIO_FINAL_DIR / "tempos_execucao.json"
    with open(timing_json_path, "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2, ensure_ascii=False)

    timing_rows = [
        {"experiment": exp, **{k: v for k, v in t.items() if k != "por_repeticao"}}
        for exp, t in timing.items() if not exp.startswith("__")
    ]
    timing_csv_path = cc.RELATORIO_FINAL_DIR / "tempos_execucao.csv"
    pd.DataFrame(timing_rows).to_csv(timing_csv_path, index=False)

    print(f"\nSalvo em:\n  {json_path}\n  {csv_path}\n  {latex_path}\n"
          f"  {timing_json_path}\n  {timing_csv_path}")

    incomplete = [r["experiment"] for r in all_results if r["n_reps_found"] < cc.N_REPS]
    if incomplete:
        print(f"\nAVISO: experimentos com menos de 10 repetições ainda: {incomplete}")


if __name__ == "__main__":
    main()
