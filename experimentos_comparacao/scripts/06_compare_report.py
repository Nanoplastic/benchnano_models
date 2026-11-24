"""Relatório consolidado da comparação / Consolidated comparison report.

PT: Reúne num único relatório (JSON e CSV) os resultados das trilhas desta comparação
    (01-05), os resultados do BenchNano (YOLO26L + SmolVLM,
    `experimentos_controlados/relatorio_final/`) e os valores de referência publicados com o
    dataset MiNa (`comp_common.REFERENCIA_PUBLICADA`, com a ressalva sobre a partição), no
    mesmo padrão de `experimentos_controlados/scripts/final_report.py`.
    Roda com o que já existir: não falha se alguma trilha ainda não rodou (lista as
    faltantes no final). Pode ser rodado de novo a qualquer momento para atualizar o
    relatório.
    Uso:
    python3 06_compare_report.py
EN: Gathers into a single report (JSON and CSV) the results of this comparison's tracks
    (01-05), the BenchNano results (YOLO26L + SmolVLM,
    `experimentos_controlados/relatorio_final/`) and the reference values published with the
    MiNa dataset (`comp_common.REFERENCIA_PUBLICADA`, with the split caveat), following the
    pattern of `experimentos_controlados/scripts/final_report.py`.
    Runs with whatever already exists: does not fail if a track has not run yet (lists the
    missing ones at the end). Can be rerun at any time to refresh the report.
    Usage:
    python3 06_compare_report.py
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comp_common as cc

CAMINHO_ESTATISTICAS_CTRL = (cc.RAIZ_PROJETO_ORIGEM / "experimentos_controlados" /
                    "relatorio_final" / "estatisticas_agregadas.json")

CHAVES_MODELOS_DET = ["yolo26s", "yolov8s", "rtdetr_l", "faster_rcnn_r101", "faster_rcnn_x101"]
CHAVES_MODELOS_SEG = ["yolov8n_seg", "yolo26n_seg", "mask_rcnn_r101", "mask_rcnn_x101"]
CONDICOES_SAM = ["sam_grid_zeroshot", "sam_points", "sam_boxes"]

CHAVE_DET_PUBLICADA = {
    "yolo26s": "yolov26s", "yolov8s": "yolov8s", "rtdetr_l": "rtdetr_l",
    "faster_rcnn_r101": "faster_rcnn_r101", "faster_rcnn_x101": "faster_rcnn_x101",
}
CHAVE_SEG_PUBLICADA = {
    "yolov8n_seg": "yolov8n_seg", "yolo26n_seg": "yolov26n_seg",
    "mask_rcnn_r101": "mask_rcnn_r101", "mask_rcnn_x101": "mask_rcnn_x101",
}


def carregar_numeros_benchnano() -> dict:
    """PT: Resultados do BenchNano: YOLO26L (média das 10 repetições controladas) e SmolVLM, de
        experimentos_controlados/relatorio_final/.
    EN: BenchNano results: YOLO26L (mean of the 10 controlled repetitions) and SmolVLM, from
        experimentos_controlados/relatorio_final/.
    """
    if not CAMINHO_ESTATISTICAS_CTRL.exists():
        print(f"[06_compare_report] aviso: {CAMINHO_ESTATISTICAS_CTRL} não encontrado "
              f"-- números do BenchNano ficarão de fora do relatório.")
        return {}
    with open(CAMINHO_ESTATISTICAS_CTRL, encoding="utf-8") as f:
        linhas_tab = json.load(f)
    saida = {}
    for r in linhas_tab:
        saida[r["experiment"]] = {m: v["mean"] for m, v in r["metrics"].items()}
    return saida


def secao_deteccao(benchnano: dict) -> list[dict]:
    linhas_tab = []
    yolo26l = benchnano.get("01_yolo26_main", {})
    if yolo26l:
        linhas_tab.append({
            "track": "detecção + polímero (4 classes)", "modelo": "YOLO26L (BenchNano, nosso)",
            "split": "travado por imagem-fonte (nosso)",
            "map50": yolo26l.get("map50"), "precision": yolo26l.get("precision"),
            "recall": yolo26l.get("recall"), "f1": None,
        })
    for chave in CHAVES_MODELOS_DET:
        s = cc.carregar_resumo(f"02_yolo_det_baselines__{chave}") or cc.carregar_resumo(f"03_rtdetr_baseline__{chave}") \
            or cc.carregar_resumo(f"04_detectron2_baselines__{chave}")
        pub = cc.REFERENCIA_PUBLICADA["detection_polymer"].get(CHAVE_DET_PUBLICADA.get(chave, ""))
        if s:
            agregado = s["metrics"].get("aggregate_metrics", s["metrics"])
            linhas_tab.append({
                "track": "detecção + polímero (4 classes)", "modelo": f"{chave} (reproduzido, split nosso)",
                "split": "travado por imagem-fonte (nosso)",
                "map50": agregado.get("map50"), "precision": agregado.get("precision"),
                "recall": agregado.get("recall"), "f1": agregado.get("f1"),
            })
        if pub:
            linhas_tab.append({
                "track": "detecção + polímero (4 classes)", "modelo": f"{chave} (referência publicada do dataset)",
                "split": "patch aleatório -- NÃO comparável (ver split_caveat)",
                "map50": pub.get("map50"), "precision": pub.get("precision"),
                "recall": pub.get("recall"), "f1": pub.get("f1"),
            })
    return linhas_tab


def secao_segmentacao() -> list[dict]:
    linhas_tab = []
    for chave in CHAVES_MODELOS_SEG:
        s = cc.carregar_resumo(f"01_yolo_seg_baselines__{chave}") or cc.carregar_resumo(f"04_detectron2_baselines__{chave}")
        pub = cc.REFERENCIA_PUBLICADA["segmentation"].get(CHAVE_SEG_PUBLICADA.get(chave, ""))
        if s:
            m = s["metrics"].get("mask", s["metrics"])
            linhas_tab.append({
                "track": "segmentação classe-agnóstica", "modelo": f"{chave} (reproduzido, split nosso)",
                "split": "travado por imagem-fonte (nosso)",
                "ap50": m.get("ap50"), "ap75": m.get("ap75"), "ap": m.get("ap"),
            })
        if pub:
            linhas_tab.append({
                "track": "segmentação classe-agnóstica", "modelo": f"{chave} (referência publicada do dataset)",
                "split": "patch aleatório -- NÃO comparável (ver split_caveat)",
                "ap50": pub.get("ap50"), "ap75": pub.get("ap75"), "ap": pub.get("ap"),
            })
    return linhas_tab


def secao_sam() -> list[dict]:
    linhas_tab = []
    s = cc.carregar_resumo("05_sam_zeroshot_eval")
    if s:
        for condicao in CONDICOES_SAM:
            m = s["metrics"].get(condicao)
            if m:
                linhas_tab.append({
                    "track": "SAM zero-shot / promptado", "modelo": f"{condicao} (reproduzido, split nosso)",
                    "split": "travado por imagem-fonte (nosso)",
                    "precision": m.get("precision"), "recall": m.get("recall"),
                    "f1": m.get("f1"), "mape": m.get("mape"),
                })
    for chave, pub in cc.REFERENCIA_PUBLICADA["sam_zeroshot"].items():
        linhas_tab.append({
            "track": "SAM zero-shot / promptado", "modelo": f"{chave} (referência publicada do dataset)",
            "split": "patch aleatório -- NÃO comparável (ver split_caveat)",
            "precision": pub.get("precision"), "recall": pub.get("recall"),
            "f1": pub.get("f1"), "mape": pub.get("mape"),
        })
    return linhas_tab


def secao_regime_tamanho_unico(benchnano: dict) -> list[dict]:
    """PT: Regime de tamanho nano/micro via SmolVLM: tarefa sem equivalente nos métodos de
        referência do dataset, reportada à parte (não é uma comparação de métrica).
    EN: Nano/micro size regime via SmolVLM: a task with no counterpart in the dataset's
        reference methods, reported separately (not a metric comparison).
    """
    linhas_tab = []
    for chave, rotulo in [("05_smolvlm_size_lora", "SmolVLM LoRA (regime de tamanho)"),
                        ("06_smolvlm_size_full", "SmolVLM full fine-tune (regime de tamanho)")]:
        m = benchnano.get(chave, {})
        if m:
            linhas_tab.append({
                "track": "regime de tamanho nano/micro (exclusivo do BenchNano)",
                "modelo": f"{rotulo} (BenchNano, nosso)", "split": "travado (nosso)",
                "test_acc": m.get("test_acc"),
                "nota": "sem equivalente nos métodos de referência do dataset -- eles não classificam regime de tamanho",
            })
    return linhas_tab


def principal():
    benchnano = carregar_numeros_benchnano()
    linhas_tab = (secao_deteccao(benchnano) + secao_segmentacao() +
            secao_sam() + secao_regime_tamanho_unico(benchnano))

    cc.RELATORIO_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    caminho_json = cc.RELATORIO_FINAL_DIR / "comparacao.json"
    with open(caminho_json, "w", encoding="utf-8") as f:
        json.dump(linhas_tab, f, indent=2, ensure_ascii=False)

    todas_chaves = sorted({k for r in linhas_tab for k in r})
    caminho_csv = cc.RELATORIO_FINAL_DIR / "comparacao.csv"
    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=todas_chaves)
        escritor.writeheader()
        escritor.writerows(linhas_tab)

    print(f"[06_compare_report] {len(linhas_tab)} linhas -- salvo em:\n  {caminho_json}\n  {caminho_csv}")

    ausentes = []
    for chave in CHAVES_MODELOS_DET:
        if not (cc.carregar_resumo(f"02_yolo_det_baselines__{chave}") or cc.carregar_resumo(f"03_rtdetr_baseline__{chave}")
                or cc.carregar_resumo(f"04_detectron2_baselines__{chave}")):
            ausentes.append(f"detecção: {chave}")
    for chave in CHAVES_MODELOS_SEG:
        if not (cc.carregar_resumo(f"01_yolo_seg_baselines__{chave}") or cc.carregar_resumo(f"04_detectron2_baselines__{chave}")):
            ausentes.append(f"segmentação: {chave}")
    if not cc.carregar_resumo("05_sam_zeroshot_eval"):
        ausentes.append("SAM zero-shot/promptado (05_)")
    if ausentes:
        print(f"\n[06_compare_report] AVISO: ainda faltam {len(ausentes)} trilhas reproduzidas "
              f"(o relatório já foi gerado só com as referências + o que já rodou):")
        for m in ausentes:
            print(f"  - {m}")


if __name__ == "__main__":
    principal()
