"""Agrega os resultados de todas as trilhas desta comparação (01-05) +
os números já publicados do BenchNano (YOLO26L + SmolVLM,
`experimentos_controlados/relatorio_final/`) + os números *publicados*
pelo paper do dataset (`comp_common.PUBLISHED_BASELINE`, com o caveat de
split) num único relatório -- JSON, CSV e bloco LaTeX, mesmo padrão de
`experimentos_controlados/scripts/final_report.py`.

Roda com o que já existir -- não trava se alguma trilha ainda não rodou
(avisa no final quais faltam). Rodar de novo a qualquer momento pra
atualizar o relatório conforme as trilhas terminam.

Uso:
    python3 06_compare_report.py
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comp_common as cc

CTRL_STATS_PATH = (cc.DATASETMINA_ROOT / "experimentos_controlados" /
                    "relatorio_final" / "estatisticas_agregadas.json")

DET_MODEL_KEYS = ["yolo26s", "yolov8s", "rtdetr_l", "faster_rcnn_r101", "faster_rcnn_x101"]
SEG_MODEL_KEYS = ["yolov8n_seg", "yolo26n_seg", "mask_rcnn_r101", "mask_rcnn_x101"]
SAM_CONDITIONS = ["sam_grid_zeroshot", "sam_points", "sam_boxes"]

PUBLISHED_DET_KEY = {
    "yolo26s": "yolov26s", "yolov8s": "yolov8s", "rtdetr_l": "rtdetr_l",
    "faster_rcnn_r101": "faster_rcnn_r101", "faster_rcnn_x101": "faster_rcnn_x101",
}
PUBLISHED_SEG_KEY = {
    "yolov8n_seg": "yolov8n_seg", "yolo26n_seg": "yolov26n_seg",
    "mask_rcnn_r101": "mask_rcnn_r101", "mask_rcnn_x101": "mask_rcnn_x101",
}


def load_benchnano_numbers() -> dict:
    """Nossos números já publicados -- YOLO26L (média 10x controlada) e
    resultados de SmolVLM, de experimentos_controlados/relatorio_final/."""
    if not CTRL_STATS_PATH.exists():
        print(f"[06_compare_report] aviso: {CTRL_STATS_PATH} não encontrado "
              f"-- números do BenchNano ficarão de fora do relatório.")
        return {}
    with open(CTRL_STATS_PATH, encoding="utf-8") as f:
        rows = json.load(f)
    out = {}
    for r in rows:
        out[r["experiment"]] = {m: v["mean"] for m, v in r["metrics"].items()}
    return out


def section_detection(benchnano: dict) -> list[dict]:
    rows = []
    yolo26l = benchnano.get("01_yolo26_main", {})
    if yolo26l:
        rows.append({
            "track": "detecção + polímero (4 classes)", "modelo": "YOLO26L (BenchNano, nosso)",
            "split": "travado por imagem-fonte (nosso)",
            "map50": yolo26l.get("map50"), "precision": yolo26l.get("precision"),
            "recall": yolo26l.get("recall"), "f1": None,
        })
    for key in DET_MODEL_KEYS:
        s = cc.load_summary(f"02_yolo_det_baselines__{key}") or cc.load_summary(f"03_rtdetr_baseline__{key}") \
            or cc.load_summary(f"04_detectron2_baselines__{key}")
        pub = cc.PUBLISHED_BASELINE["detection_polymer"].get(PUBLISHED_DET_KEY.get(key, ""))
        if s:
            agg = s["metrics"].get("aggregate_metrics", s["metrics"])
            rows.append({
                "track": "detecção + polímero (4 classes)", "modelo": f"{key} (reproduzido, split nosso)",
                "split": "travado por imagem-fonte (nosso)",
                "map50": agg.get("map50"), "precision": agg.get("precision"),
                "recall": agg.get("recall"), "f1": agg.get("f1"),
            })
        if pub:
            rows.append({
                "track": "detecção + polímero (4 classes)", "modelo": f"{key} (publicado, paper original)",
                "split": "patch aleatório -- NÃO comparável (ver split_caveat)",
                "map50": pub.get("map50"), "precision": pub.get("precision"),
                "recall": pub.get("recall"), "f1": pub.get("f1"),
            })
    return rows


def section_segmentation() -> list[dict]:
    rows = []
    for key in SEG_MODEL_KEYS:
        s = cc.load_summary(f"01_yolo_seg_baselines__{key}") or cc.load_summary(f"04_detectron2_baselines__{key}")
        pub = cc.PUBLISHED_BASELINE["segmentation"].get(PUBLISHED_SEG_KEY.get(key, ""))
        if s:
            m = s["metrics"].get("mask", s["metrics"])
            rows.append({
                "track": "segmentação classe-agnóstica", "modelo": f"{key} (reproduzido, split nosso)",
                "split": "travado por imagem-fonte (nosso)",
                "ap50": m.get("ap50"), "ap75": m.get("ap75"), "ap": m.get("ap"),
            })
        if pub:
            rows.append({
                "track": "segmentação classe-agnóstica", "modelo": f"{key} (publicado, paper original)",
                "split": "patch aleatório -- NÃO comparável (ver split_caveat)",
                "ap50": pub.get("ap50"), "ap75": pub.get("ap75"), "ap": pub.get("ap"),
            })
    return rows


def section_sam() -> list[dict]:
    rows = []
    s = cc.load_summary("05_sam_zeroshot_eval")
    if s:
        for cond in SAM_CONDITIONS:
            m = s["metrics"].get(cond)
            if m:
                rows.append({
                    "track": "SAM zero-shot / promptado", "modelo": f"{cond} (reproduzido, split nosso)",
                    "split": "travado por imagem-fonte (nosso)",
                    "precision": m.get("precision"), "recall": m.get("recall"),
                    "f1": m.get("f1"), "mape": m.get("mape"),
                })
    for key, pub in cc.PUBLISHED_BASELINE["sam_zeroshot"].items():
        rows.append({
            "track": "SAM zero-shot / promptado", "modelo": f"{key} (publicado, paper original)",
            "split": "patch aleatório -- NÃO comparável (ver split_caveat)",
            "precision": pub.get("precision"), "recall": pub.get("recall"),
            "f1": pub.get("f1"), "mape": pub.get("mape"),
        })
    return rows


def section_size_regime_unique(benchnano: dict) -> list[dict]:
    """Regime de tamanho nano/micro via SmolVLM -- sem equivalente no paper
    do dataset (eles não fazem essa tarefa). Não é uma "disputa" de
    métrica, é o diferencial de contribuição do BenchNano."""
    rows = []
    for key, label in [("05_smolvlm_size_lora", "SmolVLM LoRA (regime de tamanho)"),
                        ("06_smolvlm_size_full", "SmolVLM full fine-tune (regime de tamanho)")]:
        m = benchnano.get(key, {})
        if m:
            rows.append({
                "track": "regime de tamanho nano/micro (exclusivo do BenchNano)",
                "modelo": f"{label} (BenchNano, nosso)", "split": "travado (nosso)",
                "test_acc": m.get("test_acc"),
                "nota": "sem equivalente no paper do dataset -- eles não classificam regime de tamanho",
            })
    return rows


def main():
    benchnano = load_benchnano_numbers()
    rows = (section_detection(benchnano) + section_segmentation() +
            section_sam() + section_size_regime_unique(benchnano))

    cc.RELATORIO_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    json_path = cc.RELATORIO_FINAL_DIR / "comparacao.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)

    all_keys = sorted({k for r in rows for k in r})
    csv_path = cc.RELATORIO_FINAL_DIR / "comparacao.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[06_compare_report] {len(rows)} linhas -- salvo em:\n  {json_path}\n  {csv_path}")

    missing = []
    for key in DET_MODEL_KEYS:
        if not (cc.load_summary(f"02_yolo_det_baselines__{key}") or cc.load_summary(f"03_rtdetr_baseline__{key}")
                or cc.load_summary(f"04_detectron2_baselines__{key}")):
            missing.append(f"detecção: {key}")
    for key in SEG_MODEL_KEYS:
        if not (cc.load_summary(f"01_yolo_seg_baselines__{key}") or cc.load_summary(f"04_detectron2_baselines__{key}")):
            missing.append(f"segmentação: {key}")
    if not cc.load_summary("05_sam_zeroshot_eval"):
        missing.append("SAM zero-shot/promptado (05_)")
    if missing:
        print(f"\n[06_compare_report] AVISO: ainda faltam {len(missing)} trilhas reproduzidas "
              f"(o relatório já foi gerado só com o publicado + o que já rodou):")
        for m in missing:
            print(f"  - {m}")


if __name__ == "__main__":
    main()
