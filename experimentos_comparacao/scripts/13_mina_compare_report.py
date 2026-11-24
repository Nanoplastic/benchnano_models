"""Tabelas da comparação com os métodos do MiNa / MiNa method comparison tables.

PT: Agrega os 8 summary.json (4 arquiteturas x 2 protocolos, execução única) em duas tabelas
    CSV: métricas de caixa (precisão, recall, mAP50, mAP50:95) em `comparacao_mina.csv` e AP
    de máscara (Benchnano-seg e Mask R-CNN) em `comparacao_mina_mascara.csv`, além de
    `comparacao_mina.json` com as duas.
    Sem teste estatístico (Welch/Bonferroni/Hedges' g): com n=1 por configuração não há
    distribuição para comparar.
    Roda com o que já existir: não falha se alguma trilha ainda não rodou.
    Uso:
    python3 13_mina_compare_report.py
EN: Aggregates the 8 summary.json files (4 architectures x 2 protocols, single run) into two
    CSV tables: box metrics (precision, recall, mAP50, mAP50:95) in `comparacao_mina.csv`
    and mask AP (Benchnano-seg and Mask R-CNN) in `comparacao_mina_mascara.csv`, plus
    `comparacao_mina.json` holding both.
    No statistical test (Welch/Bonferroni/Hedges' g): with n=1 per configuration there is no
    distribution to compare.
    Runs with whatever already exists: does not fail if a track has not run yet.
    Usage:
    python3 13_mina_compare_report.py
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comp_common as cc

# PT: ordem das linhas em cada tabela
# EN: row order in each table
ORDEM_LINHAS = {
    "full": ["09_benchnano_seg", "11_fasterrcnn", "12_maskrcnn", "10_yolov10"],
    "patches": ["09_benchnano_seg", "10_yolov10", "11_fasterrcnn", "12_maskrcnn"],
}
ROTULOS = {
    "09_benchnano_seg": "Benchnano-seg (YOLO26L-seg)",
    "10_yolov10": "YOLOv10",
    "11_fasterrcnn": "Faster R-CNN",
    "12_maskrcnn": "Mask R-CNN",
}

# PT: protocolo full: Benchnano-seg e Mask R-CNN usam o conjunto completo (162 treino/15
#     teste), incluindo as 10 imagens PET mapeadas em 00_build_seg_dataset.py. Faster R-CNN
#     e YOLOv10 usam o mesmo conjunto (162/15) via datasets_yolo26_v2. A variante
#     `__full_matched` (155/14) é mantida só como registro.
# EN: full protocol: Benchnano-seg and Mask R-CNN use the complete pool (162 train/15 test),
#     including the 10 PET images mapped in 00_build_seg_dataset.py. Faster R-CNN and
#     YOLOv10 use the same pool (162/15) via datasets_yolo26_v2. The `__full_matched`
#     variant (155/14) is kept for the record only.
FONTE_PROTOCOLO_COMPLETO = {
    "09_benchnano_seg": "full_fixed",
    "12_maskrcnn": "full_fixed",
    "11_fasterrcnn": "full",
    "10_yolov10": "full",
}


def metricas_caixa(experimento: str, protocolo: str) -> dict | None:
    protocolo = FONTE_PROTOCOLO_COMPLETO[experimento] if protocolo == "full" else protocolo
    s = cc.carregar_resumo(f"{experimento}__{protocolo}")
    if s is None:
        return None
    m = s["metrics"]
    if "aggregate_metrics" in m:  # PT: 10_yolov10, 11_fasterrcnn / EN: 10_yolov10, 11_fasterrcnn
        return m["aggregate_metrics"]
    if "box" in m:  # PT: 09_benchnano_seg / EN: 09_benchnano_seg
        return m["box"]
    return None  # PT: 12_maskrcnn também tem "aggregate_metrics", coberto acima / EN: 12_maskrcnn also has "aggregate_metrics", handled above


def metricas_mascara(experimento: str, protocolo: str) -> dict | None:
    protocolo = FONTE_PROTOCOLO_COMPLETO[experimento] if protocolo == "full" else protocolo
    s = cc.carregar_resumo(f"{experimento}__{protocolo}")
    if s is None:
        return None
    m = s["metrics"]
    return m.get("mask")


def principal():
    cc.RELATORIO_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    linhas_tab = []
    linhas_mascara = []
    ausentes = []

    for protocolo in ("full", "patches"):
        for exp in ORDEM_LINHAS[protocolo]:
            bm = metricas_caixa(exp, protocolo)
            if bm is None:
                ausentes.append(f"{protocolo}: {ROTULOS[exp]}")
                linhas_tab.append({"protocol": protocolo, "method": ROTULOS[exp],
                             "precision": None, "recall": None, "map50": None, "map50_95": None})
                continue
            linhas_tab.append({
                "protocol": protocolo, "method": ROTULOS[exp],
                "precision": bm.get("precision"), "recall": bm.get("recall"),
                "map50": bm.get("map50"), "map50_95": bm.get("map50_95"),
            })
            mm = metricas_mascara(exp, protocolo)
            if mm:
                linhas_mascara.append({"protocol": protocolo, "method": ROTULOS[exp], **mm})

    caminho_json = cc.RELATORIO_FINAL_DIR / "comparacao_mina.json"
    with open(caminho_json, "w", encoding="utf-8") as f:
        json.dump({"box": linhas_tab, "mask": linhas_mascara}, f, indent=2, ensure_ascii=False)

    caminho_csv = cc.RELATORIO_FINAL_DIR / "comparacao_mina.csv"
    with open(caminho_csv, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=["protocol", "method", "precision", "recall", "map50", "map50_95"])
        escritor.writeheader()
        escritor.writerows(linhas_tab)

    # PT: tabela de AP de máscara (CSV)
    # EN: mask AP table (CSV)
    caminho_csv_mascara = cc.RELATORIO_FINAL_DIR / "comparacao_mina_mascara.csv"
    with open(caminho_csv_mascara, "w", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=["protocol", "method", "ap50", "ap75", "ap"])
        escritor.writeheader()
        escritor.writerows(linhas_mascara)

    print(f"[13_mina_compare_report] salvo em:\n  {caminho_json}\n  {caminho_csv}\n  {caminho_csv_mascara}")
    print("\nBox metrics:")
    for r in linhas_tab:
        status = "PENDENTE" if r["precision"] is None else (
            f"P={r['precision']*100:.1f}% R={r['recall']*100:.1f}% "
            f"mAP50={r['map50']*100:.1f}% mAP50:95={r['map50_95']*100:.1f}%")
        print(f"  [{r['protocol']:7s}] {r['method']:28s} {status}")
    print("\nMask AP (Benchnano-seg, Mask R-CNN -- só texto, não tabela):")
    for r in linhas_mascara:
        print(f"  [{r['protocol']:7s}] {r['method']:28s} AP50={r['ap50']*100:.1f}% AP75={r['ap75']*100:.1f}% AP={r['ap']*100:.1f}%")

    if ausentes:
        print(f"\nAVISO: {len(ausentes)} configuração(ões) ainda não rodaram:")
        for m in ausentes:
            print(f"  - {m}")


if __name__ == "__main__":
    principal()
