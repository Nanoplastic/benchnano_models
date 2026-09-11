"""Agrega os 8 summaries (4 arquiteturas x 2 protocolos, execução única --
ver README sobre o prazo de submissão) nas duas tabelas que
`main_pt.tex` já espera (`tab:mina_comparison_full`,
`tab:mina_comparison_patches`): Precisão, Recall, mAP50, mAP50:95 (caixa).
AP de máscara (Benchnano-seg e Mask R-CNN) é reportado à parte, no texto,
não nas tabelas -- mesma convenção do `.tex`.

Sem teste estatístico (Welch/Bonferroni/Hedges' g): com n=1 por
configuração não há distribuição pra comparar -- ver nota no `.tex`.

Roda com o que já existir -- não trava se alguma trilha ainda não rodou.

Uso:
    python3 13_mina_compare_report.py
"""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comp_common as cc

# ordem exata das linhas em cada tabela do main_pt.tex
ROW_ORDER = {
    "full": ["09_benchnano_seg", "11_fasterrcnn", "12_maskrcnn", "10_yolov10"],
    "patches": ["09_benchnano_seg", "10_yolov10", "11_fasterrcnn", "12_maskrcnn"],
}
LABELS = {
    "09_benchnano_seg": "Benchnano-seg (YOLO26L-seg)",
    "10_yolov10": "YOLOv10",
    "11_fasterrcnn": "Faster R-CNN",
    "12_maskrcnn": "Mask R-CNN",
}

# protocolo full: Benchnano-seg e Mask R-CNN foram retreinados com o pool
# completo (162 treino/15 teste) depois de corrigir o bug de nome de arquivo
# em 00_build_seg_dataset.py que descartava 10 imagens PET renomeadas
# (2026-09-10) -- essa é a versão oficial da tabela agora. Faster R-CNN e
# YOLOv10 sempre usaram esse mesmo pool (162/15) via datasets_yolo26_v2, não
# precisam de variante; o `__full_matched` (155/14, etapa intermediária)
# fica só como registro histórico.
FULL_PROTOCOL_SOURCE = {
    "09_benchnano_seg": "full_fixed",
    "12_maskrcnn": "full_fixed",
    "11_fasterrcnn": "full",
    "10_yolov10": "full",
}


def box_metrics(experiment: str, protocol: str) -> dict | None:
    protocol = FULL_PROTOCOL_SOURCE[experiment] if protocol == "full" else protocol
    s = cc.load_summary(f"{experiment}__{protocol}")
    if s is None:
        return None
    m = s["metrics"]
    if "aggregate_metrics" in m:  # 10_yolov10, 11_fasterrcnn
        return m["aggregate_metrics"]
    if "box" in m:  # 09_benchnano_seg
        return m["box"]
    return None  # 12_maskrcnn tem "aggregate_metrics" também -- coberto acima


def mask_metrics(experiment: str, protocol: str) -> dict | None:
    protocol = FULL_PROTOCOL_SOURCE[experiment] if protocol == "full" else protocol
    s = cc.load_summary(f"{experiment}__{protocol}")
    if s is None:
        return None
    m = s["metrics"]
    return m.get("mask")


def main():
    cc.RELATORIO_FINAL_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    mask_rows = []
    missing = []

    for protocol in ("full", "patches"):
        for exp in ROW_ORDER[protocol]:
            bm = box_metrics(exp, protocol)
            if bm is None:
                missing.append(f"{protocol}: {LABELS[exp]}")
                rows.append({"protocol": protocol, "method": LABELS[exp],
                             "precision": None, "recall": None, "map50": None, "map50_95": None})
                continue
            rows.append({
                "protocol": protocol, "method": LABELS[exp],
                "precision": bm.get("precision"), "recall": bm.get("recall"),
                "map50": bm.get("map50"), "map50_95": bm.get("map50_95"),
            })
            mm = mask_metrics(exp, protocol)
            if mm:
                mask_rows.append({"protocol": protocol, "method": LABELS[exp], **mm})

    json_path = cc.RELATORIO_FINAL_DIR / "comparacao_mina.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"box": rows, "mask": mask_rows}, f, indent=2, ensure_ascii=False)

    csv_path = cc.RELATORIO_FINAL_DIR / "comparacao_mina.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["protocol", "method", "precision", "recall", "map50", "map50_95"])
        writer.writeheader()
        writer.writerows(rows)

    # bloco LaTeX -- conferência/auditoria; a edição real do main_pt.tex é
    # feita à mão (ver plano), pra não arriscar desalinhar a tabela já
    # formatada lá.
    tex_lines = ["% Gerado por 13_mina_compare_report.py -- execução única (n=1), sem teste estatístico"]
    for protocol in ("full", "patches"):
        tex_lines.append(f"\n% --- protocolo: {protocol} ---")
        for r in rows:
            if r["protocol"] != protocol:
                continue
            if r["precision"] is None:
                tex_lines.append(f"{r['method']:28s} & PENDENTE & PENDENTE & PENDENTE & PENDENTE \\\\")
            else:
                tex_lines.append(
                    f"{r['method']:28s} & {r['precision']*100:.1f}\\% & {r['recall']*100:.1f}\\% & "
                    f"{r['map50']*100:.1f}\\% & {r['map50_95']*100:.1f}\\% \\\\"
                )
    tex_path = cc.RELATORIO_FINAL_DIR / "comparacao_mina_tabelas.tex"
    tex_path.write_text("\n".join(tex_lines) + "\n")

    print(f"[13_mina_compare_report] salvo em:\n  {json_path}\n  {csv_path}\n  {tex_path}")
    print("\nBox metrics:")
    for r in rows:
        status = "PENDENTE" if r["precision"] is None else (
            f"P={r['precision']*100:.1f}% R={r['recall']*100:.1f}% "
            f"mAP50={r['map50']*100:.1f}% mAP50:95={r['map50_95']*100:.1f}%")
        print(f"  [{r['protocol']:7s}] {r['method']:28s} {status}")
    print("\nMask AP (Benchnano-seg, Mask R-CNN -- só texto, não tabela):")
    for r in mask_rows:
        print(f"  [{r['protocol']:7s}] {r['method']:28s} AP50={r['ap50']*100:.1f}% AP75={r['ap75']*100:.1f}% AP={r['ap']*100:.1f}%")

    if missing:
        print(f"\nAVISO: {len(missing)} configuração(ões) ainda não rodaram:")
        for m in missing:
            print(f"  - {m}")


if __name__ == "__main__":
    main()
