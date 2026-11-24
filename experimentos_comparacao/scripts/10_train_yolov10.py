"""Treino do YOLOv10 / YOLOv10 training.

PT: YOLOv10, uma das arquiteturas com código de treino disponível no repositório do MiNa. Variante `s`, a mesma convenção de tamanho das
    demais comparações.
    Dois protocolos:
    --protocol full -- experimentos/datasets_yolo26_v2/ (4 classes)
    --protocol patches -- experimentos_comparacao/datasets_curated_crop/det/
    Execução única (semente 0).
    Uso:
    python3 10_train_yolov10.py --protocol full [--epochs 150] [--smoke-test]
EN: YOLOv10, one of the architectures with training code available in the MiNa repository. Variant `s`, the same size convention as the other
    comparisons.
    Two protocols:
    --protocol full -- experimentos/datasets_yolo26_v2/ (4 classes)
    --protocol patches -- experimentos_comparacao/datasets_curated_crop/det/
    Single run (seed 0).
    Usage:
    python3 10_train_yolov10.py --protocol full [--epochs 150] [--smoke-test]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comp_common as cc

from ultralytics import YOLO

PESOS = "yolov10s.pt"
SEMENTE = 0
NOMES_CLASSES = cc.CLASSES_POLIMERO


def yaml_dados_para(protocolo: str, conjunto_pareado: bool = False) -> Path:
    if protocolo == "full":
        return cc.YAML_DADOS_DET_PAREADO if conjunto_pareado else cc.YAML_DADOS_DET
    return cc.DIR_RECORTE_CURADO / "det" / "data.yaml"


def avaliar_conjunto_teste(caminho_modelo, dir_raiz: Path, tam_img: int) -> dict:
    modelo = YOLO(str(caminho_modelo))
    yaml_dados_teste = dir_raiz / "data_test.yaml"
    if not yaml_dados_teste.exists():
        yaml_dados_teste.write_text(
            f"path: {dir_raiz}\ntrain: images/test\nval: images/test\n"
            f"nc: {len(NOMES_CLASSES)}\nnames:\n" + "\n".join(f"  - {c}" for c in NOMES_CLASSES) + "\n"
        )
    metricas = modelo.val(data=str(yaml_dados_teste), split="val", imgsz=tam_img,
                         conf=0.15, iou=0.7, plots=False, verbose=False)
    agregado = {
        "precision": float(metricas.box.mp), "recall": float(metricas.box.mr),
        "map50": float(metricas.box.map50), "map50_95": float(metricas.box.map),
        "f1": float(2 * metricas.box.mp * metricas.box.mr / (metricas.box.mp + metricas.box.mr))
              if (metricas.box.mp + metricas.box.mr) > 0 else 0.0,
    }
    return {"aggregate_metrics": agregado}


def principal():
    analisador = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    analisador.add_argument("--protocol", required=True, choices=["full", "patches"])
    analisador.add_argument("--epochs", type=int, default=None)
    analisador.add_argument("--smoke-test", action="store_true")
    analisador.add_argument("--matched-pool", action="store_true",
                         help="protocolo full só: usa o mesmo pool de imagens de datasets_baseline_seg/ "
                              "(155 treino/14 teste) em vez do dataset completo (162/15) -- ver "
                              "16_build_det_full_matched.py")
    args = analisador.parse_args()
    if args.matched_pool and args.protocol != "full":
        analisador.error("--matched-pool só se aplica a --protocol full")

    yaml_dados = yaml_dados_para(args.protocol, args.matched_pool)
    dir_raiz = yaml_dados.parent
    tam_img = 256 if args.protocol == "patches" else 1024
    epocas = args.epochs or (2 if args.smoke_test else 200)
    sufixo_exp = f"{args.protocol}_matched" if args.matched_pool else args.protocol
    nome_execucao = f"yolov10_{sufixo_exp}"

    print(f"\n{'='*70}\n[10_train_yolov10] protocolo={args.protocol} epochs={epocas} imgsz={tam_img}\n{'='*70}")
    t0 = time.time()
    modelo = YOLO(PESOS, task="detect")
    modelo.train(
        data=str(yaml_dados), task="detect", epochs=epocas, imgsz=tam_img, batch=-1,
        optimizer="AdamW", lr0=1e-4, lrf=0.1, cos_lr=True, seed=SEMENTE, device="0",
        project=str(cc.DIR_EXECUCOES / "10_yolov10"), name=nome_execucao,
        save_period=-1, patience=max(epocas, 100), plots=True, verbose=True, exist_ok=True,
    )
    duracao = time.time() - t0

    melhor_pt = cc.DIR_EXECUCOES / "10_yolov10" / nome_execucao / "weights" / "best.pt"
    metricas = avaliar_conjunto_teste(melhor_pt, dir_raiz, tam_img)
    print(f"[10_train_yolov10] {args.protocol}: {metricas['aggregate_metrics']}")

    if not args.smoke_test:
        cc.salvar_resumo(f"10_yolov10__{sufixo_exp}", metricas=metricas, metadados_extras={
            "protocol": args.protocol, "matched_pool": args.matched_pool, "weights": PESOS, "best_weights": str(melhor_pt),
            "epochs": epocas, "imgsz": tam_img,
            "duration_sec": round(duracao, 1), "duration_human": cc.duracao_legivel(duracao),
        })
    else:
        print(f"[10_train_yolov10] SMOKE TEST OK -- {cc.duracao_legivel(duracao)} pra {epocas} épocas "
              f"({cc.duracao_legivel(duracao / epocas)}/época)")


if __name__ == "__main__":
    principal()
