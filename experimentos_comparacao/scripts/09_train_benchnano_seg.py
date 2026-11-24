"""Treino do Benchnano-seg / Benchnano-seg training.

PT: Benchnano-seg: YOLO26L treinado para segmentação de instância antes da classificação de
    tamanho. Mesma família do detector principal do BenchNano, com a cabeça de segmentação
    do Ultralytics (`yolo26l-seg.pt`).
    Dois protocolos:
    --protocol full -- experimentos_comparacao/datasets_baseline_seg/ (gerado por
    00_build_seg_dataset.py)
    --protocol patches -- experimentos_comparacao/datasets_curated_crop/seg/ (gerado por
    08_build_curated_crop_dataset.py)
    Execução única (semente 0).
    Uso:
    python3 09_train_benchnano_seg.py --protocol full [--epochs 150] [--smoke-test]
EN: Benchnano-seg: YOLO26L trained for instance segmentation before size classification.
    Same family as the main BenchNano detector, with the Ultralytics segmentation head
    (`yolo26l-seg.pt`).
    Two protocols:
    --protocol full -- experimentos_comparacao/datasets_baseline_seg/ (built by
    00_build_seg_dataset.py)
    --protocol patches -- experimentos_comparacao/datasets_curated_crop/seg/ (built by
    08_build_curated_crop_dataset.py)
    Single run (seed 0).
    Usage:
    python3 09_train_benchnano_seg.py --protocol full [--epochs 150] [--smoke-test]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comp_common as cc

from ultralytics import YOLO

PESOS = "yolo26l-seg.pt"
SEMENTE = 0


def yaml_dados_para(protocolo: str, conjunto_fixo: bool = False) -> Path:
    if protocolo == "full":
        if conjunto_fixo:
            return cc.RAIZ_COMP / "datasets_baseline_seg_fixed" / "data.yaml"
        return cc.DIR_DATASET_SEG / "data.yaml"
    return cc.DIR_RECORTE_CURADO / "seg" / "data.yaml"


def avaliar_conjunto_teste(caminho_modelo, yaml_dados: Path, dir_raiz: Path) -> dict:
    modelo = YOLO(str(caminho_modelo))
    yaml_dados_teste = dir_raiz / "data_test.yaml"
    if not yaml_dados_teste.exists():
        yaml_dados_teste.write_text(
            f"path: {dir_raiz}\ntrain: images/test\nval: images/test\nnc: 1\nnames:\n  - particle\n"
        )
    metricas = modelo.val(data=str(yaml_dados_teste), split="val", imgsz=1024 if dir_raiz.name != "seg" else 256,
                         conf=0.15, iou=0.7, plots=False, verbose=False)
    return {
        "box": {"precision": float(metricas.box.mp), "recall": float(metricas.box.mr),
                "map50": float(metricas.box.map50), "map50_95": float(metricas.box.map)},
        "mask": {"ap50": float(metricas.seg.map50), "ap75": float(metricas.seg.map75), "ap": float(metricas.seg.map)},
    }


def principal():
    analisador = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    analisador.add_argument("--protocol", required=True, choices=["full", "patches"])
    analisador.add_argument("--epochs", type=int, default=None)
    analisador.add_argument("--smoke-test", action="store_true", help="2 épocas, imgsz reduzido, só pra validar o pipeline")
    analisador.add_argument("--fixed-pool", action="store_true",
                         help="protocolo full só: usa datasets_baseline_seg_fixed/ (162 treino/15 teste, "
                              "com as 10 imagens PET-B recuperadas) em vez de datasets_baseline_seg/ "
                              "(155/14) -- ver 00_build_seg_dataset.py --out-dir")
    args = analisador.parse_args()
    if args.fixed_pool and args.protocol != "full":
        analisador.error("--fixed-pool só se aplica a --protocol full")

    yaml_dados = yaml_dados_para(args.protocol, args.fixed_pool)
    dir_raiz = yaml_dados.parent
    tam_img = 256 if args.protocol == "patches" else 1024
    epocas = args.epochs or (2 if args.smoke_test else 200)
    sufixo_exp = f"{args.protocol}_fixed" if args.fixed_pool else args.protocol
    nome_execucao = f"benchnano_seg_{sufixo_exp}"

    print(f"\n{'='*70}\n[09_train_benchnano_seg] protocolo={args.protocol} epochs={epocas} imgsz={tam_img}\n{'='*70}")
    t0 = time.time()
    modelo = YOLO(PESOS, task="segment")
    modelo.train(
        data=str(yaml_dados), task="segment", epochs=epocas, imgsz=tam_img, batch=-1,
        optimizer="AdamW", lr0=1e-4, lrf=0.1, cos_lr=True, seed=SEMENTE, device="0",
        project=str(cc.DIR_EXECUCOES / "09_benchnano_seg"), name=nome_execucao,
        save_period=-1, patience=max(epocas, 100), plots=True, verbose=True, exist_ok=True,
    )
    duracao = time.time() - t0

    melhor_pt = cc.DIR_EXECUCOES / "09_benchnano_seg" / nome_execucao / "weights" / "best.pt"
    metricas = avaliar_conjunto_teste(melhor_pt, yaml_dados, dir_raiz)
    print(f"[09_train_benchnano_seg] {sufixo_exp}: box={metricas['box']} mask={metricas['mask']}")

    if not args.smoke_test:
        cc.salvar_resumo(f"09_benchnano_seg__{sufixo_exp}", metricas=metricas, metadados_extras={
            "protocol": args.protocol, "fixed_pool": args.fixed_pool, "weights": PESOS, "best_weights": str(melhor_pt),
            "epochs": epocas, "imgsz": tam_img,
            "duration_sec": round(duracao, 1), "duration_human": cc.duracao_legivel(duracao),
        })
    else:
        print(f"[09_train_benchnano_seg] SMOKE TEST OK -- {cc.duracao_legivel(duracao)} pra {epocas} épocas "
              f"({cc.duracao_legivel(duracao / epocas)}/época)")


if __name__ == "__main__":
    principal()
