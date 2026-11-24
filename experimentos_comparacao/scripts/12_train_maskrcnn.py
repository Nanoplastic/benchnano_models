"""Treino do Mask R-CNN / Mask R-CNN training.

PT: Mask R-CNN (ResNet-50 FPN, torchvision); ver `rcnn_common.py` para o uso do torchvision
    em vez do Detectron2.
    Dois protocolos, ambos com COCO-seg pronto (classe única "particle"):
    --protocol full -- experimentos_comparacao/datasets_baseline_seg/coco/ (gerado por
    00_build_seg_dataset.py; limite de instâncias ativo, pelo mesmo motivo do Faster R-CNN)
    --protocol patches -- experimentos_comparacao/datasets_curated_crop/coco_seg/ (sem
    limite; os blocos são pequenos)
    Reporta mAP de caixa e AP de máscara (avaliada separadamente para Benchnano-seg e Mask
    R-CNN). Execução única (semente 0).
    Uso:
    python3 12_train_maskrcnn.py --protocol full [--epochs 20] [--smoke-test]
EN: Mask R-CNN (ResNet-50 FPN, torchvision); see `rcnn_common.py` for why torchvision is
    used instead of Detectron2.
    Two protocols, both with ready COCO-seg (single class "particle"):
    --protocol full -- experimentos_comparacao/datasets_baseline_seg/coco/ (built by
    00_build_seg_dataset.py; instance cap enabled, for the same reason as Faster R-CNN)
    --protocol patches -- experimentos_comparacao/datasets_curated_crop/coco_seg/ (no cap;
    tiles are small)
    Reports box mAP and mask AP (evaluated separately for Benchnano-seg and Mask R-CNN).
    Single run (seed 0).
    Usage:
    python3 12_train_maskrcnn.py --protocol full [--epochs 20] [--smoke-test]
"""
import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comp_common as cc
import rcnn_common as rc

SEMENTE = 0


def conjunto_para(protocolo: str, particao: str, conjunto_fixo: bool = False):
    # PT: o limite de instâncias vale só no TREINO (evita OOM na atribuição de âncoras); no
    #     teste o GT é sempre completo, senão a avaliação usaria uma amostra aleatória e não
    #     reprodutível do GT nas imagens densas (2 das 15 imagens de teste têm mais de 500
    #     partículas anotadas)
    # EN: the instance cap applies only to TRAINING (avoids OOM in anchor assignment); at
    #     test time the GT is always complete, otherwise evaluation would use a random,
    #     non-reproducible sample of the GT on dense images (2 of the 15 test images have
    #     more than 500 annotated particles)
    if protocolo == "full":
        dir_seg = (cc.RAIZ_COMP / "datasets_baseline_seg_fixed") if conjunto_fixo else cc.DIR_DATASET_SEG
        json_coco = dir_seg / "coco" / f"{particao}.json"
        dir_imagens = dir_seg / "images" / particao
        max_inst_img = cc.MAX_INSTANCIAS_POR_IMAGEM if particao == "train" else None
    else:
        json_coco = cc.DIR_RECORTE_CURADO / "coco_seg" / f"{particao}.json"
        dir_imagens = cc.DIR_RECORTE_CURADO / "seg" / "images" / particao
        max_inst_img = None
    return rc.ConjuntoCocoSegmentacao(json_coco, dir_imagens, max_instances=max_inst_img)


def principal():
    analisador = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    analisador.add_argument("--protocol", required=True, choices=["full", "patches"])
    analisador.add_argument("--epochs", type=int, default=None)
    analisador.add_argument("--smoke-test", action="store_true")
    analisador.add_argument("--fixed-pool", action="store_true",
                         help="protocolo full só: usa datasets_baseline_seg_fixed/ (162 treino/15 teste, "
                              "com as 10 imagens PET-B recuperadas) em vez de datasets_baseline_seg/ (155/14)")
    args = analisador.parse_args()
    if args.fixed_pool and args.protocol != "full":
        analisador.error("--fixed-pool só se aplica a --protocol full")

    epocas = args.epochs or (1 if args.smoke_test else (30 if args.protocol == "patches" else 100))
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    sufixo_exp = f"{args.protocol}_fixed" if args.fixed_pool else args.protocol

    print(f"\n{'='*70}\n[12_train_maskrcnn] protocolo={args.protocol} fixed_pool={args.fixed_pool} epochs={epocas} device={dispositivo}\n{'='*70}")
    torch.manual_seed(SEMENTE)

    conj_treino = conjunto_para(args.protocol, "train", args.fixed_pool)
    conj_teste = conjunto_para(args.protocol, "test", args.fixed_pool)
    print(f"[12_train_maskrcnn] treino={len(conj_treino)} imagens, teste={len(conj_teste)} imagens")

    modelo = rc.construir_maskrcnn(num_classes_com_fundo=2)  # PT: fundo + partícula / EN: background + particle
    t0 = time.time()
    historico = rc.laco_treino(modelo, conj_treino, dispositivo, epocas=epocas, tamanho_lote=2 if args.protocol == "full" else 6)
    duracao = time.time() - t0

    dir_saida = cc.DIR_EXECUCOES / "12_maskrcnn" / sufixo_exp
    dir_saida.mkdir(parents=True, exist_ok=True)
    torch.save(modelo.state_dict(), dir_saida / "model_final.pt")

    metricas = rc.avaliar(modelo, conj_teste, dispositivo, avaliar_mascaras=True)
    print(f"[12_train_maskrcnn] {sufixo_exp}: {metricas['aggregate_metrics']}")

    if not args.smoke_test:
        cc.salvar_resumo(f"12_maskrcnn__{sufixo_exp}", metricas=metricas, metadados_extras={
            "protocol": args.protocol, "fixed_pool": args.fixed_pool, "epochs": epocas,
            "max_instances_per_image": cc.MAX_INSTANCIAS_POR_IMAGEM if args.protocol == "full" else None,
            "model_path": str(dir_saida / "model_final.pt"), "train_history": historico,
            "duration_sec": round(duracao, 1), "duration_human": cc.duracao_legivel(duracao),
        })
    else:
        print(f"[12_train_maskrcnn] SMOKE TEST OK -- {cc.duracao_legivel(duracao)} pra {epocas} época(s)")


if __name__ == "__main__":
    principal()
