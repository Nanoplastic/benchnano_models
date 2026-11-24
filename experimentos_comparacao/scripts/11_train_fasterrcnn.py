"""Treino do Faster R-CNN / Faster R-CNN training.

PT: Faster R-CNN (ResNet-50 FPN, torchvision); ver `rcnn_common.py` para o uso do
    torchvision em vez do Detectron2.
    Dois protocolos:
    --protocol full -- experimentos/datasets_yolo26_v2/ (converte os rótulos YOLO existentes
    para COCO, sem alterar o dataset original; limite de instâncias ativo, pois micrografias
    densas estouram a VRAM)
    --protocol patches -- experimentos_comparacao/datasets_curated_crop/det/ (COCO pronto,
    sem limite; os blocos são pequenos)
    Execução única (semente 0).
    Uso:
    python3 11_train_fasterrcnn.py --protocol full [--epochs 20] [--smoke-test]
EN: Faster R-CNN (ResNet-50 FPN, torchvision); see `rcnn_common.py` for why torchvision is
    used instead of Detectron2.
    Two protocols:
    --protocol full -- experimentos/datasets_yolo26_v2/ (converts the existing YOLO labels
    to COCO, without altering the original dataset; instance cap enabled, since dense
    micrographs exhaust VRAM)
    --protocol patches -- experimentos_comparacao/datasets_curated_crop/det/ (ready COCO, no
    cap; tiles are small)
    Single run (seed 0).
    Usage:
    python3 11_train_fasterrcnn.py --protocol full [--epochs 20] [--smoke-test]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comp_common as cc
import rcnn_common as rc

NOMES_CLASSES = cc.CLASSES_POLIMERO
SEMENTE = 0


def construir_coco_det_de_yolo(particao: str) -> Path:
    """PT: Converte experimentos/datasets_yolo26_v2/{images,labels}/<split> (txt YOLO) para
        JSON COCO sem alterar o dataset original; o resultado fica em cache em resultados/
        para não ser refeito a cada execução.
    EN: Converts experimentos/datasets_yolo26_v2/{images,labels}/<split> (YOLO txt) to COCO
        JSON without altering the original dataset; the result is cached under resultados/
        so it is not rebuilt on every run.
    """
    dir_saida = cc.RESULTADOS_DIR / "11_train_fasterrcnn" / "coco_det_full"
    dir_saida.mkdir(parents=True, exist_ok=True)
    caminho_saida = dir_saida / f"{particao}.json"
    if caminho_saida.exists():
        return caminho_saida

    from PIL import Image
    dir_img = cc.DATASET_DET_YOLO / "images" / particao
    dir_rot = cc.DATASET_DET_YOLO / "labels" / particao
    imagens, anotacoes_coco = [], []
    id_anot = 1
    for id_img, caminho_img in enumerate(sorted(dir_img.glob("*.png")), start=1):
        with Image.open(caminho_img) as im:
            W, H = im.size
        imagens.append({"id": id_img, "file_name": caminho_img.name, "width": W, "height": H})
        caminho_rot = dir_rot / (caminho_img.stem + ".txt")
        if not caminho_rot.exists():
            continue
        for linha in caminho_rot.read_text().splitlines():
            partes = linha.split()
            if len(partes) < 5:
                continue
            id_cls, cx, cy, w, h = int(partes[0]), *map(float, partes[1:5])
            bw, bh = w * W, h * H
            x, y = cx * W - bw / 2, cy * H - bh / 2
            anotacoes_coco.append({
                "id": id_anot, "image_id": id_img, "category_id": id_cls + 1,
                "bbox": [x, y, bw, bh], "area": bw * bh, "iscrowd": 0,
            })
            id_anot += 1
    coco = {"images": imagens, "annotations": anotacoes_coco,
            "categories": [{"id": i + 1, "name": c} for i, c in enumerate(NOMES_CLASSES)]}
    with open(caminho_saida, "w") as f:
        json.dump(coco, f)
    return caminho_saida


def conjunto_para(protocolo: str, particao: str, conjunto_pareado: bool = False):
    # PT: o limite de instâncias vale só no TREINO (evita OOM na atribuição de âncoras); no
    #     teste o GT é sempre completo, senão a avaliação usaria uma amostra aleatória e não
    #     reprodutível do GT nas imagens densas (3 das 15 imagens de teste têm mais de 500
    #     partículas anotadas)
    # EN: the instance cap applies only to TRAINING (avoids OOM in anchor assignment); at
    #     test time the GT is always complete, otherwise evaluation would use a random,
    #     non-reproducible sample of the GT on dense images (3 of the 15 test images have
    #     more than 500 annotated particles)
    if protocolo == "full":
        if conjunto_pareado:
            json_coco = cc.RESULTADOS_DIR / "11_train_fasterrcnn" / "coco_det_full_matched" / f"{particao}.json"
            if not json_coco.exists():
                raise FileNotFoundError(f"{json_coco} não existe -- rode 16_build_det_full_matched.py primeiro")
        else:
            json_coco = construir_coco_det_de_yolo(particao)
        dir_imagens = cc.DATASET_DET_YOLO / "images" / particao
        max_inst_img = cc.MAX_INSTANCIAS_POR_IMAGEM if particao == "train" else None
    else:
        json_coco = cc.DIR_RECORTE_CURADO / "coco_det" / f"{particao}.json"
        dir_imagens = cc.DIR_RECORTE_CURADO / "det" / "images" / particao
        max_inst_img = None
    return rc.ConjuntoCocoDeteccao(json_coco, dir_imagens, max_instances=max_inst_img)


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

    epocas = args.epochs or (1 if args.smoke_test else (30 if args.protocol == "patches" else 100))
    dispositivo = "cuda" if torch.cuda.is_available() else "cpu"
    sufixo_exp = f"{args.protocol}_matched" if args.matched_pool else args.protocol

    print(f"\n{'='*70}\n[11_train_fasterrcnn] protocolo={args.protocol} matched_pool={args.matched_pool} epochs={epocas} device={dispositivo}\n{'='*70}")
    torch.manual_seed(SEMENTE)

    conj_treino = conjunto_para(args.protocol, "train", args.matched_pool)
    conj_teste = conjunto_para(args.protocol, "test", args.matched_pool)
    print(f"[11_train_fasterrcnn] treino={len(conj_treino)} imagens, teste={len(conj_teste)} imagens")

    modelo = rc.construir_fasterrcnn(num_classes_com_fundo=len(NOMES_CLASSES) + 1)
    t0 = time.time()
    historico = rc.laco_treino(modelo, conj_treino, dispositivo, epocas=epocas, tamanho_lote=4 if args.protocol == "full" else 8)
    duracao = time.time() - t0

    dir_saida = cc.DIR_EXECUCOES / "11_fasterrcnn" / sufixo_exp
    dir_saida.mkdir(parents=True, exist_ok=True)
    torch.save(modelo.state_dict(), dir_saida / "model_final.pt")

    metricas = rc.avaliar(modelo, conj_teste, dispositivo)
    print(f"[11_train_fasterrcnn] {sufixo_exp}: {metricas['aggregate_metrics']}")

    if not args.smoke_test:
        cc.salvar_resumo(f"11_fasterrcnn__{sufixo_exp}", metricas=metricas, metadados_extras={
            "protocol": args.protocol, "matched_pool": args.matched_pool, "epochs": epocas,
            "max_instances_per_image": cc.MAX_INSTANCIAS_POR_IMAGEM if args.protocol == "full" else None,
            "model_path": str(dir_saida / "model_final.pt"), "train_history": historico,
            "duration_sec": round(duracao, 1), "duration_human": cc.duracao_legivel(duracao),
        })
    else:
        print(f"[11_train_fasterrcnn] SMOKE TEST OK -- {cc.duracao_legivel(duracao)} pra {epocas} época(s)")


if __name__ == "__main__":
    principal()
