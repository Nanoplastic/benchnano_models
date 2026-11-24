"""Variante de detecção com conjunto pareado / Matched-pool detection variant.

PT: Constrói uma variante do protocolo de imagem inteira com conjunto de imagens IDÊNTICO ao
    de Benchnano-seg/Mask R-CNN (datasets_baseline_seg), para treinar Faster R-CNN e YOLOv10
    sem a assimetria 162/15 vs 155/14.
    Causa da assimetria: 10 imagens PET-B (substitutas de imagens com nome PS mas anotação
    PET) só têm caixa delimitadora, sem polígono; 00_build_seg_dataset.py as descarta, mas
    11_train_fasterrcnn.py/10_train_yolov10.py (protocolo full) não.
    Não regera nada em datasets_baseline_seg/ nem em experimentos/datasets_yolo26_v2/; só lê
    os dois e escreve:
    experimentos_comparacao/datasets_det_full_matched/{images,labels}/{split}/ (symlinks,
    mesmo conteúdo do YOLO26L, filtrado)
    experimentos_comparacao/datasets_det_full_matched/data.yaml
    experimentos_comparacao/resultados/11_train_fasterrcnn/coco_det_full_matched/{split}.json
    (filtro de coco_det_full/{split}.json, sem alterá-lo)
    Uso:
    python3 16_build_det_full_matched.py
EN: Builds a variant of the full-image protocol with an image pool IDENTICAL to that of
    Benchnano-seg/Mask R-CNN (datasets_baseline_seg), to train Faster R-CNN and YOLOv10
    without the 162/15 vs 155/14 asymmetry.
    Cause of the asymmetry: 10 PET-B images (replacing images named PS but annotated PET)
    only have bounding boxes, no polygons; 00_build_seg_dataset.py drops them, but
    11_train_fasterrcnn.py/10_train_yolov10.py (full protocol) do not.
    Regenerates nothing in datasets_baseline_seg/ or experimentos/datasets_yolo26_v2/; it
    only reads both and writes:
    experimentos_comparacao/datasets_det_full_matched/{images,labels}/{split}/ (symlinks,
    same content as YOLO26L, filtered)
    experimentos_comparacao/datasets_det_full_matched/data.yaml
    experimentos_comparacao/resultados/11_train_fasterrcnn/coco_det_full_matched/{split}.json
    (filtered from coco_det_full/{split}.json, leaving it untouched)
    Usage:
    python3 16_build_det_full_matched.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comp_common as cc

DIR_DET_PAREADO = cc.RAIZ_COMP / "datasets_det_full_matched"
DIR_COCO_DET_COMPLETO = cc.RESULTADOS_DIR / "11_train_fasterrcnn" / "coco_det_full"
DIR_COCO_DET_COMPLETO_PAREADO = cc.RESULTADOS_DIR / "11_train_fasterrcnn" / "coco_det_full_matched"


def lista_permitida_para(particao: str) -> set[str]:
    """PT: Conjunto real usado por Benchnano-seg/Mask R-CNN (fonte da verdade).
    EN: Actual pool used by Benchnano-seg/Mask R-CNN (source of truth).
    """
    d = cc.DIR_DATASET_SEG / "images" / particao
    if not d.exists():
        raise FileNotFoundError(f"{d} não existe -- rode 00_build_seg_dataset.py primeiro")
    return {p.name for p in d.iterdir() if p.is_file()}


def construir_yolo_pareado():
    for particao in ("train", "val", "test"):
        permitidos = lista_permitida_para(particao)
        saida_img = DIR_DET_PAREADO / "images" / particao
        saida_rot = DIR_DET_PAREADO / "labels" / particao
        saida_img.mkdir(parents=True, exist_ok=True)
        saida_rot.mkdir(parents=True, exist_ok=True)

        dir_img_origem = cc.DATASET_DET_YOLO / "images" / particao
        dir_rot_origem = cc.DATASET_DET_YOLO / "labels" / particao
        n_ligados = 0
        for nome in sorted(permitidos):
            img_origem = dir_img_origem / nome
            if not img_origem.exists():
                raise FileNotFoundError(f"esperava {img_origem} (presente em datasets_baseline_seg mas ausente em datasets_yolo26_v2 -- pool inconsistente, investigar antes de continuar)")
            img_destino = saida_img / nome
            if not img_destino.exists():
                img_destino.symlink_to(img_origem.resolve())
            rot_origem = dir_rot_origem / (Path(nome).stem + ".txt")
            rot_destino = saida_rot / (Path(nome).stem + ".txt")
            if rot_origem.exists() and not rot_destino.exists():
                rot_destino.symlink_to(rot_origem.resolve())
            n_ligados += 1
        print(f"[16_build_det_full_matched] YOLO {particao}: {n_ligados} imagens (pool = datasets_baseline_seg/images/{particao})")

    yaml_dados = DIR_DET_PAREADO / "data.yaml"
    yaml_dados.write_text(
        f"path: {DIR_DET_PAREADO}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(cc.CLASSES_POLIMERO)}\nnames:\n" + "\n".join(f"  - {c}" for c in cc.CLASSES_POLIMERO) + "\n"
    )
    print(f"[16_build_det_full_matched] data.yaml -> {yaml_dados}")


def construir_coco_pareado():
    # PT: val não é usado por 11_train_fasterrcnn.py (só train/test); coco_det_full/val.json
    #     não existe e não é necessário aqui
    # EN: val is not used by 11_train_fasterrcnn.py (train/test only);
    #     coco_det_full/val.json does not exist and is not needed here
    DIR_COCO_DET_COMPLETO_PAREADO.mkdir(parents=True, exist_ok=True)
    for particao in ("train", "test"):
        permitidos = lista_permitida_para(particao)
        caminho_origem = DIR_COCO_DET_COMPLETO / f"{particao}.json"
        if not caminho_origem.exists():
            raise FileNotFoundError(f"{caminho_origem} não existe -- rode 11_train_fasterrcnn.py --protocol full pelo menos uma vez antes (gera o coco_det_full base)")
        coco = json.loads(caminho_origem.read_text())

        imagens_mantidas = [im for im in coco["images"] if im["file_name"] in permitidos]
        ids_mantidos = {im["id"] for im in imagens_mantidas}
        anots_mantidas = [a for a in coco["annotations"] if a["image_id"] in ids_mantidos]
        ausentes = permitidos - {im["file_name"] for im in imagens_mantidas}
        if ausentes:
            raise ValueError(f"{particao}: {len(ausentes)} arquivo(s) do pool esperado não encontrados em {caminho_origem}: {sorted(ausentes)}")

        saida = {"images": imagens_mantidas, "annotations": anots_mantidas, "categories": coco["categories"]}
        caminho_saida = DIR_COCO_DET_COMPLETO_PAREADO / f"{particao}.json"
        caminho_saida.write_text(json.dumps(saida))
        print(f"[16_build_det_full_matched] COCO {particao}: {len(imagens_mantidas)} imagens, {len(anots_mantidas)} anotações -> {caminho_saida}")


def principal():
    construir_yolo_pareado()
    construir_coco_pareado()
    print("\n[16_build_det_full_matched] OK -- nada em datasets_baseline_seg/, datasets_yolo26_v2/, coco_det_full/ ou nos resultados/runs existentes foi alterado.")


if __name__ == "__main__":
    principal()
