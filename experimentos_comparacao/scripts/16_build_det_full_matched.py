"""Constrói uma variante do protocolo full-image, com pool de imagens
IDÊNTICO ao já usado por Benchnano-seg/Mask R-CNN (datasets_baseline_seg),
para treinar Faster R-CNN e YOLOv10 sem a assimetria de 162/15 vs 155/14
documentada em main_pt.tex/main_v7.tex (Section "Comparison with the MiNa
benchmark study").

Causa da assimetria (ver graphify_audit_resolved.md / sessão 2026-09-10):
10 imagens PET-B (fora das 105 micrografias oficiais do MiNa, adicionadas
depois pra substituir 10 imagens PS-nomeadas-mas-PET-anotadas do release
original) só têm caixa delimitadora, nunca tiveram polígono -- por isso
00_build_seg_dataset.py as descarta automaticamente, mas
11_train_fasterrcnn.py/10_train_yolov10.py (protocolo full) não.

Este script NÃO regera nada em datasets_baseline_seg/ nem em
experimentos/datasets_yolo26_v2/ -- só LÊ os dois e escreve output novo:
    experimentos_comparacao/datasets_det_full_matched/{images,labels}/{split}/
        (symlinks, mesmo conteúdo já usado pelo YOLO26L -- só filtrado)
    experimentos_comparacao/datasets_det_full_matched/data.yaml
    experimentos_comparacao/resultados/11_train_fasterrcnn/coco_det_full_matched/{split}.json
        (filtra coco_det_full/{split}.json já existente, sem tocar nele)

Uso:
    python3 16_build_det_full_matched.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comp_common as cc

MATCHED_DET_DIR = cc.COMP_ROOT / "datasets_det_full_matched"
COCO_DET_FULL_DIR = cc.RESULTADOS_DIR / "11_train_fasterrcnn" / "coco_det_full"
COCO_DET_FULL_MATCHED_DIR = cc.RESULTADOS_DIR / "11_train_fasterrcnn" / "coco_det_full_matched"


def allowlist_for(split: str) -> set[str]:
    """Pool real usado por Benchnano-seg/Mask R-CNN -- fonte da verdade."""
    d = cc.SEG_DATASET_DIR / "images" / split
    if not d.exists():
        raise FileNotFoundError(f"{d} não existe -- rode 00_build_seg_dataset.py primeiro")
    return {p.name for p in d.iterdir() if p.is_file()}


def build_yolo_matched():
    for split in ("train", "val", "test"):
        allow = allowlist_for(split)
        img_out = MATCHED_DET_DIR / "images" / split
        lbl_out = MATCHED_DET_DIR / "labels" / split
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        src_img_dir = cc.YOLO_DET_DATASET / "images" / split
        src_lbl_dir = cc.YOLO_DET_DATASET / "labels" / split
        n_linked = 0
        for name in sorted(allow):
            src_img = src_img_dir / name
            if not src_img.exists():
                raise FileNotFoundError(f"esperava {src_img} (presente em datasets_baseline_seg mas ausente em datasets_yolo26_v2 -- pool inconsistente, investigar antes de continuar)")
            dst_img = img_out / name
            if not dst_img.exists():
                dst_img.symlink_to(src_img.resolve())
            src_lbl = src_lbl_dir / (Path(name).stem + ".txt")
            dst_lbl = lbl_out / (Path(name).stem + ".txt")
            if src_lbl.exists() and not dst_lbl.exists():
                dst_lbl.symlink_to(src_lbl.resolve())
            n_linked += 1
        print(f"[16_build_det_full_matched] YOLO {split}: {n_linked} imagens (pool = datasets_baseline_seg/images/{split})")

    data_yaml = MATCHED_DET_DIR / "data.yaml"
    data_yaml.write_text(
        f"path: {MATCHED_DET_DIR}\n"
        f"train: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(cc.CLASSES)}\nnames:\n" + "\n".join(f"  - {c}" for c in cc.CLASSES) + "\n"
    )
    print(f"[16_build_det_full_matched] data.yaml -> {data_yaml}")


def build_coco_matched():
    # val não é usado por 11_train_fasterrcnn.py (só train/test) -- coco_det_full/val.json
    # nunca foi gerado, não precisa aqui também.
    COCO_DET_FULL_MATCHED_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "test"):
        allow = allowlist_for(split)
        src_path = COCO_DET_FULL_DIR / f"{split}.json"
        if not src_path.exists():
            raise FileNotFoundError(f"{src_path} não existe -- rode 11_train_fasterrcnn.py --protocol full pelo menos uma vez antes (gera o coco_det_full base)")
        coco = json.loads(src_path.read_text())

        kept_images = [im for im in coco["images"] if im["file_name"] in allow]
        kept_ids = {im["id"] for im in kept_images}
        kept_anns = [a for a in coco["annotations"] if a["image_id"] in kept_ids]
        missing = allow - {im["file_name"] for im in kept_images}
        if missing:
            raise ValueError(f"{split}: {len(missing)} arquivo(s) do pool esperado não encontrados em {src_path}: {sorted(missing)}")

        out = {"images": kept_images, "annotations": kept_anns, "categories": coco["categories"]}
        out_path = COCO_DET_FULL_MATCHED_DIR / f"{split}.json"
        out_path.write_text(json.dumps(out))
        print(f"[16_build_det_full_matched] COCO {split}: {len(kept_images)} imagens, {len(kept_anns)} anotações -> {out_path}")


def main():
    build_yolo_matched()
    build_coco_matched()
    print("\n[16_build_det_full_matched] OK -- nada em datasets_baseline_seg/, datasets_yolo26_v2/, coco_det_full/ ou nos resultados/runs existentes foi alterado.")


if __name__ == "__main__":
    main()
