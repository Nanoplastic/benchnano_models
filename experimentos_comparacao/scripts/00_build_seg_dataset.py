"""Track 1 (base de dados) -- constrói um dataset de SEGMENTAÇÃO classe-
agnóstica ("particle", 1 classe só) usando o MESMO split travado por
imagem-fonte já usado pelo YOLO26L principal
(`experimentos/resultados/00_split_images/data/image_splits.csv`), pra
reproduzir a Tabela 3 do paper do dataset (Rezvani/Zarrabi et al.) --
Mask R-CNN R101/X101, YOLOv8n-seg, YOLOv26n-seg -- de forma comparável com
o nosso split (sem vazamento entre imagens da mesma micrografia).

Não regera o split, não modifica `experimentos/`. As imagens usadas são as
MESMAS (byte-a-byte, confirmado via md5sum) já presentes em
`experimentos/datasets_yolo26_v2/images/{split}/` -- symlinkadas aqui, só
os labels de segmentação são novos, gerados a partir dos polígonos COCO
originais em `MPDataset/Full_Images/COCO Format/{classe}/{classe}_COCO.json`
(mesmo espaço de pixels, 1280x960, sem redimensionamento).

Saída em experimentos_comparacao/datasets_baseline_seg/:
    images/{train,val,test}/*.png          (symlinks)
    labels/{train,val,test}/*.txt          (YOLO-seg, polígono normalizado)
    coco/{train,val,test}.json             (COCO instance-seg, 1 categoria)
    data.yaml                              (pro treino via Ultralytics -seg)

Uso:
    python3 00_build_seg_dataset.py [--preview N]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comp_common as cc

CATEGORY_NAME = "particle"

# 10 imagens PET renomeadas (nome antigo "PS-*" -> nome atual "PET-B*") numa
# correção de nome de arquivo anterior neste projeto; o COCO oficial do MiNa
# (MPDataset/.../PET_COCO.json) nunca foi atualizado e ainda lista o
# file_name antigo, então o casamento por string exata abaixo falhava e
# essas 10 imagens (com polígono real, já existente) eram descartadas do
# dataset de segmentação -- confirmado 2026-09-10 batendo o image_id de
# experimentos/resultados/00_split_images/data/image_splits.csv contra o id
# do COCO oficial (mesma contagem de anotação nas 10, polígono válido).
RENAMED_FILE_ALIASES = {
    "PET-B-1000X.png": "PS-1000X.png",
    "PET-B-2-5000X.png": "PS-2-5000X.png",
    "PET-B-2000x-2.png": "PS-2000x-2.png",
    "PET-B-2000x-4.png": "PS-2000x-4.png",
    "PET-B-5000X.png": "PS-5000X.png",
    "PET-B-5000x-1.png": "PS-5000x-1.png",
    "PET-B2-2000x-1.png": "PS2-2000x-1.png",
    "PET-B2-2000x-2.png": "PS2-2000x-2.png",
    "PET-B2-2000x-4.png": "PS2-2000x-4.png",
    "PET-B2-5000x-4.png": "PS2-5000x-4.png",
}


def polygon_area(poly: list[float]) -> float:
    """Shoelace, poly = [x1,y1,x2,y2,...]."""
    n = len(poly) // 2
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = poly[2 * i], poly[2 * i + 1]
        x2, y2 = poly[2 * ((i + 1) % n)], poly[2 * ((i + 1) % n) + 1]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def largest_polygon(segmentation) -> list[float] | None:
    """segmentation pode ter múltiplos polígonos (partícula com furo/2
    partes); pega o de maior área -- limitação documentada: perde partes
    menores de anotações multi-polígono (raras neste dataset)."""
    if not isinstance(segmentation, list) or not segmentation:
        return None  # RLE (dict) ou vazio -- não suportado, pula
    polys = [p for p in segmentation if isinstance(p, list) and len(p) >= 6]
    if not polys:
        return None
    return max(polys, key=polygon_area)


def load_class_coco(cls: str) -> dict:
    path = cc.MINA_COCO_ROOT / cls / f"{cls}_COCO.json"
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    by_image_id = {}
    for ann in d["annotations"]:
        if ann.get("iscrowd"):
            continue
        poly = largest_polygon(ann.get("segmentation"))
        if poly is None:
            continue
        by_image_id.setdefault(ann["image_id"], []).append({
            "poly": poly, "bbox": ann["bbox"], "area": ann.get("area", polygon_area(poly)),
        })
    images_by_id = {im["id"]: im for im in d["images"]}
    return {"by_image_id": by_image_id, "images_by_id": images_by_id}


def yolo_seg_line(poly: list[float], W: int, H: int) -> str:
    coords = []
    for i in range(0, len(poly), 2):
        x = min(max(poly[i] / W, 0.0), 1.0)
        y = min(max(poly[i + 1] / H, 0.0), 1.0)
        coords.append(f"{x:.6f} {y:.6f}")
    return "0 " + " ".join(coords)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preview", type=int, default=3, help="quantas imagens de preview (overlay de máscara) salvar por split")
    parser.add_argument("--out-dir", type=str, default=None,
                         help="diretório de saída alternativo (default: datasets_baseline_seg/, "
                              "que fica intocado se este flag for usado)")
    args = parser.parse_args()

    manifest = cc.load_split_manifest()
    print(f"[00_build_seg_dataset] split travado: {len(manifest)} imagens-fonte "
          f"({cc.SPLIT_MANIFEST})")

    coco_by_class = {cls: load_class_coco(cls) for cls in cc.CLASSES}

    out_root = Path(args.out_dir) if args.out_dir else cc.SEG_DATASET_DIR
    for split in ("train", "val", "test"):
        (out_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    (out_root / "coco").mkdir(parents=True, exist_ok=True)

    coco_out = {s: {"images": [], "annotations": [], "categories": [{"id": 1, "name": CATEGORY_NAME}]} for s in ("train", "val", "test")}
    next_ann_id = {s: 1 for s in ("train", "val", "test")}
    next_img_id = {s: 1 for s in ("train", "val", "test")}

    stats = {s: {"n_source_images": 0, "n_files": 0, "n_annotations": 0, "n_annotations_skipped": 0} for s in ("train", "val", "test")}
    preview_saved = {s: 0 for s in ("train", "val", "test")}

    src_images_dir = cc.YOLO_DET_DATASET / "images"
    for split_dir_split in ("train", "val", "test"):
        d = src_images_dir / split_dir_split
        if not d.exists():
            continue
        for img_path in sorted(d.glob("*.png")):
            base = cc.strip_dup_prefix(img_path.name)  # remove dupN_ se houver
            # base = "{classe}_{file_name original}"
            if "_" not in base:
                continue
            cls, orig_name = base.split("_", 1)
            if cls not in cc.CLASSES:
                continue

            coco_info = coco_by_class[cls]
            # acha o image_id original pelo file_name (sem o prefixo de classe/pasta);
            # cai pro nome antigo (RENAMED_FILE_ALIASES) se o nome atual não bater --
            # ver comentário na constante acima
            lookup_name = RENAMED_FILE_ALIASES.get(orig_name, orig_name)
            image_id = None
            for iid, im in coco_info["images_by_id"].items():
                if im["file_name"].split("/", 1)[-1] == lookup_name:
                    image_id = iid
                    break
            if image_id is None:
                continue

            anns = coco_info["by_image_id"].get(image_id, [])
            im_meta = coco_info["images_by_id"][image_id]
            W, H = im_meta["width"], im_meta["height"]

            # symlink da imagem (idêntica à já usada pelo YOLO26L)
            link_path = out_root / "images" / split_dir_split / img_path.name
            if not link_path.exists():
                link_path.symlink_to(img_path.resolve())

            label_lines = [yolo_seg_line(a["poly"], W, H) for a in anns]
            label_path = out_root / "labels" / split_dir_split / (img_path.stem + ".txt")
            label_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""))

            coco_img_id = next_img_id[split_dir_split]; next_img_id[split_dir_split] += 1
            coco_out[split_dir_split]["images"].append({
                "id": coco_img_id, "file_name": img_path.name, "width": W, "height": H,
            })
            for a in anns:
                coco_out[split_dir_split]["annotations"].append({
                    "id": next_ann_id[split_dir_split], "image_id": coco_img_id, "category_id": 1,
                    "segmentation": [a["poly"]], "bbox": a["bbox"], "area": a["area"], "iscrowd": 0,
                })
                next_ann_id[split_dir_split] += 1

            stats[split_dir_split]["n_source_images"] += 1
            stats[split_dir_split]["n_files"] += 1
            stats[split_dir_split]["n_annotations"] += len(anns)

            if preview_saved[split_dir_split] < args.preview and anns:
                _save_preview(img_path, anns, out_root / "preview" / split_dir_split / img_path.name)
                preview_saved[split_dir_split] += 1

    for split in ("train", "val", "test"):
        with open(out_root / "coco" / f"{split}.json", "w", encoding="utf-8") as f:
            json.dump(coco_out[split], f)

    data_yaml = out_root / "data.yaml"
    data_yaml.write_text(
        f"path: {out_root}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: 1\n"
        f"names:\n  - {CATEGORY_NAME}\n"
    )

    print("\n[00_build_seg_dataset] resumo por split:")
    for split, s in stats.items():
        print(f"  {split:6s} arquivos={s['n_files']:4d}  anotações(polígonos)={s['n_annotations']:5d}")

    exp_name = "00_build_seg_dataset" if not args.out_dir else "00_build_seg_dataset_fixed"
    cc.save_summary(exp_name, metrics=stats, extra_metadata={
        "data_yaml": str(data_yaml),
        "split_manifest_source": str(cc.SPLIT_MANIFEST),
        "coco_source_root": str(cc.MINA_COCO_ROOT),
        "note": "reaproveita split travado por imagem-fonte de experimentos/00_split_images.py; "
                "imagens são symlinks para experimentos/datasets_yolo26_v2/images/ (byte-idênticas); "
                "multi-polígono por anotação: mantém só o de maior área.",
    })
    print(f"\n[00_build_seg_dataset] dataset em {out_root}")
    print(f"[00_build_seg_dataset] preview (overlay de máscara) em {out_root / 'preview'} -- "
          f"confira alinhamento antes de treinar.")


def _save_preview(img_path: Path, anns: list[dict], out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from PIL import Image
    except ImportError:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(img_path).convert("RGB")
    fig, ax = plt.subplots(figsize=(im.width / 100, im.height / 100), dpi=100)
    ax.imshow(im)
    for a in anns:
        pts = list(zip(a["poly"][0::2], a["poly"][1::2]))
        ax.add_patch(Polygon(pts, closed=True, fill=True, alpha=0.35, edgecolor="red", linewidth=1))
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    main()
