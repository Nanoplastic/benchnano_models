"""Protocolo "curated crop" -- réplica corrigida do recorte 256x256 em
patches usado pelo paper do dataset (Rezvani/Zarrabi et al.), com as duas
correções que `main_pt.tex` já promete (Seção "Comparison with the MiNa
benchmark study"):
    1. Patches incompletos na borda são DESCARTADOS (não mantidos
       reduzidos) -- janela deslizante 256x256, passo 256, só janelas
       inteiras.
    2. O split treino/val/teste é HERDADO da imagem-fonte (reaproveita
       experimentos/resultados/00_split_images/data/image_splits.csv,
       nunca recalculado por patch) -- ao contrário do split deles, que
       distribui patches aleatoriamente e por isso vaza dados entre
       treino/teste da mesma micrografia.

Patch sem nenhuma anotação após o recorte é descartado (mesma regra do
paper original: "patches consisting entirely of background... were
excluded").

Gera DOIS formatos a partir da mesma grade de patches:
    - bbox 4-classes (YOLO + COCO) -- para YOLOv10 e Faster R-CNN.
    - polígono 1-classe "particle" (YOLO-seg + COCO) -- para Benchnano-seg
      e Mask R-CNN.

Uso:
    python3 08_build_curated_crop_dataset.py [--preview N]
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comp_common as cc

CROP = cc.CROP_SIZE
CLASS_NAMES = cc.CLASSES  # PE, PET, PP, PS


def polygon_area(poly: list[float]) -> float:
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
    if not isinstance(segmentation, list) or not segmentation:
        return None
    polys = [p for p in segmentation if isinstance(p, list) and len(p) >= 6]
    if not polys:
        return None
    return max(polys, key=polygon_area)


def clip_polygon_to_rect(points: list[tuple[float, float]], x0, y0, x1, y1):
    """Sutherland-Hodgman: recorta um polígono (lista de (x,y)) por um
    retângulo axis-aligned [x0,x1]x[y0,y1]. Retorna lista de pontos
    (pode ser vazia se não houver interseção)."""
    def clip_edge(poly, inside_fn, intersect_fn):
        if not poly:
            return []
        out = []
        prev = poly[-1]
        prev_in = inside_fn(prev)
        for cur in poly:
            cur_in = inside_fn(cur)
            if cur_in:
                if not prev_in:
                    out.append(intersect_fn(prev, cur))
                out.append(cur)
            elif prev_in:
                out.append(intersect_fn(prev, cur))
            prev, prev_in = cur, cur_in
        return out

    def isect_x(p1, p2, xc):
        x1, y1 = p1; x2, y2 = p2
        t = (xc - x1) / (x2 - x1) if x2 != x1 else 0.0
        return (xc, y1 + t * (y2 - y1))

    def isect_y(p1, p2, yc):
        x1, y1 = p1; x2, y2 = p2
        t = (yc - y1) / (y2 - y1) if y2 != y1 else 0.0
        return (x1 + t * (x2 - x1), yc)

    poly = points
    poly = clip_edge(poly, lambda p: p[0] >= x0, lambda a, b: isect_x(a, b, x0))
    poly = clip_edge(poly, lambda p: p[0] <= x1, lambda a, b: isect_x(a, b, x1))
    poly = clip_edge(poly, lambda p: p[1] >= y0, lambda a, b: isect_y(a, b, y0))
    poly = clip_edge(poly, lambda p: p[1] <= y1, lambda a, b: isect_y(a, b, y1))
    return poly


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
        by_image_id.setdefault(ann["image_id"], []).append(poly)
    images_by_id = {im["id"]: im for im in d["images"]}
    return {"by_image_id": by_image_id, "images_by_id": images_by_id}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preview", type=int, default=3)
    args = parser.parse_args()

    manifest = cc.load_split_manifest()
    split_by_key = {(r["class"], r["file_name"]): r["split"] for r in manifest}
    coco_by_class = {cls: load_class_coco(cls) for cls in CLASS_NAMES}

    # dois roots YOLO separados (Ultralytics espera images/<split> e
    # labels/<split> irmãos) -- seg/images é symlink pras mesmas imagens de
    # det/images, já que o recorte é idêntico, só o rótulo muda.
    out = cc.CURATED_CROP_DIR
    for split in ("train", "val", "test"):
        (out / "det" / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "det" / "labels" / split).mkdir(parents=True, exist_ok=True)
        (out / "seg" / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "seg" / "labels" / split).mkdir(parents=True, exist_ok=True)
    (out / "coco_det").mkdir(parents=True, exist_ok=True)
    (out / "coco_seg").mkdir(parents=True, exist_ok=True)
    (out / "preview").mkdir(parents=True, exist_ok=True)

    coco_det = {s: {"images": [], "annotations": [], "categories": [{"id": i + 1, "name": c} for i, c in enumerate(CLASS_NAMES)]} for s in ("train", "val", "test")}
    coco_seg = {s: {"images": [], "annotations": [], "categories": [{"id": 1, "name": "particle"}]} for s in ("train", "val", "test")}
    next_img_id = {s: 1 for s in ("train", "val", "test")}
    next_ann_det_id = {s: 1 for s in ("train", "val", "test")}
    next_ann_seg_id = {s: 1 for s in ("train", "val", "test")}
    stats = {s: {"n_patches": 0, "n_particles": 0} for s in ("train", "val", "test")}
    preview_count = 0

    for cls in CLASS_NAMES:
        coco_info = coco_by_class[cls]
        cls_id0 = CLASS_NAMES.index(cls)  # 0-based, igual ao datasets_yolo26_v2
        for image_id, im_meta in coco_info["images_by_id"].items():
            orig_name = im_meta["file_name"].split("/", 1)[-1]
            split = split_by_key.get((cls, orig_name))
            if split is None:
                continue  # imagem fora do split travado (não deveria acontecer, mas não assume)

            img_path = cc.MINA_COCO_ROOT / cls / orig_name
            if not img_path.exists():
                continue
            image = Image.open(img_path).convert("RGB")
            W, H = im_meta["width"], im_meta["height"]
            polys = coco_info["by_image_id"].get(image_id, [])

            nx, ny = W // CROP, H // CROP  # só janelas inteiras -- descarta borda incompleta
            for gy in range(ny):
                for gx in range(nx):
                    x0, y0 = gx * CROP, gy * CROP
                    x1, y1 = x0 + CROP, y0 + CROP

                    patch_anns = []  # cada item: polígono clipado, em coords LOCAIS do patch
                    for poly in polys:
                        pts = list(zip(poly[0::2], poly[1::2]))
                        clipped = clip_polygon_to_rect(pts, x0, y0, x1, y1)
                        if len(clipped) < 3:
                            continue
                        local = [(px - x0, py - y0) for px, py in clipped]
                        area = polygon_area([c for p in local for c in p])
                        if area < 4.0:  # descarta fragmentos degenerados (<4px^2)
                            continue
                        patch_anns.append(local)

                    if not patch_anns:
                        continue  # patch sem partícula -- descartado (mesma regra do paper original)

                    patch_img = image.crop((x0, y0, x1, y1))
                    patch_name = f"{cls}_{Path(orig_name).stem}_r{gy}c{gx}.png"
                    det_img_path = out / "det" / "images" / split / patch_name
                    patch_img.save(det_img_path)
                    seg_img_path = out / "seg" / "images" / split / patch_name
                    if not seg_img_path.exists():
                        seg_img_path.symlink_to(det_img_path.resolve())

                    # -- bbox 4-classes --
                    det_lines = []
                    for local in patch_anns:
                        xs = [p[0] for p in local]; ys = [p[1] for p in local]
                        bx0, by0, bx1, by1 = min(xs), min(ys), max(xs), max(ys)
                        cx, cy = (bx0 + bx1) / 2 / CROP, (by0 + by1) / 2 / CROP
                        bw, bh = (bx1 - bx0) / CROP, (by1 - by0) / CROP
                        det_lines.append(f"{cls_id0} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    (out / "det" / "labels" / split / (Path(patch_name).stem + ".txt")).write_text("\n".join(det_lines) + "\n")

                    # -- polígono 1-classe --
                    seg_lines = []
                    for local in patch_anns:
                        coords = " ".join(f"{px / CROP:.6f} {py / CROP:.6f}" for px, py in local)
                        seg_lines.append(f"0 {coords}")
                    (out / "seg" / "labels" / split / (Path(patch_name).stem + ".txt")).write_text("\n".join(seg_lines) + "\n")

                    # -- coco (ambos formatos) --
                    img_id = next_img_id[split]; next_img_id[split] += 1
                    coco_det[split]["images"].append({"id": img_id, "file_name": patch_name, "width": CROP, "height": CROP})
                    coco_seg[split]["images"].append({"id": img_id, "file_name": patch_name, "width": CROP, "height": CROP})
                    for local in patch_anns:
                        xs = [p[0] for p in local]; ys = [p[1] for p in local]
                        bx0, by0, bx1, by1 = min(xs), min(ys), max(xs), max(ys)
                        flat = [c for p in local for c in p]
                        area = polygon_area(flat)
                        coco_det[split]["annotations"].append({
                            "id": next_ann_det_id[split], "image_id": img_id, "category_id": cls_id0 + 1,
                            "bbox": [bx0, by0, bx1 - bx0, by1 - by0], "area": area, "iscrowd": 0,
                        })
                        next_ann_det_id[split] += 1
                        coco_seg[split]["annotations"].append({
                            "id": next_ann_seg_id[split], "image_id": img_id, "category_id": 1,
                            "segmentation": [flat], "bbox": [bx0, by0, bx1 - bx0, by1 - by0], "area": area, "iscrowd": 0,
                        })
                        next_ann_seg_id[split] += 1

                    stats[split]["n_patches"] += 1
                    stats[split]["n_particles"] += len(patch_anns)

                    if preview_count < args.preview and split == "test":
                        _save_preview(patch_img, patch_anns, out / "preview" / patch_name)
                        preview_count += 1

    for split in ("train", "val", "test"):
        with open(out / "coco_det" / f"{split}.json", "w") as f:
            json.dump(coco_det[split], f)
        with open(out / "coco_seg" / f"{split}.json", "w") as f:
            json.dump(coco_seg[split], f)

    (out / "det" / "data.yaml").write_text(
        f"path: {out / 'det'}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(CLASS_NAMES)}\nnames:\n" + "\n".join(f"  - {c}" for c in CLASS_NAMES) + "\n"
    )
    (out / "seg" / "data.yaml").write_text(
        f"path: {out / 'seg'}\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 1\nnames:\n  - particle\n"
    )
    print("\n[08_build_curated_crop_dataset] resumo por split:")
    for split, s in stats.items():
        print(f"  {split:6s} patches={s['n_patches']:4d}  partículas={s['n_particles']:6d}")

    cc.save_summary("08_build_curated_crop_dataset", metrics=stats, extra_metadata={
        "crop_size": CROP, "split_manifest_source": str(cc.SPLIT_MANIFEST),
        "note": "janela deslizante sem sobreposicao, so janelas inteiras (borda incompleta descartada); "
                "patch sem anotacao apos recorte e descartado; split herdado da imagem-fonte.",
    })
    print(f"\n[08_build_curated_crop_dataset] dataset em {out}")
    print(f"[08_build_curated_crop_dataset] preview em {out / 'preview'} -- conferir antes de treinar.")


def _save_preview(patch_img, patch_anns, out_path: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(CROP / 100, CROP / 100), dpi=100)
    ax.imshow(patch_img)
    for local in patch_anns:
        ax.add_patch(Polygon(local, closed=True, fill=True, alpha=0.35, edgecolor="red", linewidth=1))
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    main()
