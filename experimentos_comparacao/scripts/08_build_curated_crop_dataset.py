"""Dataset do protocolo de recorte curado / Curated crop protocol dataset.

PT: Protocolo "recorte curado": réplica corrigida do recorte em blocos de 256x256 usado
    pelos métodos de referência do dataset MiNa, com duas correções:
    1. Blocos incompletos na borda são DESCARTADOS (não mantidos reduzidos): janela
    deslizante 256x256, passo 256, só janelas inteiras.
    2. A partição treino/val/teste é HERDADA da imagem de origem
    (experimentos/resultados/00_split_images/data/image_splits.csv, nunca recalculada por
    bloco). Distribuir blocos aleatoriamente vazaria dados entre treino e teste da mesma
    micrografia.
    Blocos sem nenhuma anotação após o recorte são descartados (blocos só de fundo ficam
    fora, como no protocolo de referência).
    Gera dois formatos a partir da mesma grade de blocos:
    - bbox de 4 classes (YOLO + COCO), para YOLOv10 e Faster R-CNN;
    - polígono de classe única "particle" (YOLO-seg + COCO), para Benchnano-seg e Mask
    R-CNN.
    Uso:
    python3 08_build_curated_crop_dataset.py [--preview N]
EN: "Curated crop" protocol: a corrected replica of the 256x256 tiling used by the MiNa
    dataset's reference methods, with two corrections:
    1. Incomplete border tiles are DISCARDED (not kept shrunk): 256x256 sliding window,
    stride 256, full windows only.
    2. The train/val/test split is INHERITED from the source image
    (experimentos/resultados/00_split_images/data/image_splits.csv, never recomputed per
    tile). Distributing tiles at random would leak data between train and test of the same
    micrograph.
    Tiles with no annotation left after cropping are discarded (background-only tiles are
    excluded, as in the reference protocol).
    Generates two formats from the same tile grid:
    - 4-class bbox (YOLO + COCO), for YOLOv10 and Faster R-CNN;
    - single-class "particle" polygon (YOLO-seg + COCO), for Benchnano-seg and Mask R-CNN.
    Usage:
    python3 08_build_curated_crop_dataset.py [--preview N]
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comp_common as cc

RECORTE_FIGURA = cc.TAMANHO_RECORTE
NOMES_CLASSES = cc.CLASSES_POLIMERO  # PT: PE, PET, PP, PS / EN: PE, PET, PP, PS


def calcular_area_poligono(poly: list[float]) -> float:
    n = len(poly) // 2
    if n < 3:
        return 0.0
    s = 0.0
    for i in range(n):
        x1, y1 = poly[2 * i], poly[2 * i + 1]
        x2, y2 = poly[2 * ((i + 1) % n)], poly[2 * ((i + 1) % n) + 1]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def maior_poligono(segmentacao) -> list[float] | None:
    if not isinstance(segmentacao, list) or not segmentacao:
        return None
    poligonos = [p for p in segmentacao if isinstance(p, list) and len(p) >= 6]
    if not poligonos:
        return None
    return max(poligonos, key=calcular_area_poligono)


def recortar_poligono_no_retangulo(pontos: list[tuple[float, float]], x0, y0, x1, y1):
    """PT: Sutherland-Hodgman: recorta um polígono (lista de (x, y)) por um retângulo alinhado
        aos eixos [x0,x1]x[y0,y1]. Retorna a lista de pontos (vazia se não houver
        interseção).
    EN: Sutherland-Hodgman: clips a polygon (list of (x, y)) to an axis-aligned rectangle
        [x0,x1]x[y0,y1]. Returns the list of points (empty if there is no intersection).
    """
    def recortar_borda(poly, fn_dentro, fn_intersecao):
        if not poly:
            return []
        saida = []
        anterior = poly[-1]
        anterior_dentro = fn_dentro(anterior)
        for atual in poly:
            atual_dentro = fn_dentro(atual)
            if atual_dentro:
                if not anterior_dentro:
                    saida.append(fn_intersecao(anterior, atual))
                saida.append(atual)
            elif anterior_dentro:
                saida.append(fn_intersecao(anterior, atual))
            anterior, anterior_dentro = atual, atual_dentro
        return saida

    def intersec_x(p1, p2, xc):
        x1, y1 = p1; x2, y2 = p2
        t = (xc - x1) / (x2 - x1) if x2 != x1 else 0.0
        return (xc, y1 + t * (y2 - y1))

    def intersec_y(p1, p2, yc):
        x1, y1 = p1; x2, y2 = p2
        t = (yc - y1) / (y2 - y1) if y2 != y1 else 0.0
        return (x1 + t * (x2 - x1), yc)

    poly = pontos
    poly = recortar_borda(poly, lambda p: p[0] >= x0, lambda a, b: intersec_x(a, b, x0))
    poly = recortar_borda(poly, lambda p: p[0] <= x1, lambda a, b: intersec_x(a, b, x1))
    poly = recortar_borda(poly, lambda p: p[1] >= y0, lambda a, b: intersec_y(a, b, y0))
    poly = recortar_borda(poly, lambda p: p[1] <= y1, lambda a, b: intersec_y(a, b, y1))
    return poly


def carregar_coco_classe(cls: str) -> dict:
    caminho = cc.RAIZ_MINA_COCO / cls / f"{cls}_COCO.json"
    with open(caminho, encoding="utf-8") as f:
        d = json.load(f)
    por_id_imagem = {}
    for anot in d["annotations"]:
        if anot.get("iscrowd"):
            continue
        poly = maior_poligono(anot.get("segmentation"))
        if poly is None:
            continue
        por_id_imagem.setdefault(anot["image_id"], []).append(poly)
    imagens_por_id = {im["id"]: im for im in d["images"]}
    return {"by_image_id": por_id_imagem, "images_by_id": imagens_por_id}


def principal():
    analisador = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    analisador.add_argument("--preview", type=int, default=3)
    args = analisador.parse_args()

    manifesto = cc.carregar_manifesto_particao()
    particao_por_chave = {(r["class"], r["file_name"]): r["split"] for r in manifesto}
    coco_por_classe = {cls: carregar_coco_classe(cls) for cls in NOMES_CLASSES}

    # PT: duas raízes YOLO separadas (o Ultralytics espera images/<split> e labels/<split>
    #     irmãos); seg/images é symlink para as mesmas imagens de det/images, já que o
    #     recorte é idêntico e só o rótulo muda
    # EN: two separate YOLO roots (Ultralytics expects sibling images/<split> and
    #     labels/<split>); seg/images is a symlink to the same images as det/images, since
    #     the crop is identical and only the label changes
    saida = cc.DIR_RECORTE_CURADO
    for particao in ("train", "val", "test"):
        (saida / "det" / "images" / particao).mkdir(parents=True, exist_ok=True)
        (saida / "det" / "labels" / particao).mkdir(parents=True, exist_ok=True)
        (saida / "seg" / "images" / particao).mkdir(parents=True, exist_ok=True)
        (saida / "seg" / "labels" / particao).mkdir(parents=True, exist_ok=True)
    (saida / "coco_det").mkdir(parents=True, exist_ok=True)
    (saida / "coco_seg").mkdir(parents=True, exist_ok=True)
    (saida / "preview").mkdir(parents=True, exist_ok=True)

    coco_deteccao = {s: {"images": [], "annotations": [], "categories": [{"id": i + 1, "name": c} for i, c in enumerate(NOMES_CLASSES)]} for s in ("train", "val", "test")}
    coco_segmentacao = {s: {"images": [], "annotations": [], "categories": [{"id": 1, "name": "particle"}]} for s in ("train", "val", "test")}
    prox_id_img = {s: 1 for s in ("train", "val", "test")}
    prox_id_anot_det = {s: 1 for s in ("train", "val", "test")}
    prox_id_anot_seg = {s: 1 for s in ("train", "val", "test")}
    estatisticas = {s: {"n_patches": 0, "n_particles": 0} for s in ("train", "val", "test")}
    contagem_previas = 0

    for cls in NOMES_CLASSES:
        info_coco = coco_por_classe[cls]
        id_cls0 = NOMES_CLASSES.index(cls)  # PT: base 0, igual a datasets_yolo26_v2 / EN: 0-based, same as datasets_yolo26_v2
        for id_imagem, meta_img in info_coco["images_by_id"].items():
            nome_orig = meta_img["file_name"].split("/", 1)[-1]
            particao = particao_por_chave.get((cls, nome_orig))
            if particao is None:
                continue  # PT: imagem fora da partição fixa (não esperado, mas verificado) / EN: image outside the fixed split (not expected, but checked)

            caminho_img = cc.RAIZ_MINA_COCO / cls / nome_orig
            if not caminho_img.exists():
                continue
            imagem = Image.open(caminho_img).convert("RGB")
            W, H = meta_img["width"], meta_img["height"]
            poligonos = info_coco["by_image_id"].get(id_imagem, [])

            nx, ny = W // RECORTE_FIGURA, H // RECORTE_FIGURA  # PT: só janelas inteiras; descarta a borda incompleta / EN: full windows only; the incomplete border is discarded
            for gy in range(ny):
                for gx in range(nx):
                    x0, y0 = gx * RECORTE_FIGURA, gy * RECORTE_FIGURA
                    x1, y1 = x0 + RECORTE_FIGURA, y0 + RECORTE_FIGURA

                    anots_bloco = []  # PT: cada item: polígono recortado, em coordenadas LOCAIS do bloco / EN: each item: clipped polygon, in tile-LOCAL coordinates
                    for poly in poligonos:
                        pts = list(zip(poly[0::2], poly[1::2]))
                        recortado = recortar_poligono_no_retangulo(pts, x0, y0, x1, y1)
                        if len(recortado) < 3:
                            continue
                        local = [(px - x0, py - y0) for px, py in recortado]
                        area = calcular_area_poligono([c for p in local for c in p])
                        if area < 4.0:  # PT: descarta fragmentos degenerados (< 4 px^2) / EN: drops degenerate fragments (< 4 px^2)
                            continue
                        anots_bloco.append(local)

                    if not anots_bloco:
                        continue  # PT: bloco sem partícula: descartado / EN: tile with no particle: discarded

                    img_bloco = imagem.crop((x0, y0, x1, y1))
                    nome_bloco = f"{cls}_{Path(nome_orig).stem}_r{gy}c{gx}.png"
                    caminho_img_det = saida / "det" / "images" / particao / nome_bloco
                    img_bloco.save(caminho_img_det)
                    caminho_img_seg = saida / "seg" / "images" / particao / nome_bloco
                    if not caminho_img_seg.exists():
                        caminho_img_seg.symlink_to(caminho_img_det.resolve())

                    # PT: -- bbox de 4 classes --
                    # EN: -- 4-class bbox --
                    linhas_det = []
                    for local in anots_bloco:
                        xs = [p[0] for p in local]; ys = [p[1] for p in local]
                        bx0, by0, bx1, by1 = min(xs), min(ys), max(xs), max(ys)
                        cx, cy = (bx0 + bx1) / 2 / RECORTE_FIGURA, (by0 + by1) / 2 / RECORTE_FIGURA
                        bw, bh = (bx1 - bx0) / RECORTE_FIGURA, (by1 - by0) / RECORTE_FIGURA
                        linhas_det.append(f"{id_cls0} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                    (saida / "det" / "labels" / particao / (Path(nome_bloco).stem + ".txt")).write_text("\n".join(linhas_det) + "\n")

                    # PT: -- polígono de classe única --
                    # EN: -- single-class polygon --
                    linhas_seg = []
                    for local in anots_bloco:
                        coordenadas = " ".join(f"{px / RECORTE_FIGURA:.6f} {py / RECORTE_FIGURA:.6f}" for px, py in local)
                        linhas_seg.append(f"0 {coordenadas}")
                    (saida / "seg" / "labels" / particao / (Path(nome_bloco).stem + ".txt")).write_text("\n".join(linhas_seg) + "\n")

                    # PT: -- COCO (ambos os formatos) --
                    # EN: -- COCO (both formats) --
                    id_img = prox_id_img[particao]; prox_id_img[particao] += 1
                    coco_deteccao[particao]["images"].append({"id": id_img, "file_name": nome_bloco, "width": RECORTE_FIGURA, "height": RECORTE_FIGURA})
                    coco_segmentacao[particao]["images"].append({"id": id_img, "file_name": nome_bloco, "width": RECORTE_FIGURA, "height": RECORTE_FIGURA})
                    for local in anots_bloco:
                        xs = [p[0] for p in local]; ys = [p[1] for p in local]
                        bx0, by0, bx1, by1 = min(xs), min(ys), max(xs), max(ys)
                        plano = [c for p in local for c in p]
                        area = calcular_area_poligono(plano)
                        coco_deteccao[particao]["annotations"].append({
                            "id": prox_id_anot_det[particao], "image_id": id_img, "category_id": id_cls0 + 1,
                            "bbox": [bx0, by0, bx1 - bx0, by1 - by0], "area": area, "iscrowd": 0,
                        })
                        prox_id_anot_det[particao] += 1
                        coco_segmentacao[particao]["annotations"].append({
                            "id": prox_id_anot_seg[particao], "image_id": id_img, "category_id": 1,
                            "segmentation": [plano], "bbox": [bx0, by0, bx1 - bx0, by1 - by0], "area": area, "iscrowd": 0,
                        })
                        prox_id_anot_seg[particao] += 1

                    estatisticas[particao]["n_patches"] += 1
                    estatisticas[particao]["n_particles"] += len(anots_bloco)

                    if contagem_previas < args.preview and particao == "test":
                        _salvar_previa(img_bloco, anots_bloco, saida / "preview" / nome_bloco)
                        contagem_previas += 1

    for particao in ("train", "val", "test"):
        with open(saida / "coco_det" / f"{particao}.json", "w") as f:
            json.dump(coco_deteccao[particao], f)
        with open(saida / "coco_seg" / f"{particao}.json", "w") as f:
            json.dump(coco_segmentacao[particao], f)

    (saida / "det" / "data.yaml").write_text(
        f"path: {saida / 'det'}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"nc: {len(NOMES_CLASSES)}\nnames:\n" + "\n".join(f"  - {c}" for c in NOMES_CLASSES) + "\n"
    )
    (saida / "seg" / "data.yaml").write_text(
        f"path: {saida / 'seg'}\ntrain: images/train\nval: images/val\ntest: images/test\nnc: 1\nnames:\n  - particle\n"
    )
    print("\n[08_build_curated_crop_dataset] resumo por split:")
    for particao, s in estatisticas.items():
        print(f"  {particao:6s} patches={s['n_patches']:4d}  partículas={s['n_particles']:6d}")

    cc.salvar_resumo("08_build_curated_crop_dataset", metricas=estatisticas, metadados_extras={
        "crop_size": RECORTE_FIGURA, "split_manifest_source": str(cc.MANIFESTO_PARTICAO),
        "note": "janela deslizante sem sobreposicao, so janelas inteiras (borda incompleta descartada); "
                "patch sem anotacao apos recorte e descartado; split herdado da imagem-fonte.",
    })
    print(f"\n[08_build_curated_crop_dataset] dataset em {saida}")
    print(f"[08_build_curated_crop_dataset] preview em {saida / 'preview'} -- conferir antes de treinar.")


def _salvar_previa(img_bloco, anots_bloco, caminho_saida: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(RECORTE_FIGURA / 100, RECORTE_FIGURA / 100), dpi=100)
    ax.imshow(img_bloco)
    for local in anots_bloco:
        ax.add_patch(Polygon(local, closed=True, fill=True, alpha=0.35, edgecolor="red", linewidth=1))
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(caminho_saida, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    principal()
