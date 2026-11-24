"""Dataset de segmentação de imagem inteira / Full-image segmentation dataset.

PT: Constrói um dataset de segmentação sem classe ("particle", classe única) com a mesma
    partição fixa por imagem de origem do YOLO26L principal
    (`experimentos/resultados/00_split_images/data/image_splits.csv`), para treinar Mask
    R-CNN e YOLO-seg sem vazamento entre recortes da mesma micrografia.
    Não regera a partição nem modifica `experimentos/`. As imagens são as mesmas (idênticas
    byte a byte) de `experimentos/datasets_yolo26_v2/images/{split}/`, ligadas por symlink;
    só os rótulos de segmentação são novos, gerados a partir dos polígonos COCO de
    `MPDataset/Full_Images/COCO Format/{classe}/{classe}_COCO.json` (mesmo espaço de pixels,
    1280x960, sem redimensionamento).
    Saída em experimentos_comparacao/datasets_baseline_seg/:
    images/{train,val,test}/*.png (symlinks)
    labels/{train,val,test}/*.txt (YOLO-seg, polígono normalizado)
    coco/{train,val,test}.json (COCO instance-seg, 1 categoria)
    data.yaml (treino via Ultralytics -seg)
    Uso:
    python3 00_build_seg_dataset.py [--preview N]
EN: Builds a class-agnostic segmentation dataset ("particle", single class) with the same
    fixed source-image split as the main YOLO26L
    (`experimentos/resultados/00_split_images/data/image_splits.csv`), to train Mask R-CNN
    and YOLO-seg without leakage between crops of the same micrograph.
    Does not regenerate the split or modify `experimentos/`. Images are the same
    (byte-identical) as in `experimentos/datasets_yolo26_v2/images/{split}/`, linked via
    symlinks; only the segmentation labels are new, generated from the COCO polygons in
    `MPDataset/Full_Images/COCO Format/{classe}/{classe}_COCO.json` (same pixel space,
    1280x960, no resizing).
    Output in experimentos_comparacao/datasets_baseline_seg/:
    images/{train,val,test}/*.png (symlinks)
    labels/{train,val,test}/*.txt (YOLO-seg, normalized polygon)
    coco/{train,val,test}.json (COCO instance-seg, 1 category)
    data.yaml (training via Ultralytics -seg)
    Usage:
    python3 00_build_seg_dataset.py [--preview N]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comp_common as cc

NOME_CATEGORIA = "particle"

# PT: 10 imagens PET têm nome atual "PET-B*", mas o COCO do MiNa
#     (MPDataset/.../PET_COCO.json) ainda lista o file_name antigo "PS-*". Sem este
#     mapeamento, o pareamento por nome exato descartaria essas 10 imagens (que têm polígono
#     válido). O mapeamento foi conferido pelo image_id de
#     experimentos/resultados/00_split_images/data/image_splits.csv contra o id do COCO
#     (mesma contagem de anotações nas 10).
# EN: 10 PET images are currently named "PET-B*", but the MiNa COCO file
#     (MPDataset/.../PET_COCO.json) still lists the old "PS-*" file_name. Without this
#     mapping, exact name matching would drop these 10 images (which have valid polygons).
#     The mapping was checked via the image_id in
#     experimentos/resultados/00_split_images/data/image_splits.csv against the COCO id
#     (same annotation count for all 10).
APELIDOS_ARQUIVOS_RENOMEADOS = {
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


def calcular_area_poligono(poly: list[float]) -> float:
    """PT: Fórmula do laço (shoelace), poligono = [x1,y1,x2,y2,...].
    EN: Shoelace formula, poligono = [x1,y1,x2,y2,...].
    """
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
    """PT: segmentacao pode ter vários polígonos (partícula com furo ou em 2 partes); usa o de
        maior área. Partes menores de anotações com vários polígonos (raras neste dataset)
        são perdidas.
    EN: segmentacao may hold several polygons (particle with a hole or in 2 parts); the
        largest one is used. Smaller parts of multi-polygon annotations (rare in this
        dataset) are lost.
    """
    if not isinstance(segmentacao, list) or not segmentacao:
        return None  # PT: RLE (dict) ou vazio: não suportado, ignorado / EN: RLE (dict) or empty: unsupported, skipped
    poligonos = [p for p in segmentacao if isinstance(p, list) and len(p) >= 6]
    if not poligonos:
        return None
    return max(poligonos, key=calcular_area_poligono)


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
        por_id_imagem.setdefault(anot["image_id"], []).append({
            "poly": poly, "bbox": anot["bbox"], "area": anot.get("area", calcular_area_poligono(poly)),
        })
    imagens_por_id = {im["id"]: im for im in d["images"]}
    return {"by_image_id": por_id_imagem, "images_by_id": imagens_por_id}


def linha_seg_yolo(poly: list[float], W: int, H: int) -> str:
    coordenadas = []
    for i in range(0, len(poly), 2):
        x = min(max(poly[i] / W, 0.0), 1.0)
        y = min(max(poly[i + 1] / H, 0.0), 1.0)
        coordenadas.append(f"{x:.6f} {y:.6f}")
    return "0 " + " ".join(coordenadas)


def principal():
    analisador = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    analisador.add_argument("--preview", type=int, default=3, help="quantas imagens de preview (overlay de máscara) salvar por split")
    analisador.add_argument("--out-dir", type=str, default=None,
                         help="diretório de saída alternativo (default: datasets_baseline_seg/, "
                              "que fica intocado se este flag for usado)")
    args = analisador.parse_args()

    manifesto = cc.carregar_manifesto_particao()
    print(f"[00_build_seg_dataset] split travado: {len(manifesto)} imagens-fonte "
          f"({cc.MANIFESTO_PARTICAO})")

    coco_por_classe = {cls: carregar_coco_classe(cls) for cls in cc.CLASSES_POLIMERO}

    raiz_saida = Path(args.out_dir) if args.out_dir else cc.DIR_DATASET_SEG
    for particao in ("train", "val", "test"):
        (raiz_saida / "images" / particao).mkdir(parents=True, exist_ok=True)
        (raiz_saida / "labels" / particao).mkdir(parents=True, exist_ok=True)
    (raiz_saida / "coco").mkdir(parents=True, exist_ok=True)

    saida_coco = {s: {"images": [], "annotations": [], "categories": [{"id": 1, "name": NOME_CATEGORIA}]} for s in ("train", "val", "test")}
    prox_id_anot = {s: 1 for s in ("train", "val", "test")}
    prox_id_img = {s: 1 for s in ("train", "val", "test")}

    estatisticas = {s: {"n_source_images": 0, "n_files": 0, "n_annotations": 0, "n_annotations_skipped": 0} for s in ("train", "val", "test")}
    previa_salva = {s: 0 for s in ("train", "val", "test")}

    dir_imagens_origem = cc.DATASET_DET_YOLO / "images"
    for particao_dir_particao in ("train", "val", "test"):
        d = dir_imagens_origem / particao_dir_particao
        if not d.exists():
            continue
        for caminho_img in sorted(d.glob("*.png")):
            base = cc.remover_prefixo_duplicado(caminho_img.name)  # PT: remove dupN_ se houver / EN: strips dupN_ if present
            # PT: base = "{classe}_{file_name original}"
            # EN: base = "{class}_{original file_name}"
            if "_" not in base:
                continue
            cls, nome_orig = base.split("_", 1)
            if cls not in cc.CLASSES_POLIMERO:
                continue

            info_coco = coco_por_classe[cls]
            # PT: encontra o image_id original pelo file_name (sem o prefixo de
            #     classe/pasta); usa o nome antigo (APELIDOS_ARQUIVOS_RENOMEADOS) se o nome
            #     atual não for encontrado
            # EN: finds the original image_id by file_name (without the class/folder
            #     prefix); falls back to the old name (APELIDOS_ARQUIVOS_RENOMEADOS) if the
            #     current name is not found
            nome_busca = APELIDOS_ARQUIVOS_RENOMEADOS.get(nome_orig, nome_orig)
            id_imagem = None
            for iid, im in info_coco["images_by_id"].items():
                if im["file_name"].split("/", 1)[-1] == nome_busca:
                    id_imagem = iid
                    break
            if id_imagem is None:
                continue

            anots = info_coco["by_image_id"].get(id_imagem, [])
            meta_img = info_coco["images_by_id"][id_imagem]
            W, H = meta_img["width"], meta_img["height"]

            # PT: symlink da imagem (idêntica à usada pelo YOLO26L)
            # EN: image symlink (identical to the one used by YOLO26L)
            caminho_link = raiz_saida / "images" / particao_dir_particao / caminho_img.name
            if not caminho_link.exists():
                caminho_link.symlink_to(caminho_img.resolve())

            linhas_rotulo = [linha_seg_yolo(a["poly"], W, H) for a in anots]
            caminho_rotulo = raiz_saida / "labels" / particao_dir_particao / (caminho_img.stem + ".txt")
            caminho_rotulo.write_text("\n".join(linhas_rotulo) + ("\n" if linhas_rotulo else ""))

            id_img_coco = prox_id_img[particao_dir_particao]; prox_id_img[particao_dir_particao] += 1
            saida_coco[particao_dir_particao]["images"].append({
                "id": id_img_coco, "file_name": caminho_img.name, "width": W, "height": H,
            })
            for a in anots:
                saida_coco[particao_dir_particao]["annotations"].append({
                    "id": prox_id_anot[particao_dir_particao], "image_id": id_img_coco, "category_id": 1,
                    "segmentation": [a["poly"]], "bbox": a["bbox"], "area": a["area"], "iscrowd": 0,
                })
                prox_id_anot[particao_dir_particao] += 1

            estatisticas[particao_dir_particao]["n_source_images"] += 1
            estatisticas[particao_dir_particao]["n_files"] += 1
            estatisticas[particao_dir_particao]["n_annotations"] += len(anots)

            if previa_salva[particao_dir_particao] < args.preview and anots:
                _salvar_previa(caminho_img, anots, raiz_saida / "preview" / particao_dir_particao / caminho_img.name)
                previa_salva[particao_dir_particao] += 1

    for particao in ("train", "val", "test"):
        with open(raiz_saida / "coco" / f"{particao}.json", "w", encoding="utf-8") as f:
            json.dump(saida_coco[particao], f)

    yaml_dados = raiz_saida / "data.yaml"
    yaml_dados.write_text(
        f"path: {raiz_saida}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"nc: 1\n"
        f"names:\n  - {NOME_CATEGORIA}\n"
    )

    print("\n[00_build_seg_dataset] resumo por split:")
    for particao, s in estatisticas.items():
        print(f"  {particao:6s} arquivos={s['n_files']:4d}  anotações(polígonos)={s['n_annotations']:5d}")

    nome_exp = "00_build_seg_dataset" if not args.out_dir else "00_build_seg_dataset_fixed"
    cc.salvar_resumo(nome_exp, metricas=estatisticas, metadados_extras={
        "data_yaml": str(yaml_dados),
        "split_manifest_source": str(cc.MANIFESTO_PARTICAO),
        "coco_source_root": str(cc.RAIZ_MINA_COCO),
        "note": "reaproveita split travado por imagem-fonte de experimentos/00_split_images.py; "
                "imagens são symlinks para experimentos/datasets_yolo26_v2/images/ (byte-idênticas); "
                "multi-polígono por anotação: mantém só o de maior área.",
    })
    print(f"\n[00_build_seg_dataset] dataset em {raiz_saida}")
    print(f"[00_build_seg_dataset] preview (overlay de máscara) em {raiz_saida / 'preview'} -- "
          f"confira alinhamento antes de treinar.")


def _salvar_previa(caminho_img: Path, anots: list[dict], caminho_saida: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        from PIL import Image
    except ImportError:
        return
    caminho_saida.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(caminho_img).convert("RGB")
    fig, ax = plt.subplots(figsize=(im.width / 100, im.height / 100), dpi=100)
    ax.imshow(im)
    for a in anots:
        pts = list(zip(a["poly"][0::2], a["poly"][1::2]))
        ax.add_patch(Polygon(pts, closed=True, fill=True, alpha=0.35, edgecolor="red", linewidth=1))
    ax.axis("off")
    fig.tight_layout(pad=0)
    fig.savefig(caminho_saida, bbox_inches="tight", pad_inches=0)
    plt.close(fig)


if __name__ == "__main__":
    principal()
