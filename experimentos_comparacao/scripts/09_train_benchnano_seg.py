"""Benchnano-seg -- YOLO26L treinado para segmentação (main_pt.tex linha
1019: "A second pipeline, named Benchnano-seg, was built for the present
study, with an instance segmentation stage before size classification").
Mesma arquitetura/família do detector principal do BenchNano, só que com a
cabeça de segmentação da Ultralytics (`yolo26l-seg.pt`).

Dois protocolos:
    --protocol full     -- experimentos_comparacao/datasets_baseline_seg/
                            (já existe, construído por 00_build_seg_dataset.py)
    --protocol patches   -- experimentos_comparacao/datasets_curated_crop/seg/
                            (construído por 08_build_curated_crop_dataset.py)

Execução única (seed=0) -- ver README sobre o prazo de submissão.

Uso:
    python3 09_train_benchnano_seg.py --protocol full [--epochs 150] [--smoke-test]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comp_common as cc

from ultralytics import YOLO

WEIGHTS = "yolo26l-seg.pt"
SEED = 0


def data_yaml_for(protocol: str, fixed_pool: bool = False) -> Path:
    if protocol == "full":
        if fixed_pool:
            return cc.COMP_ROOT / "datasets_baseline_seg_fixed" / "data.yaml"
        return cc.SEG_DATASET_DIR / "data.yaml"
    return cc.CURATED_CROP_DIR / "seg" / "data.yaml"


def eval_test_set(model_path, data_yaml: Path, root_dir: Path) -> dict:
    model = YOLO(str(model_path))
    data_yaml_test = root_dir / "data_test.yaml"
    if not data_yaml_test.exists():
        data_yaml_test.write_text(
            f"path: {root_dir}\ntrain: images/test\nval: images/test\nnc: 1\nnames:\n  - particle\n"
        )
    metrics = model.val(data=str(data_yaml_test), split="val", imgsz=1024 if root_dir.name != "seg" else 256,
                         conf=0.15, iou=0.7, plots=False, verbose=False)
    return {
        "box": {"precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
                "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map)},
        "mask": {"ap50": float(metrics.seg.map50), "ap75": float(metrics.seg.map75), "ap": float(metrics.seg.map)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", required=True, choices=["full", "patches"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true", help="2 épocas, imgsz reduzido, só pra validar o pipeline")
    parser.add_argument("--fixed-pool", action="store_true",
                         help="protocolo full só: usa datasets_baseline_seg_fixed/ (162 treino/15 teste, "
                              "com as 10 imagens PET-B recuperadas) em vez de datasets_baseline_seg/ "
                              "(155/14) -- ver 00_build_seg_dataset.py --out-dir")
    args = parser.parse_args()
    if args.fixed_pool and args.protocol != "full":
        parser.error("--fixed-pool só se aplica a --protocol full")

    data_yaml = data_yaml_for(args.protocol, args.fixed_pool)
    root_dir = data_yaml.parent
    imgsz = 256 if args.protocol == "patches" else 1024
    epochs = args.epochs or (2 if args.smoke_test else 200)
    exp_suffix = f"{args.protocol}_fixed" if args.fixed_pool else args.protocol
    run_name = f"benchnano_seg_{exp_suffix}"

    print(f"\n{'='*70}\n[09_train_benchnano_seg] protocolo={args.protocol} epochs={epochs} imgsz={imgsz}\n{'='*70}")
    t0 = time.time()
    model = YOLO(WEIGHTS, task="segment")
    model.train(
        data=str(data_yaml), task="segment", epochs=epochs, imgsz=imgsz, batch=-1,
        optimizer="AdamW", lr0=1e-4, lrf=0.1, cos_lr=True, seed=SEED, device="0",
        project=str(cc.RUNS_DIR / "09_benchnano_seg"), name=run_name,
        save_period=-1, patience=max(epochs, 100), plots=True, verbose=True, exist_ok=True,
    )
    duration = time.time() - t0

    best_pt = cc.RUNS_DIR / "09_benchnano_seg" / run_name / "weights" / "best.pt"
    metrics = eval_test_set(best_pt, data_yaml, root_dir)
    print(f"[09_train_benchnano_seg] {exp_suffix}: box={metrics['box']} mask={metrics['mask']}")

    if not args.smoke_test:
        cc.save_summary(f"09_benchnano_seg__{exp_suffix}", metrics=metrics, extra_metadata={
            "protocol": args.protocol, "fixed_pool": args.fixed_pool, "weights": WEIGHTS, "best_weights": str(best_pt),
            "epochs": epochs, "imgsz": imgsz,
            "duration_sec": round(duration, 1), "duration_human": cc.human_duration(duration),
        })
    else:
        print(f"[09_train_benchnano_seg] SMOKE TEST OK -- {cc.human_duration(duration)} pra {epochs} épocas "
              f"({cc.human_duration(duration / epochs)}/época)")


if __name__ == "__main__":
    main()
