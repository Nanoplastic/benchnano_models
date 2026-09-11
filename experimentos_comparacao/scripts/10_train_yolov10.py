"""YOLOv10 -- uma das arquiteturas com código de treino original disponível
no repo do MiNa (github.com/naviiidz/MiNa-dataset), por isso mantida na
comparação oficial (ao contrário de RT-DETR/YOLOv26, sem código lá --
main_pt.tex linha 1025). Variante `s`, mesma convenção de tamanho usada
nas outras comparações "s" do paper deles.

Dois protocolos:
    --protocol full     -- experimentos/datasets_yolo26_v2/ (já existe, 4 classes)
    --protocol patches   -- experimentos_comparacao/datasets_curated_crop/det/

Execução única (seed=0) -- ver README sobre o prazo de submissão.

Uso:
    python3 10_train_yolov10.py --protocol full [--epochs 150] [--smoke-test]
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import comp_common as cc

from ultralytics import YOLO

WEIGHTS = "yolov10s.pt"
SEED = 0
CLASS_NAMES = cc.CLASSES


def data_yaml_for(protocol: str, matched_pool: bool = False) -> Path:
    if protocol == "full":
        return cc.DET_DATA_YAML_MATCHED if matched_pool else cc.DET_DATA_YAML
    return cc.CURATED_CROP_DIR / "det" / "data.yaml"


def eval_test_set(model_path, root_dir: Path, imgsz: int) -> dict:
    model = YOLO(str(model_path))
    data_yaml_test = root_dir / "data_test.yaml"
    if not data_yaml_test.exists():
        data_yaml_test.write_text(
            f"path: {root_dir}\ntrain: images/test\nval: images/test\n"
            f"nc: {len(CLASS_NAMES)}\nnames:\n" + "\n".join(f"  - {c}" for c in CLASS_NAMES) + "\n"
        )
    metrics = model.val(data=str(data_yaml_test), split="val", imgsz=imgsz,
                         conf=0.15, iou=0.7, plots=False, verbose=False)
    agg = {
        "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
        "f1": float(2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr))
              if (metrics.box.mp + metrics.box.mr) > 0 else 0.0,
    }
    return {"aggregate_metrics": agg}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", required=True, choices=["full", "patches"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--matched-pool", action="store_true",
                         help="protocolo full só: usa o mesmo pool de imagens de datasets_baseline_seg/ "
                              "(155 treino/14 teste) em vez do dataset completo (162/15) -- ver "
                              "16_build_det_full_matched.py")
    args = parser.parse_args()
    if args.matched_pool and args.protocol != "full":
        parser.error("--matched-pool só se aplica a --protocol full")

    data_yaml = data_yaml_for(args.protocol, args.matched_pool)
    root_dir = data_yaml.parent
    imgsz = 256 if args.protocol == "patches" else 1024
    epochs = args.epochs or (2 if args.smoke_test else 200)
    exp_suffix = f"{args.protocol}_matched" if args.matched_pool else args.protocol
    run_name = f"yolov10_{exp_suffix}"

    print(f"\n{'='*70}\n[10_train_yolov10] protocolo={args.protocol} epochs={epochs} imgsz={imgsz}\n{'='*70}")
    t0 = time.time()
    model = YOLO(WEIGHTS, task="detect")
    model.train(
        data=str(data_yaml), task="detect", epochs=epochs, imgsz=imgsz, batch=-1,
        optimizer="AdamW", lr0=1e-4, lrf=0.1, cos_lr=True, seed=SEED, device="0",
        project=str(cc.RUNS_DIR / "10_yolov10"), name=run_name,
        save_period=-1, patience=max(epochs, 100), plots=True, verbose=True, exist_ok=True,
    )
    duration = time.time() - t0

    best_pt = cc.RUNS_DIR / "10_yolov10" / run_name / "weights" / "best.pt"
    metrics = eval_test_set(best_pt, root_dir, imgsz)
    print(f"[10_train_yolov10] {args.protocol}: {metrics['aggregate_metrics']}")

    if not args.smoke_test:
        cc.save_summary(f"10_yolov10__{exp_suffix}", metrics=metrics, extra_metadata={
            "protocol": args.protocol, "matched_pool": args.matched_pool, "weights": WEIGHTS, "best_weights": str(best_pt),
            "epochs": epochs, "imgsz": imgsz,
            "duration_sec": round(duration, 1), "duration_human": cc.human_duration(duration),
        })
    else:
        print(f"[10_train_yolov10] SMOKE TEST OK -- {cc.human_duration(duration)} pra {epochs} épocas "
              f"({cc.human_duration(duration / epochs)}/época)")


if __name__ == "__main__":
    main()
