"""Baseline zero-shot do SmolVLM-500M-Instruct na tarefa de regime de
tamanho (NANOPLASTIC/MICROPLASTIC, limiar 1µm) -- Camada 2 do artigo do
colega, condição "reference-box" (recorte já usa a bbox de referência do
MiNa, com margem de 10px -- mesma convenção da linha de polímero).

Sem treino nenhum. Roda primeiro nessa linha, mesmo diagnóstico de sempre:
ver se o modelo colapsa numa classe default antes de gastar tempo com as
duas rodadas de fine-tuning (17 e 18).
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import (
    RESULTS_ROOT, SIZE_CLASSIFY_PROMPT, SIZE_LABELS, load_base_model,
    run_classification_eval, save_experiment_summary, setup_experiment_logging,
)

EXP_NAME = "16_size_zeroshot_eval"
EXP_DIR, LOG_PATH, LOGGER, RUNTIME_METADATA = setup_experiment_logging(EXP_NAME, __file__)

MANIFEST = RESULTS_ROOT / "15_build_size_regime_manifest" / "data" / "manifest_size_regime.csv"
REPORTS_DIR = EXP_DIR / "reports"


def main():
    if not MANIFEST.exists():
        raise FileNotFoundError(f"{MANIFEST} não existe -- rode 15_build_size_regime_manifest.py primeiro.")

    df = pd.read_csv(MANIFEST)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    LOGGER.info("iniciando baseline zero-shot (regime de tamanho) no split real test")
    LOGGER.info("n_test=%s", len(test_df))
    print(f"avaliando zero-shot (NANO/MICRO) em {len(test_df)} recortes reais de teste...")

    processor, model = load_base_model()
    results, acc = run_classification_eval(
        processor, model, test_df, "size_zeroshot_real_test",
        out_dir=REPORTS_DIR, prompt=SIZE_CLASSIFY_PROMPT, labels=SIZE_LABELS,
    )
    pred_counts = results["pred_label"].value_counts().to_dict()
    print(results["pred_label"].value_counts())
    print(f"\nacurácia zero-shot regime de tamanho (real test, n={len(test_df)}): {acc:.3f}")
    print("\nMatriz de confusão (linha=real, coluna=previsto):")
    print(pd.crosstab(results["true_label"], results["pred_label"]))

    summary = {
        "n_test": len(test_df),
        "acc": float(acc),
        "pred_label_counts": pred_counts,
        "output_csv": str(REPORTS_DIR / "size_zeroshot_real_test_predictions.csv"),
    }
    save_experiment_summary(EXP_NAME, RUNTIME_METADATA, summary)
    LOGGER.info("Resumo zero-shot regime de tamanho: n_test=%s acc=%.4f pred_counts=%s", len(test_df), acc, pred_counts)


if __name__ == "__main__":
    main()
