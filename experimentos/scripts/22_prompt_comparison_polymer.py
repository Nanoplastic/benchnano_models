"""Comparação de prompts pro SmolVLM zero-shot na tarefa de classificação de
POLÍMERO (PE/PET/PP/PS) -- mesma metodologia do 21_prompt_comparison.py
(Goodfellow cap. 11: selecao só na validação; Jain cap. 13.4: comparação
pareada com IC por bootstrap e McNemar exato contra o prompt-baseline já
usado em toda a linha de polímero, scripts 04/05/06/08/09).

Cinco variações, cada uma testando um eixo diferente:

  P1 baseline    -- o CLASSIFY_PROMPT já usado em 04/05/06/09 (e em todo o
                    piloto VLM-only de polímero), sem mudança.
  P2 morfologia  -- injeta as pistas heurísticas de morfologia por polímero
                    já usadas em 08_describe_then_classify.py (aparência
                    típica de cada plástico no MEV), pra ver se dar essa
                    informação de antemão ajuda o zero-shot (sem o custo do
                    pipeline de duas etapas do 08, que piorou o resultado).
  P3 minimo      -- versão bem mais enxuta.
  P4 raciocinio  -- pede uma frase de raciocínio sobre forma/textura antes
                    da resposta final (chain-of-thought).
  P5 eliminacao  -- pede pra descartar candidatos antes de decidir, testando
                    se enquadrar como eliminação (em vez de escolha direta)
                    ajuda em casos ambíguos como PP vs PET.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import (
    LABELS, load_base_model, load_real_split, run_classification_eval,
    save_experiment_summary, setup_experiment_logging,
)

EXP_NAME = "22_prompt_comparison_polymer"
EXP_DIR, LOG_PATH, LOGGER, RUNTIME_METADATA = setup_experiment_logging(EXP_NAME, __file__)

REPORTS_DIR = EXP_DIR / "reports"
BASELINE_TEST_PREDICTIONS = Path(__file__).resolve().parent.parent / "resultados" / "reports" / "zeroshot_real_test_predictions.csv"

ALPHA = 0.05
N_BOOTSTRAP = 10_000
SEED = 42

PROMPTS = {
    "P1_baseline": (
        "This is a scanning electron microscope (SEM) image from a controlled "
        "reference sample containing a single known plastic polymer, showing "
        "particle(s) deposited on a filter membrane. Which polymer is this "
        "sample made of: PE (polyethylene), PET (polyethylene terephthalate), "
        "PP (polypropylene), or PS (polystyrene)? Answer with just one label: "
        "PE, PET, PP, or PS."
    ),
    "P2_morfologia": (
        "This is a scanning electron microscope (SEM) image from a controlled "
        "reference sample containing a single known plastic polymer. PE "
        "(polyethylene) fragments typically appear as irregular, waxy, folded "
        "flakes with smooth rounded edges. PET (polyethylene terephthalate) "
        "fragments typically appear as angular, glassy fragments with sharp "
        "edges, sometimes fibrous strands. PP (polypropylene) fragments "
        "typically appear as irregular blocky fragments with a rougher, more "
        "granular surface texture. PS (polystyrene) particles typically "
        "appear as smooth spherical beads or brittle angular fragments with "
        "a glassy surface. Based on the particle(s) shown, which polymer is "
        "this sample made of? Answer with just one label: PE, PET, PP, or PS."
    ),
    "P3_minimo": (
        "SEM image of a plastic particle sample. Which polymer: PE, PET, PP, "
        "or PS? Answer with one label only."
    ),
    "P4_raciocinio": (
        "This is a scanning electron microscope (SEM) image from a controlled "
        "reference sample containing a single known plastic polymer, showing "
        "particle(s) deposited on a filter membrane. In one short sentence, "
        "describe the particle's shape, edges, and surface texture. Then, on "
        "a new line, give your final answer as exactly one label: PE, PET, "
        "PP, or PS."
    ),
    "P5_eliminacao": (
        "This is a scanning electron microscope (SEM) image from a controlled "
        "reference sample containing a single known plastic polymer, showing "
        "particle(s) deposited on a filter membrane. Consider the four "
        "candidates PE, PET, PP, and PS. Rule out the ones that clearly do "
        "not match the visible shape and surface texture, then answer with "
        "the single most likely label: PE, PET, PP, or PS."
    ),
}


def bootstrap_ci(df: pd.DataFrame, manifest: pd.DataFrame, n_bootstrap=N_BOOTSTRAP, alpha=ALPHA, seed=SEED):
    merged = df.merge(manifest[["image_path", "source_image"]], on="image_path", how="left")
    rng = np.random.default_rng(seed)
    imgs = merged["source_image"].unique()
    groups = {s: g for s, g in merged.groupby("source_image")}
    accs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sampled = rng.choice(imgs, size=len(imgs), replace=True)
        rep = pd.concat([groups[s] for s in sampled], ignore_index=True)
        accs[i] = rep["correct"].mean()
    lo, hi = np.percentile(accs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return lo, hi, len(imgs)


def mcnemar_exact(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
    merged = df_a.merge(df_b, on="image_path", suffixes=("_a", "_b"), validate="one_to_one")
    a_only = int(((merged["correct_a"]) & (~merged["correct_b"])).sum())
    b_only = int(((~merged["correct_a"]) & (merged["correct_b"])).sum())
    n_discordant = a_only + b_only
    p = 1.0 if n_discordant == 0 else stats.binomtest(
        min(a_only, b_only), n_discordant, p=0.5, alternative="two-sided"
    ).pvalue
    return {"a_only": a_only, "b_only": b_only, "n_discordant": n_discordant, "p_value": p}


def main():
    val_df = load_real_split("val")
    test_df = load_real_split("test")
    from vlm_common import REAL_MANIFEST
    full_manifest = pd.read_csv(REAL_MANIFEST)

    LOGGER.info("=== selecao de prompt (polimero) na VALIDACAO (n=%s), zero-shot ===", len(val_df))
    print(f"=== Etapa 1: selecao de prompt na validacao (n={len(val_df)}), zero-shot ===\n")

    processor, model = load_base_model()

    val_results = {}
    for name, prompt in PROMPTS.items():
        _, acc = run_classification_eval(
            processor, model, val_df, f"prompt_poly_{name}_val",
            out_dir=REPORTS_DIR, prompt=prompt, labels=LABELS,
        )
        val_results[name] = acc
        print(f"{name:15s} acc_val={acc:.3f}")
        LOGGER.info("prompt=%s acc_val=%.4f", name, acc)

    winner = max(val_results, key=val_results.get)
    print(f"\nPrompt vencedor na validacao: {winner} (acc_val={val_results[winner]:.3f})")
    LOGGER.info("prompt_vencedor=%s acc_val=%.4f", winner, val_results[winner])

    print(f"\n=== Etapa 2: avaliacao UNICA do prompt vencedor no teste travado (n={len(test_df)}) ===\n")
    winner_results, winner_test_acc = run_classification_eval(
        processor, model, test_df, f"prompt_poly_{winner}_test",
        out_dir=REPORTS_DIR, prompt=PROMPTS[winner], labels=LABELS,
    )
    print(f"Acuracia do prompt vencedor ({winner}) no teste: {winner_test_acc:.3f}")
    print("Matriz de confusao:")
    print(pd.crosstab(winner_results["true_label"], winner_results["pred_label"]))
    print("\nRecall por classe:")
    print(winner_results.groupby("true_label")["correct"].mean())

    lo, hi, n_imgs = bootstrap_ci(winner_results, full_manifest)
    print(f"\nIC 95% (bootstrap por imagem, {N_BOOTSTRAP} replicas, {n_imgs} imagens): [{lo:.3f}, {hi:.3f}]")

    print(f"\n=== Etapa 3: comparacao pareada contra o prompt-baseline (P1, ja usado em 04/05/06/09) ===\n")
    if not BASELINE_TEST_PREDICTIONS.exists():
        print(f"AVISO: {BASELINE_TEST_PREDICTIONS} nao encontrado -- pulando comparacao pareada.")
        mcnemar_result = None
    else:
        baseline_df = pd.read_csv(BASELINE_TEST_PREDICTIONS)
        baseline_df["correct"] = baseline_df["correct"].astype(bool)
        winner_results["correct"] = winner_results["correct"].astype(bool)
        mcnemar_result = mcnemar_exact(
            baseline_df[["image_path", "correct"]], winner_results[["image_path", "correct"]]
        )
        acc_baseline = baseline_df["correct"].mean()
        sig = mcnemar_result["p_value"] < ALPHA
        print(f"Acuracia baseline (P1) no teste: {acc_baseline:.3f}")
        print(f"Acuracia vencedor ({winner}) no teste: {winner_test_acc:.3f}")
        print(f"McNemar exato: baseline_certo/vencedor_errado={mcnemar_result['a_only']}  "
              f"vencedor_certo/baseline_errado={mcnemar_result['b_only']}  "
              f"discordantes={mcnemar_result['n_discordant']}  p={mcnemar_result['p_value']:.4f}")
        print(f"Diferenca estatisticamente significativa (alpha=0.05)? {'SIM' if sig else 'nao'}")

    out_rows = [{"prompt": name, "acc_val": acc} for name, acc in val_results.items()]
    pd.DataFrame(out_rows).to_csv(REPORTS_DIR / "prompt_comparison_polymer_val_accuracies.csv", index=False)

    save_experiment_summary(EXP_NAME, RUNTIME_METADATA, {
        "val_accuracies_by_prompt": val_results,
        "winner_prompt": winner,
        "winner_test_acc": float(winner_test_acc),
        "winner_test_ci95": [float(lo), float(hi)],
        "mcnemar_vs_baseline": mcnemar_result,
        "prompts": PROMPTS,
    })
    print(f"\nLog completo: {LOG_PATH}")


if __name__ == "__main__":
    main()
