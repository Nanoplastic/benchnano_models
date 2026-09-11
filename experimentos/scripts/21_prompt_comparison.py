"""Comparação de prompts pro SmolVLM zero-shot na tarefa de regime de
tamanho -- experimento que faltava e que o próprio main.tex já promete
("The prompt is finalized using development data") sem nunca ter sido
executado: até aqui usamos SEMPRE o mesmo prompt fixo (SIZE_CLASSIFY_PROMPT
em vlm_common.py), sem nunca comparar variações de redação.

Metodologia (ver graphify-out/GRAPH_REPORT.md e raw/jain_performance_
analysis_book.md, raw/goodfellow_deep_learning_book.md, já catalogados
neste projeto):

  - Goodfellow, cap. 11 (Practical Methodology): a escolha de configuração
    usa SÓ o conjunto de validação. O teste travado só é tocado UMA vez,
    depois que o prompt vencedor já está decidido -- a mesma disciplina já
    usada em toda seleção de checkpoint neste projeto (nunca escolher nada
    olhando pro teste).

  - Jain, cap. 13.4 (Comparing Two Alternatives), estendido pra várias
    alternativas: acurácia com IC 95% por bootstrap (imagem-fonte) pra cada
    prompt, e comparação pareada por McNemar exato entre o prompt vencedor
    e o prompt-baseline (o que já era usado em todos os experimentos
    anteriores), com o mesmo n do teste travado.

Cinco variações de prompt, cada uma testando um eixo diferente de práticas
conhecidas de prompt engineering pra VLM (não uma citação específica, é
prática geral do campo):

  P1 baseline    -- o prompt já usado em 16/17/18/20, sem mudança.
  P2 escala      -- pede explicitamente pro modelo usar a barra de escala/
                    magnificação visível na imagem antes de decidir (o
                    próprio main.tex enfatiza que a escala física é
                    necessária, não só o tamanho em pixel).
  P3 minimo      -- versão bem mais enxuta, testa se elaboração demais
                    atrapalha.
  P4 raciocinio  -- pede um raciocínio breve antes da resposta final
                    (chain-of-thought), técnica geral de prompting.
  P5 conversao   -- reforça explicitamente a conversão de unidade
                    (1 micrometro = 1000 nanometros) e pede comparação
                    direta contra a barra de escala impressa na imagem.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import (
    RESULTS_ROOT, SIZE_LABELS, load_base_model, run_classification_eval,
    save_experiment_summary, setup_experiment_logging,
)

EXP_NAME = "21_prompt_comparison"
EXP_DIR, LOG_PATH, LOGGER, RUNTIME_METADATA = setup_experiment_logging(EXP_NAME, __file__)

MANIFEST = RESULTS_ROOT / "15_build_size_regime_manifest" / "data" / "manifest_size_regime.csv"
REPORTS_DIR = EXP_DIR / "reports"
BASELINE_TEST_PREDICTIONS = RESULTS_ROOT / "16_size_zeroshot_eval" / "reports" / "size_zeroshot_real_test_predictions.csv"

ALPHA = 0.05
N_BOOTSTRAP = 10_000
SEED = 42

PROMPTS = {
    "P1_baseline": (
        "You are analyzing a scanning electron microscopy image containing "
        "plastic particles. The highlighted particle was detected by an "
        "object-detection model together with the image's physical scale "
        "information. Return NANOPLASTIC when the particle's characteristic "
        "size is below 1 micrometer. Return MICROPLASTIC otherwise. Answer "
        "with just one label: NANOPLASTIC or MICROPLASTIC."
    ),
    "P2_escala": (
        "You are analyzing a scanning electron microscopy image of a plastic "
        "particle. Before answering, look at the scale bar and magnification "
        "value printed in the image, and use them to judge the particle's "
        "real physical size -- do not judge size from pixel dimensions alone. "
        "Return NANOPLASTIC when the particle's characteristic size is below "
        "1 micrometer. Return MICROPLASTIC otherwise. Answer with just one "
        "label: NANOPLASTIC or MICROPLASTIC."
    ),
    "P3_minimo": (
        "SEM image of a plastic particle. Is its size below or above "
        "1 micrometer? Answer NANOPLASTIC (below) or MICROPLASTIC (above)."
    ),
    "P4_raciocinio": (
        "You are analyzing a scanning electron microscopy image of a plastic "
        "particle. In one short sentence, reason about the particle's size "
        "relative to the scale bar shown in the image. Then, on a new line, "
        "give your final answer as exactly one label: NANOPLASTIC if the "
        "characteristic size is below 1 micrometer, or MICROPLASTIC if it is "
        "1 micrometer or above."
    ),
    "P5_conversao": (
        "You are analyzing a scanning electron microscopy image of a plastic "
        "particle. Remember: 1 micrometer equals 1000 nanometers. Compare the "
        "particle's visible size directly against the printed scale bar. "
        "Return NANOPLASTIC when the particle's characteristic size is below "
        "1 micrometer (1000 nanometers). Return MICROPLASTIC otherwise. "
        "Answer with just one label: NANOPLASTIC or MICROPLASTIC."
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
    if not MANIFEST.exists():
        raise FileNotFoundError(f"{MANIFEST} não existe -- rode 15_build_size_regime_manifest.py primeiro.")

    df_manifest = pd.read_csv(MANIFEST)
    val_df = df_manifest[df_manifest["split"] == "val"].reset_index(drop=True)
    test_df = df_manifest[df_manifest["split"] == "test"].reset_index(drop=True)

    LOGGER.info("=== selecao de prompt na VALIDACAO (n=%s), zero-shot ===", len(val_df))
    print(f"=== Etapa 1: selecao de prompt na validacao (n={len(val_df)}), zero-shot ===\n")

    processor, model = load_base_model()

    val_results = {}
    for name, prompt in PROMPTS.items():
        _, acc = run_classification_eval(
            processor, model, val_df, f"prompt_{name}_val",
            out_dir=REPORTS_DIR, prompt=prompt, labels=SIZE_LABELS,
        )
        val_results[name] = acc
        print(f"{name:15s} acc_val={acc:.3f}")
        LOGGER.info("prompt=%s acc_val=%.4f", name, acc)

    winner = max(val_results, key=val_results.get)
    print(f"\nPrompt vencedor na validacao: {winner} (acc_val={val_results[winner]:.3f})")
    LOGGER.info("prompt_vencedor=%s acc_val=%.4f", winner, val_results[winner])

    print(f"\n=== Etapa 2: avaliacao UNICA do prompt vencedor no teste travado (n={len(test_df)}) ===\n")
    winner_results, winner_test_acc = run_classification_eval(
        processor, model, test_df, f"prompt_{winner}_test",
        out_dir=REPORTS_DIR, prompt=PROMPTS[winner], labels=SIZE_LABELS,
    )
    print(f"Acuracia do prompt vencedor ({winner}) no teste: {winner_test_acc:.3f}")
    print("Matriz de confusao:")
    print(pd.crosstab(winner_results["true_label"], winner_results["pred_label"]))

    lo, hi, n_imgs = bootstrap_ci(winner_results, df_manifest)
    print(f"IC 95% (bootstrap por imagem, {N_BOOTSTRAP} replicas, {n_imgs} imagens): [{lo:.3f}, {hi:.3f}]")

    print(f"\n=== Etapa 3: comparacao pareada contra o prompt-baseline (P1, ja usado em 16/17/18/20) ===\n")
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
    pd.DataFrame(out_rows).to_csv(REPORTS_DIR / "prompt_comparison_val_accuracies.csv", index=False)

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
