"""Compara as três configurações da tarefa de regime de tamanho (zero-shot,
LoRA, full fine-tuning) com o mesmo rigor estatístico já usado em
11_compare_experiments_statistical.py -- Jain, "The Art of Computer Systems
Performance Analysis", cap. 13 (ver raw/jain_performance_analysis_book.md,
achado no graphify-out):

  - Cap. 13.2/13.8 (IC pra média/proporção): cada configuração recebe um IC
    95% da própria acurácia via bootstrap, reamostrando IMAGEM-FONTE (não
    partícula) -- mesma unidade de reamostragem de 11/14, pelo mesmo motivo
    (partículas da mesma imagem não são observações independentes).

  - Cap. 13.4 (Comparing Two Alternatives), estendido pra 3 alternativas:
    McNemar exato pareado em cada um dos 3 pares (zero-shot x LoRA,
    zero-shot x full FT, LoRA x full FT) -- não dá pra comparar as três de
    uma vez com um teste só sem inflar falso positivo, então cada par usa
    seu próprio teste.

  - Correção de múltiplas comparações: com 3 testes pareados no mesmo
    conjunto de dados, o alpha nominal de 0.05 por teste infla a chance de
    achar "significativo" por acaso em pelo menos um dos três. Aplicamos
    correção de Bonferroni (alpha_corrigido = 0.05/3 ≈ 0.0167) -- a mesma
    política que a seção de Statistical Analysis do artigo já declara
    pretender seguir ("Multiple-comparison correction is applied when the
    final analysis requires it").

Roda só depois que 16, 17 e 18 já geraram as predições de teste.
"""
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import REPORTS_DIR, RESULTS_ROOT

CONFIGS = {
    "zero-shot": "size_zeroshot_real_test",
    "LoRA": "size_lora_real_test",
    "full fine-tuning": "size_full_ft_real_test",
}
MANIFEST = RESULTS_ROOT / "15_build_size_regime_manifest" / "data" / "manifest_size_regime.csv"
ALPHA = 0.05
N_BOOTSTRAP = 10_000
SEED = 42


def load_predictions(tag: str) -> pd.DataFrame:
    path = REPORTS_DIR / f"{tag}_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Não encontrei {path}. Rode primeiro o experimento correspondente "
            f"(16/17/18) até ele gerar essa predição de teste."
        )
    df = pd.read_csv(path)
    df["correct"] = df["correct"].astype(bool)
    return df[["image_path", "true_label", "pred_label", "correct"]]


def bootstrap_ci_by_source_image(df: pd.DataFrame, manifest: pd.DataFrame,
                                  n_bootstrap: int = N_BOOTSTRAP, alpha: float = ALPHA,
                                  seed: int = SEED) -> dict:
    merged = df.merge(manifest[["image_path", "source_image"]], on="image_path", how="left")
    if merged["source_image"].isna().any():
        raise ValueError("Algum image_path não foi encontrado no manifest_size_regime.csv.")

    rng = np.random.default_rng(seed)
    source_images = merged["source_image"].unique()
    groups = {s: g for s, g in merged.groupby("source_image")}
    accs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sampled = rng.choice(source_images, size=len(source_images), replace=True)
        rows = pd.concat([groups[s] for s in sampled], ignore_index=True)
        accs[i] = rows["correct"].mean()

    ci_low, ci_high = np.percentile(accs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"n_source_images": len(source_images), "ci_low": ci_low, "ci_high": ci_high}


def mcnemar_exact(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
    merged = df_a.merge(df_b, on="image_path", suffixes=("_a", "_b"), validate="one_to_one")
    a_only = int(((merged["correct_a"]) & (~merged["correct_b"])).sum())
    b_only = int(((~merged["correct_a"]) & (merged["correct_b"])).sum())
    n_discordant = a_only + b_only
    p_value = 1.0 if n_discordant == 0 else stats.binomtest(
        min(a_only, b_only), n_discordant, p=0.5, alternative="two-sided"
    ).pvalue
    return {"n": len(merged), "a_only": a_only, "b_only": b_only,
            "n_discordant": n_discordant, "p_value": p_value}


def main():
    manifest = pd.read_csv(MANIFEST)

    print("=== Acurácia por configuração, com IC 95% (bootstrap por imagem-fonte) ===\n")
    preds = {}
    for name, tag in CONFIGS.items():
        df = load_predictions(tag)
        preds[name] = df
        acc = df["correct"].mean()
        ci = bootstrap_ci_by_source_image(df, manifest)
        print(f"{name:18s} n={len(df):4d}  acc={acc:.3f}  "
              f"IC95%=[{ci['ci_low']:.3f}, {ci['ci_high']:.3f}]  "
              f"({ci['n_source_images']} imagens-fonte, {N_BOOTSTRAP} réplicas)")

    n_pairs = len(list(combinations(CONFIGS.keys(), 2)))
    alpha_corrected = ALPHA / n_pairs
    print(f"\n=== McNemar exato pareado, {n_pairs} pares -- correção de Bonferroni "
          f"(alpha={ALPHA}/{n_pairs}={alpha_corrected:.4f}) ===\n")

    rows = []
    for name_a, name_b in combinations(CONFIGS.keys(), 2):
        res = mcnemar_exact(preds[name_a], preds[name_b])
        sig = res["p_value"] < alpha_corrected
        print(f"{name_a} vs {name_b}: n={res['n']} discordantes={res['n_discordant']} "
              f"({name_a} certo/{name_b} errado={res['a_only']}, o inverso={res['b_only']}) "
              f"p={res['p_value']:.4f}  significativo (Bonferroni)? {'SIM' if sig else 'não'}")
        rows.append({
            "config_a": name_a, "config_b": name_b, **res,
            "alpha_corrected": alpha_corrected, "significant_bonferroni": sig,
        })

    out_path = REPORTS_DIR / "compare_size_regime_configs.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"\nResumo salvo em {out_path}")


if __name__ == "__main__":
    main()
