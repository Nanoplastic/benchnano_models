"""Compara dois experimentos (por padrão, A=real-only vs B=real+sintético) com
rigor estatístico -- Jain, "The Art of Computer Systems Performance Analysis",
Cap. 13.4 "Comparing Two Alternatives" (observações pareadas), complementado
por um teste específico para classificadores pareados (McNemar exato) e por
bootstrap reamostrando na unidade correta.

Motivação (ver docs/resultados.md e Modelos___Nanoplastic___Artigo_2/main.tex,
seção "Statistical Analysis", ainda com \\TBD{} para intervalo de confiança e
teste de significância): até agora os Experimentos A (72.1%) e B (69.2%) foram
comparados como dois números soltos, sem CI nem teste -- exatamente o tipo de
comparação que o Cap. 13 do Jain (e a própria seção de metodologia do artigo)
diz que não deveria ser aceita sem mais rigor.

Três análises, todas sobre o MESMO test set (par a par, imagem por imagem):

1. IC pareado ingênuo (Jain 13.4.1): trata cada recorte/partícula como uma
   observação pareada independente, IC-t para a média da diferença
   (acerto_A - acerto_B). Rápido, mas ignora que vários recortes vêm da
   mesma micrografia-fonte (não são realmente independentes).

2. Bootstrap por IMAGEM-FONTE (a unidade de reamostragem correta, conforme já
   exigido pela seção de metodologia do artigo -- "resampling unit" = a
   micrografia original, não a partícula, pelo mesmo motivo do split
   treino/val/teste em 00_split_images.py: partículas da mesma imagem
   compartilham ruído/textura e não são estatisticamente independentes).
   Reamostra as imagens-fonte com reposição, recalcula a diferença de
   acurácia em cada réplica, e usa os percentis 2.5/97.5 como IC 95%.

3. Teste de McNemar exato (binomial nas discordâncias): teste padrão pra
   comparar dois classificadores no MESMO test set, faz o par certo com a
   pergunta "A e B discordam mais frequentemente a favor de um dos dois do
   que se fosse por acaso?" -- devolve um p-valor exato, o que a seção de
   estatística do artigo pede explicitamente ("report the test name, sample
   size n, ..., exact P value").

A palavra "significativo" só é usada aqui se o teste correspondente sustentar
isso (p<alpha) -- mesma regra que a seção de Statistical Analysis do artigo
já declara pretender seguir.
"""
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import REAL_MANIFEST, REPORTS_DIR, ROOT

TAG_A = "mina_real_only_lora"
TAG_B = "mina_real_plus_synthetic_lora_v2"
LABEL_A = "Experimento A (só real)"
LABEL_B = "Experimento B (real+sintético)"

ALPHA = 0.05
N_BOOTSTRAP = 10_000
SEED = 42


def load_predictions(tag: str) -> pd.DataFrame:
    path = REPORTS_DIR / f"{tag}_real_test_predictions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Não encontrei {path}. Rode primeiro o experimento correspondente "
            f"(05_finetune_real_only.py ou 06_finetune_real_plus_synthetic.py) "
            f"até ele gerar essa predição de teste."
        )
    df = pd.read_csv(path)
    df["correct"] = df["correct"].astype(bool)
    return df[["image_path", "true_label", "pred_label", "correct"]]


def merge_paired(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    merged = df_a.merge(
        df_b, on="image_path", suffixes=("_a", "_b"), validate="one_to_one"
    )
    if len(merged) != len(df_a) or len(merged) != len(df_b):
        raise ValueError(
            f"Test sets diferentes entre os dois experimentos "
            f"(A={len(df_a)}, B={len(df_b)}, pareados={len(merged)}). "
            f"A comparação pareada exige o MESMO conjunto de teste nos dois."
        )
    mismatched_labels = merged[merged["true_label_a"] != merged["true_label_b"]]
    if len(mismatched_labels):
        raise ValueError(
            f"{len(mismatched_labels)} imagens têm true_label diferente entre "
            f"A e B -- os dois experimentos não usaram o mesmo test set."
        )
    return merged


def naive_paired_ci(merged: pd.DataFrame, alpha: float = ALPHA) -> dict:
    """Jain 13.4.1: IC-t pra média da diferença pareada, tratando cada linha
    (recorte de partícula) como observação independente -- ignora a estrutura
    de agrupamento por imagem-fonte, ver bootstrap_by_source_image() abaixo
    pra versão que respeita isso."""
    d = merged["correct_a"].astype(int) - merged["correct_b"].astype(int)
    n = len(d)
    mean_d = d.mean()
    se_d = d.std(ddof=1) / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    return {
        "n": n,
        "mean_diff": mean_d,
        "ci_low": mean_d - t_crit * se_d,
        "ci_high": mean_d + t_crit * se_d,
    }


def bootstrap_by_source_image(
    merged: pd.DataFrame,
    manifest: pd.DataFrame,
    n_bootstrap: int = N_BOOTSTRAP,
    alpha: float = ALPHA,
    seed: int = SEED,
) -> dict:
    """Reamostra por imagem-fonte (a unidade correta -- ver docstring do
    módulo), não por partícula. Cada réplica bootstrap sorteia imagens-fonte
    COM reposição e recalcula a diferença de acurácia usando todos os
    recortes pertencentes às imagens sorteadas."""
    merged = merged.merge(
        manifest[["image_path", "source_image"]], on="image_path", how="left"
    )
    if merged["source_image"].isna().any():
        raise ValueError(
            "Algum image_path das predições não foi encontrado no manifest_real.csv "
            "-- confira se REAL_MANIFEST corresponde ao mesmo split usado nos experimentos."
        )

    rng = np.random.default_rng(seed)
    source_images = merged["source_image"].unique()
    diffs = np.empty(n_bootstrap)

    groups = {s: g for s, g in merged.groupby("source_image")}

    for i in range(n_bootstrap):
        sampled = rng.choice(source_images, size=len(source_images), replace=True)
        rows = pd.concat([groups[s] for s in sampled], ignore_index=True)
        diffs[i] = rows["correct_a"].mean() - rows["correct_b"].mean()

    ci_low, ci_high = np.percentile(diffs, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "n_source_images": len(source_images),
        "n_bootstrap": n_bootstrap,
        "mean_diff": diffs.mean(),
        "ci_low": ci_low,
        "ci_high": ci_high,
    }


def mcnemar_exact(merged: pd.DataFrame) -> dict:
    """Teste de McNemar exato (binomial nas discordâncias) -- o teste padrão
    pra comparar dois classificadores pareados no mesmo test set. Só as
    linhas onde A e B discordam (um acerta, outro erra) entram no teste;
    sob H0 (sem diferença sistemática entre A e B), essas discordâncias
    deveriam se dividir ~50/50 entre "A acerta e B erra" e "B acerta e A erra"."""
    a_only = int(((merged["correct_a"]) & (~merged["correct_b"])).sum())
    b_only = int(((~merged["correct_a"]) & (merged["correct_b"])).sum())
    n_discordant = a_only + b_only
    if n_discordant == 0:
        p_value = 1.0
    else:
        p_value = stats.binomtest(
            min(a_only, b_only), n_discordant, p=0.5, alternative="two-sided"
        ).pvalue
    return {
        "a_correct_b_wrong": a_only,
        "b_correct_a_wrong": b_only,
        "n_discordant": n_discordant,
        "p_value": p_value,
    }


def main():
    print(f"Comparando {LABEL_A} (tag={TAG_A}) vs. {LABEL_B} (tag={TAG_B})\n")

    df_a = load_predictions(TAG_A)
    df_b = load_predictions(TAG_B)
    merged = merge_paired(df_a, df_b)
    n = len(merged)

    acc_a = merged["correct_a"].mean()
    acc_b = merged["correct_b"].mean()
    print(f"n = {n} imagens de teste (pareadas, mesmo test set)")
    print(f"Acurácia {LABEL_A}: {acc_a:.3f}")
    print(f"Acurácia {LABEL_B}: {acc_b:.3f}")
    print(f"Diferença (A - B): {acc_a - acc_b:+.3f}\n")

    print("=== 1. IC pareado ingênuo (Jain 13.4.1, por partícula) ===")
    naive = naive_paired_ci(merged)
    sig_naive = not (naive["ci_low"] <= 0 <= naive["ci_high"])
    print(f"Diferença média: {naive['mean_diff']:+.4f}")
    print(f"IC 95%: [{naive['ci_low']:+.4f}, {naive['ci_high']:+.4f}]")
    print(f"Exclui zero (diferença significativa neste teste)? {'SIM' if sig_naive else 'não'}\n")

    print("=== 2. Bootstrap por imagem-fonte (unidade de reamostragem correta) ===")
    if not REAL_MANIFEST.exists():
        print(f"AVISO: {REAL_MANIFEST} não encontrado -- pulando bootstrap por imagem-fonte.")
        boot = None
    else:
        manifest = pd.read_csv(REAL_MANIFEST)
        boot = bootstrap_by_source_image(merged, manifest)
        sig_boot = not (boot["ci_low"] <= 0 <= boot["ci_high"])
        print(f"{boot['n_source_images']} imagens-fonte distintas, {boot['n_bootstrap']} réplicas bootstrap")
        print(f"Diferença média: {boot['mean_diff']:+.4f}")
        print(f"IC 95%: [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}]")
        print(f"Exclui zero (diferença significativa neste teste)? {'SIM' if sig_boot else 'não'}\n")

    print("=== 3. Teste de McNemar exato (discordâncias pareadas) ===")
    mcnemar = mcnemar_exact(merged)
    sig_mcnemar = mcnemar["p_value"] < ALPHA
    print(f"A certo / B errado: {mcnemar['a_correct_b_wrong']}")
    print(f"B certo / A errado: {mcnemar['b_correct_a_wrong']}")
    print(f"n discordante: {mcnemar['n_discordant']}")
    print(f"p-valor (exato, bicaudal): {mcnemar['p_value']:.4f}")
    print(f"Significativo em alpha={ALPHA}? {'SIM' if sig_mcnemar else 'não'}\n")

    print("=== Conclusão ===")
    if sig_mcnemar:
        melhor = LABEL_A if acc_a > acc_b else LABEL_B
        print(
            f"McNemar exato rejeita H0 (p={mcnemar['p_value']:.4f} < {ALPHA}): "
            f"a diferença de acurácia entre os dois experimentos é estatisticamente "
            f"significativa, favorecendo {melhor}."
        )
    else:
        print(
            f"McNemar exato NÃO rejeita H0 (p={mcnemar['p_value']:.4f} >= {ALPHA}): "
            f"não há evidência estatística suficiente (n={n}) para afirmar que "
            f"{LABEL_A} e {LABEL_B} têm acurácias verdadeiramente diferentes -- "
            f"a diferença observada ({acc_a - acc_b:+.3f}) pode ser ruído de amostragem."
        )

    out_rows = [{
        "tag_a": TAG_A, "tag_b": TAG_B, "n": n,
        "acc_a": acc_a, "acc_b": acc_b, "diff": acc_a - acc_b,
        "naive_ci_low": naive["ci_low"], "naive_ci_high": naive["ci_high"],
        "bootstrap_ci_low": boot["ci_low"] if boot else None,
        "bootstrap_ci_high": boot["ci_high"] if boot else None,
        "mcnemar_p_value": mcnemar["p_value"],
        "mcnemar_significant": sig_mcnemar,
    }]
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / f"compare_{TAG_A}_vs_{TAG_B}.csv"
    pd.DataFrame(out_rows).to_csv(out_path, index=False)
    print(f"\nResumo salvo em {out_path}")


if __name__ == "__main__":
    main()
