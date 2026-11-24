"""Comparação estatística entre dois experimentos / Statistical comparison of two experiments.

PT: Compara dois experimentos de classificação avaliados no mesmo conjunto de teste (por
    padrão, A = só real vs B = real + sintético), recorte a recorte. Usa observações pareadas, com um teste
    específico para classificadores pareados (McNemar exato) e bootstrap na unidade de
    reamostragem correta.
    Três análises:
    1. IC pareado ingênuo: cada recorte é uma observação pareada independente;
    IC-t da média da diferença (acerto_A - acerto_B). Ignora que vários recortes vêm da
    mesma micrografia.
    2. Bootstrap por imagem de origem: reamostra micrografias com reposição (partículas da
    mesma imagem compartilham ruído e textura e não são independentes, mesmo motivo da
    divisão em 00_split_images.py), recalcula a diferença de acurácia em cada réplica e usa
    os percentis 2,5/97,5 como IC 95%.
    3. McNemar exato (binomial nas discordâncias): verifica se A e B discordam a favor de um
    deles mais do que o esperado ao acaso; devolve p-valor exato.
    O rótulo "significativo" só é atribuído quando o teste correspondente dá p < alfa.
EN: Compares two classification experiments evaluated on the same test set (by default, A =
    real only vs B = real + synthetic), crop by crop. Uses paired observations, plus a test specific to
    paired classifiers (exact McNemar) and a bootstrap on the correct resampling unit.
    Three analyses:
    1. Naive paired CI: each crop is an independent paired observation; t-CI
    of the mean difference (correct_A - correct_B). Ignores that several crops come from the
    same micrograph.
    2. Source-image bootstrap: resamples micrographs with replacement (particles from the
    same image share noise and texture and are not independent, same reason as the split in
    00_split_images.py), recomputes the accuracy difference in each replicate and uses the
    2.5/97.5 percentiles as the 95% CI.
    3. Exact McNemar (binomial on discordant pairs): checks whether A and B disagree in
    favour of one of them more than expected by chance; returns an exact p-value.
    The "significant" label is only assigned when the corresponding test gives p < alpha.
"""
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import MANIFESTO_REAL, DIR_RELATORIOS, RAIZ

ROTULO_EXP_A = "mina_real_only_lora"
ROTULO_EXP_B = "mina_real_plus_synthetic_lora_v2"
ROTULO_A = "Experimento A (só real)"
ROTULO_B = "Experimento B (real+sintético)"

ALFA = 0.05
N_BOOTSTRAP = 10_000
SEMENTE = 42


def carregar_predicoes(rotulo_exp: str) -> pd.DataFrame:
    caminho = DIR_RELATORIOS / f"{rotulo_exp}_real_test_predictions.csv"
    if not caminho.exists():
        raise FileNotFoundError(
            f"Não encontrei {caminho}. Rode primeiro o experimento correspondente "
            f"(05_finetune_real_only.py ou 06_finetune_real_plus_synthetic.py) "
            f"até ele gerar essa predição de teste."
        )
    df = pd.read_csv(caminho)
    df["correct"] = df["correct"].astype(bool)
    return df[["image_path", "true_label", "pred_label", "correct"]]


def mesclar_pareados(df_a: pd.DataFrame, df_b: pd.DataFrame) -> pd.DataFrame:
    mesclado = df_a.merge(
        df_b, on="image_path", suffixes=("_a", "_b"), validate="one_to_one"
    )
    if len(mesclado) != len(df_a) or len(mesclado) != len(df_b):
        raise ValueError(
            f"Test sets diferentes entre os dois experimentos "
            f"(A={len(df_a)}, B={len(df_b)}, pareados={len(mesclado)}). "
            f"A comparação pareada exige o MESMO conjunto de teste nos dois."
        )
    rotulos_divergentes = mesclado[mesclado["true_label_a"] != mesclado["true_label_b"]]
    if len(rotulos_divergentes):
        raise ValueError(
            f"{len(rotulos_divergentes)} imagens têm true_label diferente entre "
            f"A e B -- os dois experimentos não usaram o mesmo test set."
        )
    return mesclado


def ic_pareado_ingenuo(mesclado: pd.DataFrame, alfa: float = ALFA) -> dict:
    """PT: IC-t da média da diferença pareada, tratando cada linha (recorte de
        partícula) como observação independente. Ignora o agrupamento por imagem de origem;
        ver bootstrap_por_imagem_origem() para a versão que o respeita.
    EN: t-CI of the mean paired difference, treating each row (particle crop)
        as an independent observation. Ignores grouping by source image; see
        bootstrap_por_imagem_origem() for the version that accounts for it.
    """
    d = mesclado["correct_a"].astype(int) - mesclado["correct_b"].astype(int)
    n = len(d)
    media_d = d.mean()
    se_d = d.std(ddof=1) / np.sqrt(n)
    t_critico = stats.t.ppf(1 - alfa / 2, df=n - 1)
    return {
        "n": n,
        "mean_diff": media_d,
        "ci_low": media_d - t_critico * se_d,
        "ci_high": media_d + t_critico * se_d,
    }


def bootstrap_por_imagem_origem(
    mesclado: pd.DataFrame,
    manifesto: pd.DataFrame,
    n_bootstrap: int = N_BOOTSTRAP,
    alfa: float = ALFA,
    semente: int = SEMENTE,
) -> dict:
    """PT: Bootstrap por imagem de origem, não por partícula. Cada réplica sorteia imagens de
        origem com reposição e recalcula a diferença de acurácia com todos os recortes das
        imagens sorteadas.
    EN: Bootstrap by source image, not by particle. Each replicate draws source images with
        replacement and recomputes the accuracy difference using every crop from the drawn
        images.
    """
    mesclado = mesclado.merge(
        manifesto[["image_path", "source_image"]], on="image_path", how="left"
    )
    if mesclado["source_image"].isna().any():
        raise ValueError(
            "Algum image_path das predições não foi encontrado no manifest_real.csv "
            "-- confira se REAL_MANIFEST corresponde ao mesmo split usado nos experimentos."
        )

    gerador = np.random.default_rng(semente)
    imagens_origem = mesclado["source_image"].unique()
    diferencas = np.empty(n_bootstrap)

    grupos = {s: g for s, g in mesclado.groupby("source_image")}

    for i in range(n_bootstrap):
        amostrados = gerador.choice(imagens_origem, size=len(imagens_origem), replace=True)
        linhas_tab = pd.concat([grupos[s] for s in amostrados], ignore_index=True)
        diferencas[i] = linhas_tab["correct_a"].mean() - linhas_tab["correct_b"].mean()

    ic_inf, ic_sup = np.percentile(diferencas, [100 * alfa / 2, 100 * (1 - alfa / 2)])
    return {
        "n_source_images": len(imagens_origem),
        "n_bootstrap": n_bootstrap,
        "mean_diff": diferencas.mean(),
        "ci_low": ic_inf,
        "ci_high": ic_sup,
    }


def mcnemar_exato(mesclado: pd.DataFrame) -> dict:
    """PT: Teste de McNemar exato (binomial nas discordâncias) para dois classificadores
        pareados no mesmo conjunto de teste. Só entram as linhas em que A e B discordam; sob
        H0 (sem diferença sistemática) essas discordâncias se dividem ~50/50 entre "A acerta
        e B erra" e "B acerta e A erra".
    EN: Exact McNemar test (binomial on discordant pairs) for two paired classifiers on the
        same test set. Only rows where A and B disagree are used; under H0 (no systematic
        difference) these disagreements split ~50/50 between "A right, B wrong" and "B
        right, A wrong".
    """
    so_a = int(((mesclado["correct_a"]) & (~mesclado["correct_b"])).sum())
    so_b = int(((~mesclado["correct_a"]) & (mesclado["correct_b"])).sum())
    n_discordantes = so_a + so_b
    if n_discordantes == 0:
        valor_p = 1.0
    else:
        valor_p = stats.binomtest(
            min(so_a, so_b), n_discordantes, p=0.5, alternative="two-sided"
        ).pvalue
    return {
        "a_correct_b_wrong": so_a,
        "b_correct_a_wrong": so_b,
        "n_discordant": n_discordantes,
        "p_value": valor_p,
    }


def principal():
    print(f"Comparando {ROTULO_A} (tag={ROTULO_EXP_A}) vs. {ROTULO_B} (tag={ROTULO_EXP_B})\n")

    df_a = carregar_predicoes(ROTULO_EXP_A)
    df_b = carregar_predicoes(ROTULO_EXP_B)
    mesclado = mesclar_pareados(df_a, df_b)
    n = len(mesclado)

    acur_a = mesclado["correct_a"].mean()
    acur_b = mesclado["correct_b"].mean()
    print(f"n = {n} imagens de teste (pareadas, mesmo test set)")
    print(f"Acurácia {ROTULO_A}: {acur_a:.3f}")
    print(f"Acurácia {ROTULO_B}: {acur_b:.3f}")
    print(f"Diferença (A - B): {acur_a - acur_b:+.3f}\n")

    print("=== 1. IC pareado ingênuo (por partícula) ===")
    ingenuo = ic_pareado_ingenuo(mesclado)
    sig_ingenuo = not (ingenuo["ci_low"] <= 0 <= ingenuo["ci_high"])
    print(f"Diferença média: {ingenuo['mean_diff']:+.4f}")
    print(f"IC 95%: [{ingenuo['ci_low']:+.4f}, {ingenuo['ci_high']:+.4f}]")
    print(f"Exclui zero (diferença significativa neste teste)? {'SIM' if sig_ingenuo else 'não'}\n")

    print("=== 2. Bootstrap por imagem-fonte (unidade de reamostragem correta) ===")
    if not MANIFESTO_REAL.exists():
        print(f"AVISO: {MANIFESTO_REAL} não encontrado -- pulando bootstrap por imagem-fonte.")
        reamostras = None
    else:
        manifesto = pd.read_csv(MANIFESTO_REAL)
        reamostras = bootstrap_por_imagem_origem(mesclado, manifesto)
        sig_bootstrap = not (reamostras["ci_low"] <= 0 <= reamostras["ci_high"])
        print(f"{reamostras['n_source_images']} imagens-fonte distintas, {reamostras['n_bootstrap']} réplicas bootstrap")
        print(f"Diferença média: {reamostras['mean_diff']:+.4f}")
        print(f"IC 95%: [{reamostras['ci_low']:+.4f}, {reamostras['ci_high']:+.4f}]")
        print(f"Exclui zero (diferença significativa neste teste)? {'SIM' if sig_bootstrap else 'não'}\n")

    print("=== 3. Teste de McNemar exato (discordâncias pareadas) ===")
    mcnemar = mcnemar_exato(mesclado)
    significativo_mcnemar = mcnemar["p_value"] < ALFA
    print(f"A certo / B errado: {mcnemar['a_correct_b_wrong']}")
    print(f"B certo / A errado: {mcnemar['b_correct_a_wrong']}")
    print(f"n discordante: {mcnemar['n_discordant']}")
    print(f"p-valor (exato, bicaudal): {mcnemar['p_value']:.4f}")
    print(f"Significativo em alpha={ALFA}? {'SIM' if significativo_mcnemar else 'não'}\n")

    print("=== Conclusão ===")
    if significativo_mcnemar:
        melhor = ROTULO_A if acur_a > acur_b else ROTULO_B
        print(
            f"McNemar exato rejeita H0 (p={mcnemar['p_value']:.4f} < {ALFA}): "
            f"a diferença de acurácia entre os dois experimentos é estatisticamente "
            f"significativa, favorecendo {melhor}."
        )
    else:
        print(
            f"McNemar exato NÃO rejeita H0 (p={mcnemar['p_value']:.4f} >= {ALFA}): "
            f"não há evidência estatística suficiente (n={n}) para afirmar que "
            f"{ROTULO_A} e {ROTULO_B} têm acurácias verdadeiramente diferentes -- "
            f"a diferença observada ({acur_a - acur_b:+.3f}) pode ser ruído de amostragem."
        )

    linhas_saida = [{
        "tag_a": ROTULO_EXP_A, "tag_b": ROTULO_EXP_B, "n": n,
        "acc_a": acur_a, "acc_b": acur_b, "diff": acur_a - acur_b,
        "naive_ci_low": ingenuo["ci_low"], "naive_ci_high": ingenuo["ci_high"],
        "bootstrap_ci_low": reamostras["ci_low"] if reamostras else None,
        "bootstrap_ci_high": reamostras["ci_high"] if reamostras else None,
        "mcnemar_p_value": mcnemar["p_value"],
        "mcnemar_significant": significativo_mcnemar,
    }]
    DIR_RELATORIOS.mkdir(parents=True, exist_ok=True)
    caminho_saida = DIR_RELATORIOS / f"compare_{ROTULO_EXP_A}_vs_{ROTULO_EXP_B}.csv"
    pd.DataFrame(linhas_saida).to_csv(caminho_saida, index=False)
    print(f"\nResumo salvo em {caminho_saida}")


if __name__ == "__main__":
    principal()
