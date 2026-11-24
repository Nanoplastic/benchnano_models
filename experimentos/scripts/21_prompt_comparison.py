"""Comparação de prompts para regime de tamanho / Prompt comparison for size regime.

PT: Compara variações de prompt para o SmolVLM zero-shot na classificação de regime de
    tamanho. O prompt-base é PROMPT_CLASSIFICACAO_TAMANHO de vlm_common.py.
    Metodologia:
    - Seleção só no conjunto de validação; o teste é
    usado uma única vez, depois que o prompt vencedor está decidido.
    - Acurácia com IC 95% por bootstrap
    (imagem de origem) para cada prompt e McNemar exato pareado entre o vencedor e o
    prompt-base, no mesmo conjunto de teste.
    Cinco variações, cada uma testando um eixo de engenharia de prompt para VLM:
    P1 base -- o prompt usado em 16/17/18/20, sem mudança.
    P2 escala -- pede que o modelo use a barra de escala/ampliação visível na imagem antes
    de decidir (a escala física é necessária, não só o tamanho em pixels).
    P3 mínimo -- versão enxuta; testa se elaboração demais atrapalha.
    P4 raciocínio -- pede um raciocínio breve antes da resposta final (chain-of-thought).
    P5 conversão -- reforça a conversão de unidade (1 micrômetro = 1000 nanômetros) e pede
    comparação direta com a barra de escala impressa.
EN: Compares prompt variants for zero-shot SmolVLM on size regime classification. The base
    prompt is PROMPT_CLASSIFICACAO_TAMANHO from vlm_common.py.
    Methodology:
    - Selection on the validation set only; the test
    set is used once, after the winning prompt is chosen.
    - Accuracy with a 95% bootstrap CI
    (source image) for each prompt and paired exact McNemar between the winner and the base
    prompt, on the same test set.
    Five variants, each testing one VLM prompt engineering axis:
    P1 base -- the prompt used in 16/17/18/20, unchanged.
    P2 scale -- asks the model to use the scale bar/magnification shown in the image before
    deciding (physical scale is needed, not just pixel size).
    P3 minimal -- a lean version; tests whether too much elaboration hurts.
    P4 reasoning -- asks for brief reasoning before the final answer (chain-of-thought).
    P5 conversion -- stresses the unit conversion (1 micrometre = 1000 nanometres) and asks
    for a direct comparison against the printed scale bar.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import (
    RAIZ_RESULTADOS, ROTULOS_TAMANHO, carregar_modelo_base, avaliar_classificacao,
    salvar_resumo_experimento, configurar_log_experimento,
)

NOME_EXP = "21_prompt_comparison"
DIR_EXP, CAMINHO_LOG, REGISTRADOR, METADADOS_EXECUCAO = configurar_log_experimento(NOME_EXP, __file__)

MANIFESTO = RAIZ_RESULTADOS / "15_build_size_regime_manifest" / "data" / "manifest_size_regime.csv"
DIR_RELATORIOS = DIR_EXP / "reports"
PREDICOES_TESTE_BASE = RAIZ_RESULTADOS / "16_size_zeroshot_eval" / "reports" / "size_zeroshot_real_test_predictions.csv"

ALFA = 0.05
N_BOOTSTRAP = 10_000
SEMENTE = 42

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


def ic_bootstrap(df: pd.DataFrame, manifesto: pd.DataFrame, n_bootstrap=N_BOOTSTRAP, alfa=ALFA, semente=SEMENTE):
    mesclado = df.merge(manifesto[["image_path", "source_image"]], on="image_path", how="left")
    gerador = np.random.default_rng(semente)
    imgs = mesclado["source_image"].unique()
    grupos = {s: g for s, g in mesclado.groupby("source_image")}
    acurs = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        amostrados = gerador.choice(imgs, size=len(imgs), replace=True)
        rep = pd.concat([grupos[s] for s in amostrados], ignore_index=True)
        acurs[i] = rep["correct"].mean()
    lo, hi = np.percentile(acurs, [100 * alfa / 2, 100 * (1 - alfa / 2)])
    return lo, hi, len(imgs)


def mcnemar_exato(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict:
    mesclado = df_a.merge(df_b, on="image_path", suffixes=("_a", "_b"), validate="one_to_one")
    so_a = int(((mesclado["correct_a"]) & (~mesclado["correct_b"])).sum())
    so_b = int(((~mesclado["correct_a"]) & (mesclado["correct_b"])).sum())
    n_discordantes = so_a + so_b
    p = 1.0 if n_discordantes == 0 else stats.binomtest(
        min(so_a, so_b), n_discordantes, p=0.5, alternative="two-sided"
    ).pvalue
    return {"a_only": so_a, "b_only": so_b, "n_discordant": n_discordantes, "p_value": p}


def principal():
    if not MANIFESTO.exists():
        raise FileNotFoundError(f"{MANIFESTO} não existe -- rode 15_build_size_regime_manifest.py primeiro.")

    df_manifesto = pd.read_csv(MANIFESTO)
    df_val = df_manifesto[df_manifesto["split"] == "val"].reset_index(drop=True)
    df_teste = df_manifesto[df_manifesto["split"] == "test"].reset_index(drop=True)

    REGISTRADOR.info("=== selecao de prompt na VALIDACAO (n=%s), zero-shot ===", len(df_val))
    print(f"=== Etapa 1: selecao de prompt na validacao (n={len(df_val)}), zero-shot ===\n")

    processador, modelo = carregar_modelo_base()

    resultados_val = {}
    for nome, prompt in PROMPTS.items():
        _, acur = avaliar_classificacao(
            processador, modelo, df_val, f"prompt_{nome}_val",
            dir_saida=DIR_RELATORIOS, prompt=prompt, rotulos=ROTULOS_TAMANHO,
        )
        resultados_val[nome] = acur
        print(f"{nome:15s} acc_val={acur:.3f}")
        REGISTRADOR.info("prompt=%s acc_val=%.4f", nome, acur)

    vencedor = max(resultados_val, key=resultados_val.get)
    print(f"\nPrompt vencedor na validacao: {vencedor} (acc_val={resultados_val[vencedor]:.3f})")
    REGISTRADOR.info("prompt_vencedor=%s acc_val=%.4f", vencedor, resultados_val[vencedor])

    print(f"\n=== Etapa 2: avaliacao UNICA do prompt vencedor no teste travado (n={len(df_teste)}) ===\n")
    resultados_vencedor, acur_teste_vencedor = avaliar_classificacao(
        processador, modelo, df_teste, f"prompt_{vencedor}_test",
        dir_saida=DIR_RELATORIOS, prompt=PROMPTS[vencedor], rotulos=ROTULOS_TAMANHO,
    )
    print(f"Acuracia do prompt vencedor ({vencedor}) no teste: {acur_teste_vencedor:.3f}")
    print("Matriz de confusao:")
    print(pd.crosstab(resultados_vencedor["true_label"], resultados_vencedor["pred_label"]))

    lo, hi, n_imgs = ic_bootstrap(resultados_vencedor, df_manifesto)
    print(f"IC 95% (bootstrap por imagem, {N_BOOTSTRAP} replicas, {n_imgs} imagens): [{lo:.3f}, {hi:.3f}]")

    print(f"\n=== Etapa 3: comparacao pareada contra o prompt-baseline (P1, ja usado em 16/17/18/20) ===\n")
    if not PREDICOES_TESTE_BASE.exists():
        print(f"AVISO: {PREDICOES_TESTE_BASE} nao encontrado -- pulando comparacao pareada.")
        resultado_mcnemar = None
    else:
        df_base = pd.read_csv(PREDICOES_TESTE_BASE)
        df_base["correct"] = df_base["correct"].astype(bool)
        resultados_vencedor["correct"] = resultados_vencedor["correct"].astype(bool)
        resultado_mcnemar = mcnemar_exato(
            df_base[["image_path", "correct"]], resultados_vencedor[["image_path", "correct"]]
        )
        acur_base = df_base["correct"].mean()
        significativo = resultado_mcnemar["p_value"] < ALFA
        print(f"Acuracia baseline (P1) no teste: {acur_base:.3f}")
        print(f"Acuracia vencedor ({vencedor}) no teste: {acur_teste_vencedor:.3f}")
        print(f"McNemar exato: baseline_certo/vencedor_errado={resultado_mcnemar['a_only']}  "
              f"vencedor_certo/baseline_errado={resultado_mcnemar['b_only']}  "
              f"discordantes={resultado_mcnemar['n_discordant']}  p={resultado_mcnemar['p_value']:.4f}")
        print(f"Diferenca estatisticamente significativa (alpha=0.05)? {'SIM' if significativo else 'nao'}")

    linhas_saida = [{"prompt": nome, "acc_val": acur} for nome, acur in resultados_val.items()]
    pd.DataFrame(linhas_saida).to_csv(DIR_RELATORIOS / "prompt_comparison_val_accuracies.csv", index=False)

    salvar_resumo_experimento(NOME_EXP, METADADOS_EXECUCAO, {
        "val_accuracies_by_prompt": resultados_val,
        "winner_prompt": vencedor,
        "winner_test_acc": float(acur_teste_vencedor),
        "winner_test_ci95": [float(lo), float(hi)],
        "mcnemar_vs_baseline": resultado_mcnemar,
        "prompts": PROMPTS,
    })
    print(f"\nLog completo: {CAMINHO_LOG}")


if __name__ == "__main__":
    principal()
