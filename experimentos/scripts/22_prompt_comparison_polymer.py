"""Comparação de prompts para polímero / Prompt comparison for polymer.

PT: Compara variações de prompt para o SmolVLM zero-shot na classificação de polímero
    (PE/PET/PP/PS), com a mesma metodologia de 21_prompt_comparison.py (seleção só na
    validação; IC por bootstrap e McNemar exato contra o prompt-base dos scripts
    04/05/06/08/09).
    Cinco variações:
    P1 base -- PROMPT_CLASSIFICACAO usado em 04/05/06/09, sem mudança.
    P2 morfologia -- inclui as dicas heurísticas de morfologia por polímero de
    08_describe_then_classify.py (aparência típica de cada plástico em MEV), sem o pipeline
    de dois estágios.
    P3 mínimo -- versão enxuta.
    P4 raciocínio -- pede uma frase de raciocínio sobre forma/textura antes da resposta
    final (chain-of-thought).
    P5 eliminação -- pede para descartar candidatos antes de decidir, testando se o
    enquadramento por eliminação ajuda em casos ambíguos como PP vs PET.
EN: Compares prompt variants for zero-shot SmolVLM on polymer classification (PE/PET/PP/PS),
    with the same methodology as 21_prompt_comparison.py (selection on validation only;
    bootstrap CI and exact McNemar against the base prompt of scripts 04/05/06/08/09).
    Five variants:
    P1 base -- PROMPT_CLASSIFICACAO used in 04/05/06/09, unchanged.
    P2 morphology -- includes the per-polymer morphology heuristics from
    08_describe_then_classify.py (typical SEM appearance of each plastic), without the
    two-stage pipeline.
    P3 minimal -- a lean version.
    P4 reasoning -- asks for one sentence of shape/texture reasoning before the final answer
    (chain-of-thought).
    P5 elimination -- asks to rule out candidates before deciding, testing whether an
    elimination framing helps in ambiguous cases such as PP vs PET.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import (
    ROTULOS, carregar_modelo_base, carregar_particao_real, avaliar_classificacao,
    salvar_resumo_experimento, configurar_log_experimento,
)

NOME_EXP = "22_prompt_comparison_polymer"
DIR_EXP, CAMINHO_LOG, REGISTRADOR, METADADOS_EXECUCAO = configurar_log_experimento(NOME_EXP, __file__)

DIR_RELATORIOS = DIR_EXP / "reports"
PREDICOES_TESTE_BASE = Path(__file__).resolve().parent.parent / "resultados" / "reports" / "zeroshot_real_test_predictions.csv"

ALFA = 0.05
N_BOOTSTRAP = 10_000
SEMENTE = 42

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
    df_val = carregar_particao_real("val")
    df_teste = carregar_particao_real("test")
    from vlm_common import MANIFESTO_REAL
    manifesto_completo = pd.read_csv(MANIFESTO_REAL)

    REGISTRADOR.info("=== selecao de prompt (polimero) na VALIDACAO (n=%s), zero-shot ===", len(df_val))
    print(f"=== Etapa 1: selecao de prompt na validacao (n={len(df_val)}), zero-shot ===\n")

    processador, modelo = carregar_modelo_base()

    resultados_val = {}
    for nome, prompt in PROMPTS.items():
        _, acur = avaliar_classificacao(
            processador, modelo, df_val, f"prompt_poly_{nome}_val",
            dir_saida=DIR_RELATORIOS, prompt=prompt, rotulos=ROTULOS,
        )
        resultados_val[nome] = acur
        print(f"{nome:15s} acc_val={acur:.3f}")
        REGISTRADOR.info("prompt=%s acc_val=%.4f", nome, acur)

    vencedor = max(resultados_val, key=resultados_val.get)
    print(f"\nPrompt vencedor na validacao: {vencedor} (acc_val={resultados_val[vencedor]:.3f})")
    REGISTRADOR.info("prompt_vencedor=%s acc_val=%.4f", vencedor, resultados_val[vencedor])

    print(f"\n=== Etapa 2: avaliacao UNICA do prompt vencedor no teste travado (n={len(df_teste)}) ===\n")
    resultados_vencedor, acur_teste_vencedor = avaliar_classificacao(
        processador, modelo, df_teste, f"prompt_poly_{vencedor}_test",
        dir_saida=DIR_RELATORIOS, prompt=PROMPTS[vencedor], rotulos=ROTULOS,
    )
    print(f"Acuracia do prompt vencedor ({vencedor}) no teste: {acur_teste_vencedor:.3f}")
    print("Matriz de confusao:")
    print(pd.crosstab(resultados_vencedor["true_label"], resultados_vencedor["pred_label"]))
    print("\nRecall por classe:")
    print(resultados_vencedor.groupby("true_label")["correct"].mean())

    lo, hi, n_imgs = ic_bootstrap(resultados_vencedor, manifesto_completo)
    print(f"\nIC 95% (bootstrap por imagem, {N_BOOTSTRAP} replicas, {n_imgs} imagens): [{lo:.3f}, {hi:.3f}]")

    print(f"\n=== Etapa 3: comparacao pareada contra o prompt-baseline (P1, ja usado em 04/05/06/09) ===\n")
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
    pd.DataFrame(linhas_saida).to_csv(DIR_RELATORIOS / "prompt_comparison_polymer_val_accuracies.csv", index=False)

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
