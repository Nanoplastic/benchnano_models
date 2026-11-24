"""Ajuste fino LoRA de tamanho pela regra eq1 / Size LoRA fine-tuning with the eq1 rule.

PT: Mesmo treino de 17_size_finetune_lora.py, trocando só o manifesto: usa
    manifest_size_regime_eq1.csv (regime pelo diâmetro equivalente da área do polígono COCO,
    script 57) em vez de manifest_size_regime.csv (regime pela caixa, regra eq2).
    É um novo ajuste fino de execução única (zero-shot/LoRA/completo), não uma
    reclassificação: reclassificar as predições do modelo treinado com a regra eq2 contra o
    novo gabarito, sem retreinar, subestimaria o modelo, que nunca viu a fronteira correta
    no treino. As 10 repetições controladas com a regra eq1 estão em
    experimentos_controlados (rótulo *_eq1_ctrl_seed*). Receita idêntica à de 17 (r=8,
    alfa=8, 6 épocas, semente 42, mesmas partições train/val/test); só muda o manifesto de
    origem.
    Não sobrescreve o adaptador de 17 (rótulo size_lora_eq1_single em vez de size_lora); a
    execução com a regra eq2 continua em resultados/checkpoints/size_lora e
    resultados/reports/size_lora_real_test_predictions.csv.
    Uso:
    python3 experimentos/scripts/80_size_finetune_lora_eq1_single.py
EN: Same training as 17_size_finetune_lora.py, changing only the manifest: uses
    manifest_size_regime_eq1.csv (regime from the area-equivalent diameter of the COCO
    polygon, script 57) instead of manifest_size_regime.csv (regime from the box, eq2 rule).
    It is a new single-run fine-tuning (zero-shot/LoRA/full), not a relabelling: relabelling
    the predictions of the eq2-trained model against the new ground truth without retraining
    would underestimate the model, which never saw the correct boundary during training. The
    10 controlled repetitions with the eq1 rule are in experimentos_controlados (tag
    *_eq1_ctrl_seed*). Recipe identical to 17 (r=8, alpha=8, 6 epochs, seed 42, same
    train/val/test splits); only the source manifest changes.
    Does not overwrite the adapter of 17 (tag size_lora_eq1_single instead of size_lora);
    the eq2-rule run remains in resultados/checkpoints/size_lora and
    resultados/reports/size_lora_real_test_predictions.csv.
    Usage:
    python3 experimentos/scripts/80_size_finetune_lora_eq1_single.py
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import (
    RAIZ_RESULTADOS, PROMPT_CLASSIFICACAO_TAMANHO, ROTULOS_TAMANHO, ajuste_fino_lora,
    salvar_resumo_experimento, configurar_log_experimento,
)

ROTULO_EXP = "size_lora_eq1_single"
NOME_EXP = "80_size_finetune_lora_eq1_single"
DIR_EXP, CAMINHO_LOG, REGISTRADOR, METADADOS_EXECUCAO = configurar_log_experimento(NOME_EXP, __file__, {"tag": ROTULO_EXP})

MANIFESTO = RAIZ_RESULTADOS / "57_build_size_regime_manifest_eq1" / "data" / "manifest_size_regime_eq1.csv"
TAMANHOS_PARTICAO_ESPERADOS = {"train": 1000, "val": 240, "test": 208}  # PT: mesmas contagens de 15/17 (regra eq2) / EN: same counts as 15/17 (eq2 rule)


def principal():
    if not MANIFESTO.exists():
        raise FileNotFoundError(f"{MANIFESTO} não existe -- rode 57_build_size_regime_manifest_eq1.py primeiro.")

    df = pd.read_csv(MANIFESTO)
    df_treino = df[df["split"] == "train"].reset_index(drop=True)
    df_val = df[df["split"] == "val"].reset_index(drop=True)
    df_teste = df[df["split"] == "test"].reset_index(drop=True)

    for nome, sub, esperado in [("train", df_treino, TAMANHOS_PARTICAO_ESPERADOS["train"]),
                                 ("val", df_val, TAMANHOS_PARTICAO_ESPERADOS["val"]),
                                 ("test", df_teste, TAMANHOS_PARTICAO_ESPERADOS["test"])]:
        if len(sub) != esperado:
            raise SystemExit(f"sanidade falhou: split '{nome}' tem {len(sub)} linhas, esperado {esperado} "
                              f"(mesma contagem da corrida Eq.2 original) -- abortando antes de treinar.")

    print(f"LoRA regime de tamanho (Eq.1, corrida unica): train={len(df_treino)}  val={len(df_val)}  test={len(df_teste)}")
    print(df_treino["label"].value_counts())
    REGISTRADOR.info("train=%s val=%s test=%s classes=%s", len(df_treino), len(df_val), len(df_teste),
                df_treino["label"].value_counts().to_dict())

    resultado = ajuste_fino_lora(df_treino, df_val, df_teste, rotulo_exp=ROTULO_EXP,
                            prompt=PROMPT_CLASSIFICACAO_TAMANHO, rotulos=ROTULOS_TAMANHO)

    resumo = {
        "tag": ROTULO_EXP,
        "manifest": str(MANIFESTO),
        "train_samples": len(df_treino),
        "val_samples": len(df_val),
        "test_samples": len(df_teste),
        "best_val_acc": float(resultado["best_val_acc"]),
        "test_acc": float(resultado["test_acc"]),
    }
    salvar_resumo_experimento(NOME_EXP, METADADOS_EXECUCAO, resumo)
    REGISTRADOR.info("Resumo LoRA regime de tamanho (Eq.1, corrida unica): best_val_acc=%.4f test_acc=%.4f",
                resultado["best_val_acc"], resultado["test_acc"])


if __name__ == "__main__":
    principal()
