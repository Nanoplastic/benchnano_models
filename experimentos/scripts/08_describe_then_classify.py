"""Descrever e depois classificar (zero-shot) / Describe-then-classify (zero-shot).

PT: Pipeline de dois estágios sem treino: o VLM primeiro descreve a partícula em texto e um
    segundo modelo de linguagem classifica o polímero a partir dessa descrição. Avaliado em
    `real_crops/test` (n=208, 4 classes PE/PET/PP/PS).
    As classes do MiNa são identidade química do polímero, sem uma forma canônica por classe
    (PP e PET aparecem como fragmentos irregulares densos sobre a mesma textura de
    membrana). As dicas de morfologia usadas no prompt são heurísticas genéricas de
    caracterização de microplásticos em MEV e não foram validadas especificamente contra as
    imagens do MiNa.
EN: Two-stage pipeline with no training: the VLM first describes the particle in text and a
    second language model classifies the polymer from that description. Evaluated on
    `real_crops/test` (n=208, 4 classes PE/PET/PP/PS).
    MiNa classes are the chemical identity of the polymer, with no canonical shape per class
    (PP and PET appear as dense irregular fragments over the same membrane texture). The
    morphology hints used in the prompt are generic SEM microplastic characterization
    heuristics and were not validated specifically against MiNa images.
"""
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import DISPOSITIVO, DIR_RELATORIOS, RAIZ, extrair_rotulo, carregar_modelo_base, carregar_particao_real

ID_MODELO_RACIOCINIO = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

PROMPT_DESCRICAO = (
    "Describe the physical shape, edges, and surface texture of the "
    "particle(s) visible in this scanning electron microscope image, in "
    "2-3 sentences. Do not name any material or polymer -- only describe "
    "what you see."
)

MODELO_CLASSIFICAR_POR_DESCRICAO = (
    "Considering that PE (polyethylene) microplastic fragments typically "
    "appear as irregular, waxy, folded flakes with smooth rounded edges. "
    "Considering that PET (polyethylene terephthalate) fragments typically "
    "appear as angular, glassy fragments with sharp edges, sometimes "
    "fibrous strands. Considering that PP (polypropylene) fragments "
    "typically appear as irregular blocky fragments with a rougher, more "
    "granular surface texture. Considering that PS (polystyrene) particles "
    "typically appear as smooth spherical beads or brittle angular "
    "fragments with a glassy surface.\n\n"
    'Here is a description of particle(s) seen under SEM: "{description}"\n\n'
    "Based only on this description, which of these four best matches: PE, "
    "PET, PP, or PS? Answer with just one label: PE, PET, PP, or PS."
)


@torch.no_grad()
def descrever_imagem(processador, vlm, caminho_imagem) -> str:
    vlm.eval()
    imagem = Image.open(caminho_imagem).convert("RGB")
    mensagens = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT_DESCRICAO}]}]
    prompt = processador.apply_chat_template(mensagens, add_generation_prompt=True)
    entradas = processador(text=prompt, images=[imagem], return_tensors="pt").to(DISPOSITIVO)
    ids_saida = vlm.generate(**entradas, max_new_tokens=80, do_sample=False)
    gerado = ids_saida[:, entradas["input_ids"].shape[1]:]
    return processador.batch_decode(gerado, skip_special_tokens=True)[0].strip()


def carregar_modelo_raciocinio():
    tokenizador = AutoTokenizer.from_pretrained(ID_MODELO_RACIOCINIO)
    modelo = AutoModelForCausalLM.from_pretrained(ID_MODELO_RACIOCINIO, torch_dtype=torch.bfloat16).to(DISPOSITIVO)
    modelo.eval()
    return tokenizador, modelo


@torch.no_grad()
def classificar_por_descricao(tokenizador, modelo_raciocinio, descricao: str) -> str:
    texto_prompt = MODELO_CLASSIFICAR_POR_DESCRICAO.format(description=descricao)
    mensagens = [{"role": "user", "content": texto_prompt}]
    prompt = tokenizador.apply_chat_template(mensagens, add_generation_prompt=True, tokenize=False)
    entradas = tokenizador(prompt, return_tensors="pt").to(DISPOSITIVO)
    ids_saida = modelo_raciocinio.generate(**entradas, max_new_tokens=30, do_sample=False)
    gerado = ids_saida[:, entradas["input_ids"].shape[1]:]
    return tokenizador.batch_decode(gerado, skip_special_tokens=True)[0].strip()


def principal():
    import pandas as pd

    df_real = carregar_particao_real("test")
    print(f"MiNa real_crops/test (n={len(df_real)})\n")

    print("Estágio 1: descrevendo partículas com SmolVLM-500M-Instruct (sem nomes de classe no prompt)...")
    processador, vlm = carregar_modelo_base()
    descricoes = []
    for i, (_, r) in enumerate(df_real.iterrows()):
        descr = descrever_imagem(processador, vlm, RAIZ / r["image_path"])
        descricoes.append(descr)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(df_real)} descritas...")

    del vlm
    torch.cuda.empty_cache()

    print(f"\nEstágio 2: classificando a partir das descrições com {ID_MODELO_RACIOCINIO}...")
    tokenizador, modelo_raciocinio = carregar_modelo_raciocinio()
    linhas_tab = []
    for i, ((_, r), descr) in enumerate(zip(df_real.iterrows(), descricoes)):
        bruto = classificar_por_descricao(tokenizador, modelo_raciocinio, descr)
        pred = extrair_rotulo(bruto)
        linhas_tab.append({
            "image_path": r["image_path"],
            "true_label": r["label"],
            "description": descr,
            "raw_answer": bruto,
            "pred_label": pred,
            "correct": pred == r["label"],
        })
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(df_real)} classificadas...")

    df_saida = pd.DataFrame(linhas_tab)
    DIR_RELATORIOS.mkdir(parents=True, exist_ok=True)
    df_saida.to_csv(DIR_RELATORIOS / "describe_then_classify_mina_predictions.csv", index=False)

    acur = df_saida["correct"].mean()
    print(f"\nAcurácia describe-then-classify no MiNa real test (n={len(df_saida)}): "
          f"{acur:.3f} ({df_saida['correct'].sum()}/{len(df_saida)})")
    print("\nMatriz de confusão (linha=real, coluna=previsto):")
    print(pd.crosstab(df_saida["true_label"], df_saida["pred_label"]))
    print("\nRecall por classe:")
    print(df_saida.groupby("true_label")["correct"].agg(["sum", "count"]))


if __name__ == "__main__":
    principal()
