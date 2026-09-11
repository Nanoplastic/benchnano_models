"""Experimento #3 do plano (plano_experimentos_vlm.md): pipeline
descrever-depois-classificar, zero-shot, sem treino nenhum -- portado de
`tarefa_vlm_nanoplastic_fewshot/scripts/07_describe_then_classify.py`
(Figueiredo & Melo 2025, UEA: separar "perceber" de "rotular" levou
SmolVLM de 20%->63.3% em outro domínio) para o dataset MiNa completo
(`real_crops/test`, n=208, 4 classes PE/PET/PP/PS), em vez das 7 imagens
Moon et al. onde essa técnica já foi testada e não ajudou (42.9%, mesmo
teto de sempre -- mas lá o problema era percepção de FORMA em n=7).

Diferença importante em relação à linha anterior: lá as classes eram
categorias de FORMA definidas no próprio artigo-fonte (Nylon=fibra,
PS=flake, PET=ball-stick), então a definição pro estágio 2 vinha direto da
literatura Moon et al. Aqui as classes são identidade QUÍMICA de polímero
-- não há uma forma canônica única por classe no MiNa (a matriz de
confusão do Experimento A mostra PP e PET frequentemente confundidos por
serem "morfologicamente parecidos": fragmentos irregulares densos sobre a
mesma textura de membrana). As dicas de morfologia abaixo são heurísticas
genéricas da literatura de caracterização de microplástico (aparência
típica de cada polímero sob SEM), NÃO validadas contra as imagens do MiNa
especificamente -- essa é uma limitação a declarar no resultado, não algo
resolvido.
"""
import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_common import DEVICE, REPORTS_DIR, ROOT, extract_label, load_base_model, load_real_split

REASONER_MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

DESCRIBE_PROMPT = (
    "Describe the physical shape, edges, and surface texture of the "
    "particle(s) visible in this scanning electron microscope image, in "
    "2-3 sentences. Do not name any material or polymer -- only describe "
    "what you see."
)

CLASSIFY_FROM_DESCRIPTION_TEMPLATE = (
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
def describe_image(processor, vlm, image_path) -> str:
    vlm.eval()
    image = Image.open(image_path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": DESCRIBE_PROMPT}]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE)
    output_ids = vlm.generate(**inputs, max_new_tokens=80, do_sample=False)
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def load_reasoner():
    tokenizer = AutoTokenizer.from_pretrained(REASONER_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(REASONER_MODEL_ID, torch_dtype=torch.bfloat16).to(DEVICE)
    model.eval()
    return tokenizer, model


@torch.no_grad()
def classify_from_description(tokenizer, reasoner, description: str) -> str:
    prompt_text = CLASSIFY_FROM_DESCRIPTION_TEMPLATE.format(description=description)
    messages = [{"role": "user", "content": prompt_text}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    output_ids = reasoner.generate(**inputs, max_new_tokens=30, do_sample=False)
    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


def main():
    import pandas as pd

    real_df = load_real_split("test")
    print(f"MiNa real_crops/test (n={len(real_df)})\n")

    print("Estágio 1: descrevendo partículas com SmolVLM-500M-Instruct (sem nomes de classe no prompt)...")
    processor, vlm = load_base_model()
    descriptions = []
    for i, (_, r) in enumerate(real_df.iterrows()):
        desc = describe_image(processor, vlm, ROOT / r["image_path"])
        descriptions.append(desc)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(real_df)} descritas...")

    del vlm
    torch.cuda.empty_cache()

    print(f"\nEstágio 2: classificando a partir das descrições com {REASONER_MODEL_ID}...")
    tokenizer, reasoner = load_reasoner()
    rows = []
    for i, ((_, r), desc) in enumerate(zip(real_df.iterrows(), descriptions)):
        raw = classify_from_description(tokenizer, reasoner, desc)
        pred = extract_label(raw)
        rows.append({
            "image_path": r["image_path"],
            "true_label": r["label"],
            "description": desc,
            "raw_answer": raw,
            "pred_label": pred,
            "correct": pred == r["label"],
        })
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(real_df)} classificadas...")

    out_df = pd.DataFrame(rows)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(REPORTS_DIR / "describe_then_classify_mina_predictions.csv", index=False)

    acc = out_df["correct"].mean()
    print(f"\nAcurácia describe-then-classify no MiNa real test (n={len(out_df)}): "
          f"{acc:.3f} ({out_df['correct'].sum()}/{len(out_df)})")
    print("\nMatriz de confusão (linha=real, coluna=previsto):")
    print(pd.crosstab(out_df["true_label"], out_df["pred_label"]))
    print("\nRecall por classe:")
    print(out_df.groupby("true_label")["correct"].agg(["sum", "count"]))


if __name__ == "__main__":
    main()
