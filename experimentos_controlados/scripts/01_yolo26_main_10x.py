"""10 repetições controladas do YOLO26L / 10 controlled YOLO26L repetitions.

PT: 10 repetições controladas do treino do YOLO26L principal, espelhando
    experimentos/scripts/13_train_yolo26_corrected.py (mesmos hiperparâmetros, dataset,
    partição e checkpoint pré-treinado), variando só a semente do Ultralytics (0..9,
    explícita). Cada repetição treina do zero a partir de yolo26l.pt.
    Saída isolada em experimentos_controlados/ (nunca sobrescreve
    experimentos/runs_v2/microplastic_yolo26_v2/):
    runs/01_yolo26_main/rep_00/weights/best.pt ... rep_09/
    resultados/01_yolo26_main/rep_00/summary.json ... rep_09/
    Após cada treino, avalia na mesma partição de teste (datasets_yolo26_v2/images/test, 15
    imagens) com as métricas agregadas de 14_yolo_test_eval.py
    (precisão/recall/mAP50/mAP50:95). A repetição com semente 0 deve reproduzir bit a bit o
    resultado original (60,7%/46,7%/28,0%/8,6%), já que o Ultralytics é determinístico com
    semente fixa.
    Rodar (treino longo em GPU; acompanhar no terminal):
    python3 01_yolo26_main_10x.py [--only-seed N]
EN: 10 controlled repetitions of the main YOLO26L training, mirroring
    experimentos/scripts/13_train_yolo26_corrected.py (same hyperparameters, dataset, split
    and pretrained checkpoint), varying only the Ultralytics seed (0..9, explicit). Each
    repetition trains from scratch from yolo26l.pt.
    Output isolated in experimentos_controlados/ (never overwrites
    experimentos/runs_v2/microplastic_yolo26_v2/):
    runs/01_yolo26_main/rep_00/weights/best.pt ... rep_09/
    resultados/01_yolo26_main/rep_00/summary.json ... rep_09/
    After each training run, evaluates on the same test split
    (datasets_yolo26_v2/images/test, 15 images) with the aggregate metrics of
    14_yolo_test_eval.py (precision/recall/mAP50/mAP50:95). The seed 0 repetition should
    reproduce the original result bit for bit (60.7%/46.7%/28.0%/8.6%), since Ultralytics is
    deterministic with a fixed seed.
    Run (long GPU job; follow it in the terminal):
    python3 01_yolo26_main_10x.py [--only-seed N]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ctrl_common as cc
from eval_and_pair import avaliar_yolo_conjunto_teste

from ultralytics import YOLO

EXPERIMENTO = "01_yolo26_main"
RAIZ_PROJETO_ORIGEM = cc.RAIZ_PROJETO_ORIGEM
YAML_DADOS = str(RAIZ_PROJETO_ORIGEM / "experimentos" / "datasets_yolo26_v2" / "data.yaml")
PESOS_BASE = str(RAIZ_PROJETO_ORIGEM / "experimentos" / "yolo26l.pt")
DIR_PROJETO = str(cc.DIR_EXECUCOES / EXPERIMENTO)


def kwargs_treino_para(semente: int, nome_execucao: str) -> dict:
    # PT: IDÊNTICO a 13_train_yolo26_corrected.py; mudam só seed/project/name.
    # EN: IDENTICAL to 13_train_yolo26_corrected.py; only seed/project/name change.
    return dict(
        data=YAML_DADOS,
        task="detect",
        epochs=400,
        imgsz=1024,
        batch=-1,
        optimizer="AdamW",
        lr0=1e-4,
        lrf=0.1,
        cos_lr=True,
        seed=semente,
        device="0",
        project=DIR_PROJETO,
        name=nome_execucao,
        save_period=-1,
        patience=100,
        close_mosaic=15,
        max_det=2000,
        degrees=180.0,
        translate=0.1,
        scale=0.2,
        flipud=0.5,
        fliplr=0.5,
        perspective=0.0,
        hsv_h=0.0,
        hsv_s=0.0,
        hsv_v=0.25,
        mosaic=0.3,
        mixup=0.0,
        erasing=0.0,
        plots=True,
        verbose=True,
        exist_ok=True,
    )


def executar_repeticao(semente: int, estado: dict):
    if cc.rep_concluida(estado, EXPERIMENTO, semente):
        print(f"[{EXPERIMENTO}] rep {semente:02d} já concluída, pulando.")
        return

    nome_execucao = f"rep_{semente:02d}"
    print(f"\n=== {EXPERIMENTO} — repetição {semente:02d}/9 (seed={semente}) ===")
    cc.definir_tarefa_atual(EXPERIMENTO, semente)
    cc.fixar_sementes(semente)

    modelo = YOLO(PESOS_BASE, task="detect")
    kwargs = kwargs_treino_para(semente, nome_execucao)
    modelo.train(**kwargs)

    melhor_pt = Path(DIR_PROJETO) / nome_execucao / "weights" / "best.pt"
    print(f"[{EXPERIMENTO}] treino concluído: {melhor_pt}")

    metricas = avaliar_yolo_conjunto_teste(melhor_pt)
    print(f"[{EXPERIMENTO}] avaliação rep {semente:02d}: {metricas['aggregate_metrics']}")

    cc.salvar_resumo_rep(EXPERIMENTO, semente, semente, metricas, metadados_extras={
        "best_weights": str(melhor_pt),
        "train_kwargs": kwargs,
    })
    cc.marcar_rep_concluida(estado, EXPERIMENTO, semente)
    cc.limpar_tarefa_atual()


def principal():
    analisador = argparse.ArgumentParser(description=__doc__)
    analisador.add_argument("--only-seed", type=int, default=None,
                         help="roda só uma repetição específica (0-9), útil pro --dry-run da fila")
    args = analisador.parse_args()

    estado = cc.carregar_estado_fila()
    sementes = [args.only_seed] if args.only_seed is not None else cc.SEMENTES
    for semente in sementes:
        executar_repeticao(semente, estado)


if __name__ == "__main__":
    principal()
