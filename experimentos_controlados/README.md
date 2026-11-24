# Repetições controladas (10x)

Repete de forma **controlada e padronizada, com 10 repetições** (sementes
0 a 9), os treinos principais que em `experimentos/` têm execução única
(ou, no caso da classificação de polímero com VLM, 4 execuções avulsas).

## Escopo

Os 6 treinos principais e a avaliação fim a fim pareada. Validação
cruzada e o detector só com nanopartículas (`experimentos/` 43–52) ficam
fora desta fila.

| Script aqui | Espelha | O que é | Resultado agregado |
|---|---|---|---|
| `01_yolo26_main_10x.py` | `13_train_yolo26_corrected.py` | YOLO26L principal | precisão, recall, mAP50, mAP50:95 |
| `02_vlm_polymer_real_only_10x.py` | `05_finetune_real_only.py` | SmolVLM LoRA polímero, só real | acurácia de teste |
| `03_vlm_polymer_real_synth_10x.py` | `06_finetune_real_plus_synthetic.py` | SmolVLM LoRA polímero, real+sintético | acurácia de teste |
| `04_vlm_polymer_bigvlm_10x.py` | `09_finetune_bigger_vlm.py` | SmolVLM ~2B polímero | acurácia de teste |
| `05_smolvlm_size_lora_10x.py` | `17_size_finetune_lora.py` | SmolVLM LoRA regime de tamanho | acurácia de teste |
| `06_smolvlm_size_full_10x.py` | `18_size_finetune_full.py` | SmolVLM ajuste fino completo, regime de tamanho | acurácia de teste |
| `eval_and_pair.py` (`--pair`) | `20_predicted_box_and_end_to_end.py` | Fim a fim pareado (YOLO rep _i_ + SmolVLM rep _i_) | taxas D, P, S e E |

### Repetições corrigidas (valores finais)

Os scripts de avaliação originais do detector não passavam `max_det` ao
preditor do Ultralytics e usavam o padrão 300 em vez do `max_det=2000` do
treino; o regime de tamanho foi depois redefinido pela regra `eq1`
(diâmetro equivalente da área do polígono). As configurações corrigidas
são as usadas nos valores finais de `relatorio_final/estatisticas_agregadas.csv`:

| Configuração | O que é |
|---|---|
| `01_yolo26_main_maxdet2000` | YOLO26L reavaliado com `max_det=2000` (10 sementes em `experimentos/resultados/61_reeval_detector_maxdet/summary.json`) |
| `07_smolvlm_size_lora_eq1` | SmolVLM LoRA, regime de tamanho pela regra `eq1` |
| `08_smolvlm_size_full_eq1` | SmolVLM ajuste fino completo, regime de tamanho pela regra `eq1` |
| `09_end_to_end_paired_maxdet2000` | Fim a fim pareado com `max_det=2000` |

As entradas `01_yolo26_main`, `05_smolvlm_size_lora`,
`06_smolvlm_size_full` e `07_end_to_end_paired` são mantidas para
rastreabilidade e marcadas como SUPERADAS.

## O que varia entre as 10 repetições, o que fica fixo

- **Varia**: só a semente (0 a 9), explícita e registrada por repetição.
  Ela controla inicialização de pesos, dropout e ordem de embaralhamento
  (`ctrl_common.fixar_sementes`). Os treinos do YOLO26L são
  determinísticos bit a bit com semente fixa (Ultralytics); nos do SmolVLM
  a semente é aplicada também à RNG global do PyTorch, além do
  embaralhamento dos dados.
- **Fixo**: partição de dados (as partições de `experimentos/`, nunca
  regeradas), hiperparâmetros idênticos aos scripts originais e a mesma
  receita de treino (`vlm_common.ajuste_fino_lora` /
  `ajuste_fino_completo`, sem modificação).
- **Ambiente**: uma única GPU (RTX 4060 Ti), registrada por repetição
  (commit do git, versões de bibliotecas, GPU) em cada `summary.json`, mais
  um log contínuo do estado da máquina durante toda a fila (ver abaixo).

## Onde os arquivos ficam

- **Checkpoints/CSVs de predição do SmolVLM**: na pasta compartilhada
  (`experimentos/resultados/checkpoints/`,
  `experimentos/resultados/reports/`), com rótulos únicos
  `<rotulo>_ctrl_seedN`, a mesma convenção dos `_run2`/`_run3`/`_run4`,
  padronizada para 10 execuções com semente explícita.
  **Nada existente é sobrescrito.**
- **Checkpoints do YOLO26L**: isolados em `runs/01_yolo26_main/rep_00
  .. rep_09/` (nunca toca `experimentos/runs_v2/`).
- **Resumo por repetição**: `resultados/<experimento>/rep_00
  .. rep_09/summary.json` (`ctrl_common.salvar_resumo_rep`).
- **Log contínuo da máquina**: `logs_maquina/system_monitor_YYYYMMDD.csv`
  (amostra a cada 10 s: temperatura/potência/utilização/memória da GPU via
  pynvml, CPU/RAM via psutil, disco livre e a repetição em execução em cada
  instante) + `logs_maquina/session_manifest.json` (retrato único: SO,
  driver/CUDA, `pip freeze`, commit do git).
- **Relatório final**: `relatorio_final/`, com média, desvio-padrão e
  IC 95% (n=10, distribuição t) por métrica (`estatisticas_agregadas.csv`
  / `.json`), tempos de execução (`tempos_execucao.csv` / `.json`) e as
  comparações entre configurações (`comparacoes_ctrl_repetitions.csv` /
  `.json`, gerado por `compare_ctrl_repetitions.py`).

## Como rodar

Treinos longos em GPU: rodar no terminal e acompanhar, pelo menos nas
primeiras repetições.

```bash
cd experimentos_controlados/scripts

# 1. valida que tudo roda (só semente 0 de cada experimento) antes de
#    comprometer ~60 h de GPU:
python3 run_queue.py --dry-run

# 2. compare o resultado do dry-run com a execução única original em
#    experimentos/resultados/ (semente 0 deve bater bit a bit no YOLO e
#    ficar próxima no SmolVLM)

# 3. fila completa (retomável: pode interromper com Ctrl+C e rodar de
#    novo depois; retoma de onde parou):
python3 run_queue.py
```

`run_queue.py` inicia e mantém o `system_monitor.py` rodando em paralelo
durante toda a fila, roda os 6 experimentos na ordem (YOLO26L →
SmolVLM regime de tamanho LoRA/completo → SmolVLM polímero
só real/real+sintético/VLM maior), depois `eval_and_pair.py` (fim a fim
pareado) e por fim `final_report.py` (agregação estatística).

## Tempo estimado

~60 horas de GPU (~2,5 dias corridos) para a fila completa.

## Verificação recomendada

1. `run_queue.py --dry-run` roda sem erro.
2. A repetição com semente 0 de cada experimento coincide (YOLO, bit a
   bit) ou fica próxima (SmolVLM, mesma ordem de grandeza) do resultado de
   execução única correspondente em `experimentos/resultados/`.
3. A fila completa roda até o fim; `final_report.py` mostra 10/10
   repetições para cada experimento, com IC 95% plausíveis (não
   degenerados).
4. `logs_maquina/system_monitor_*.csv` não tem lacunas ao longo das ~60 h.
