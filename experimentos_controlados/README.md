# Repetições controladas (10x) — só experimentos citados no artigo

Pedido do orientador: repetir os experimentos que sustentam os números do
artigo (`legado/Paper_2___Modelos/main_V5.tex`) de forma **controlada e
padronizada, com 10 repetições**, em vez das execuções únicas atuais (ou,
no caso do estudo-piloto de VLM, das 4 execuções ad hoc que já existiam).

Ver o plano completo (contexto, decisões de design, estimativas de tempo)
em `/home/mi/.claude/plans/parallel-giggling-ocean.md`.

## Escopo

Só os 6 treinos que aparecem no artigo — nada de CV, SAHI tiled ou
nano-only (todos marcados "pendente"/exploratório no
`experimentos/README.md`, fora desta rodada):

| Script aqui | Espelha | O que é | Onde aparece no artigo |
|---|---|---|---|
| `01_yolo26_main_10x.py` | `13_train_yolo26_corrected.py` | YOLO26L principal | Tabela `yolo_results` |
| `02_vlm_polymer_real_only_10x.py` | `05_finetune_real_only.py` | SmolVLM LoRA polímero, só real | Tabela `vlm_pilot_results` |
| `03_vlm_polymer_real_synth_10x.py` | `06_finetune_real_plus_synthetic.py` | SmolVLM LoRA polímero, real+sintético | Tabela `vlm_pilot_results` |
| `04_vlm_polymer_bigvlm_10x.py` | `09_finetune_bigger_vlm.py` | SmolVLM ~2B polímero | Seção do estudo-piloto |
| `05_smolvlm_size_lora_10x.py` | `17_size_finetune_lora.py` | SmolVLM LoRA regime de tamanho | 90,9%/90,8% no abstract |
| `06_smolvlm_size_full_10x.py` | `18_size_finetune_full.py` | SmolVLM full fine-tune regime de tamanho | Tabela de comparação de configs |
| `eval_and_pair.py` (`--pair`) | `20_predicted_box_and_end_to_end.py` | Fim-a-fim pareado (YOLO rep _i_ + SmolVLM rep _i_) | 29,2% fim-a-fim estrito no abstract |

## O que varia entre as 10 repetições, o que fica fixo

- **Varia**: só a seed (0 a 9), explícita e documentada por repetição —
  controla inicialização de pesos, dropout e ordem de embaralhamento
  (`ctrl_common.seed_everything`). Achado importante: os scripts originais
  do YOLO26L **já são determinísticos bit-a-bit** com seed fixa
  (Ultralytics); os do SmolVLM só seedavam o embaralhamento dos dados, não
  a RNG global do PyTorch — por isso a seed completa é controlada aqui.
- **Fixo**: split de dados (partições já existentes em `experimentos/`,
  nunca regeradas), hiperparâmetros idênticos aos scripts originais, mesma
  receita de treino (`vlm_common.finetune_lora`/`finetune_full` reaproveitados
  sem modificação).
- **Ambiente**: única GPU disponível (RTX 4060 Ti); registrado por
  repetição (git commit, versões de biblioteca, GPU) em cada
  `summary.json`, mais um log contínuo de estado da máquina durante toda a
  fila (ver abaixo).

## Onde os arquivos vão parar

- **Checkpoints/CSVs de predição do SmolVLM**: na mesma pasta compartilhada
  que o projeto já usa (`experimentos/resultados/checkpoints/`,
  `experimentos/resultados/reports/`), com tags únicas `<tag>_ctrl_seedN` —
  é a MESMA convenção que os `_run2`/`_run3`/`_run4` ad hoc originais já
  usavam, só que padronizada e completa (10 execuções, seed explícita).
  **Nada existente é sobrescrito.**
- **Checkpoints do YOLO26L**: isolados em `runs/01_yolo26_main/rep_00
  .. rep_09/` (nunca toca `experimentos/runs_v2/`).
- **Resumo por repetição**: `resultados/<experimento>/rep_00
  .. rep_09/summary.json` (formato próprio, `ctrl_common.save_rep_summary`).
- **Log contínuo de máquina**: `logs_maquina/system_monitor_YYYYMMDD.csv`
  (amostra a cada 10s: GPU temp/potência/utilização/memória via pynvml,
  CPU/RAM via psutil, disco livre, e qual repetição estava rodando em cada
  instante) + `logs_maquina/session_manifest.json` (snapshot único: SO,
  driver/CUDA, `pip freeze`, commit git).
- **Relatório final**: `relatorio_final/` — média, desvio-padrão e IC95%
  (n=10, distribuição t) por métrica, em JSON, CSV e um bloco LaTeX pronto
  pra colar no artigo.

## Como rodar

Jobs longos de GPU — rodar no terminal e acompanhar (não em background sem
supervisão pelo menos nas primeiras repetições):

```bash
cd experimentos_controlados/scripts

# 1. valida que tudo roda (só seed=0 de cada experimento) antes de
#    comprometer ~60h de GPU:
python3 run_queue.py --dry-run

# 2. confira os resultados do dry-run contra os números originais do
#    artigo (seed=0 deve bater bit-a-bit pro YOLO, ficar perto pro SmolVLM)

# 3. fila completa (resumível -- pode interromper com Ctrl+C e rodar de
#    novo depois, retoma de onde parou):
python3 run_queue.py
```

`run_queue.py` inicia e mantém o `system_monitor.py` rodando em paralelo
durante toda a fila, roda os 6 experimentos na ordem (YOLO26L →
SmolVLM regime de tamanho LoRA/full → SmolVLM polímero
real-only/real+sintético/bigvlm), depois `eval_and_pair.py` (fim-a-fim
pareado) e por fim `final_report.py` (agregação estatística).

## Orçamento de tempo estimado

~60 horas de GPU (~2,5 dias corridos) — ver detalhamento por experimento no
plano (`/home/mi/.claude/plans/parallel-giggling-ocean.md`).

## Verificação recomendada

1. `run_queue.py --dry-run` roda sem erro.
2. Repetição seed=0 de cada experimento bate (YOLO, bit-a-bit) ou fica
   perto (SmolVLM, mesma ordem de grandeza) do número já publicado no
   artigo.
3. Fila completa roda até o fim; `final_report.py` mostra 10/10 repetições
   pra cada experimento, com IC95% plausíveis (não degenerados).
4. `logs_maquina/system_monitor_*.csv` não tem buracos ao longo das ~60h.
