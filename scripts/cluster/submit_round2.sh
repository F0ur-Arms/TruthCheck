#!/bin/bash
# Round 2 experiments. Run on the master node from ~/FoodFactCheck/TruthCheck_eval.
# The job limit is one per queue, so independent jobs go to different queues and run in parallel.
# Each job writes its PBS log as TRUTHCHECK-EVAL.o<id> in this directory.
set -e
# PBS -v splits on commas, so the origin list uses ":".
ORIGINS=scifact_dataset:healthver_dataset:pubhealth_dataset:ai4bharat_dataset
B=evaluation/baselines/round2
PBS=scripts/cluster/eval_baseline.pbs

# qstat truncates job names, so detect the LLM server by its queue.
qstat -u "$USER" | grep -q Gpu1-10g || qsub ~/vllm_prod.pbs

# E1: local NLI on gold evidence, full passage vs top-2 sentences
qsub -q Gpu4-40g -v "MODE=nli,OUT_DIR=$B/e1_nli_full,NLI_MODELS=MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli,EXTRA=--origins+$ORIGINS" $PBS
qsub -q Gpu4-40g -v "MODE=nli,OUT_DIR=$B/e1_nli_top2,NLI_MODELS=MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli,EXTRA=--origins+$ORIGINS+--select-sentences+2" $PBS
# E2: LLM judge on gold evidence, full passage vs top-2 sentences
qsub -q Gpu2-10g -v "MODE=llmnli,OUT_DIR=$B/e2_llm_full,EXTRA=--origins+$ORIGINS" $PBS
qsub -q Gpu3-10g -v "MODE=llmnli,OUT_DIR=$B/e2_llm_top2,EXTRA=--origins+$ORIGINS+--select-sentences+2" $PBS
# E3: full graph on the cleaner public claims, 8 shards split over two 40GB queues.
# Each half merges only the shards it can see; merge again after both finish.
qsub -q Gpu5-40g -v "MODE=graph,N=8,SHARDS=0+1+2+3,OUT_DIR=$B/e3_graph_public,EXTRA=--public+--details" $PBS
qsub -q Gpu6-40g -v "MODE=graph,N=8,SHARDS=4+5+6+7,OUT_DIR=$B/e3_graph_public,EXTRA=--public+--details" $PBS

qstat -u "$USER"
