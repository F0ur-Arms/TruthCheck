#!/bin/bash
# Rounds 14-15: more knowledge-base coverage, one change per round, on the public benchmark.
#   round 14: round 13 (batch 3, gate 0.1, wider retrieval, prompt v1) + KB batch 4 (MedlinePlus, NCCIH)
#   round 15: round 14 + judge prompt v3 (v3 and wider retrieval were never run together)
# The SciFact abstract corpus was considered and left out: it holds the gold evidence quote of 159 of
# the 167 SciFact benchmark claims verbatim, so it would let the system look the answers up.
#
# Before running:
#   1. Check the TRUTHCHECK_* values below match the ones round 13 used; only the change named in
#      each round may differ, or the comparison is not fair.
#   2. Build the index once for the new files (the eval jobs only read it):
#        qsub -v PROJECT_DIR=$PWD scripts/cluster/embed_medical_kb.pbs
#      and wait for it to finish (the cluster runs one job per queue, two per user).
#   3. Start the 32B server if it is not running: qsub scripts/cluster/vllm_32b.pbs
# Run on the master node from the checkout that holds the new knowledge base, one round at a time,
# then merge:  python evaluation/merge_shards.py evaluation/baselines/<round>/graph_public_32b_gate01
set -e
ROUND=${1:?usage: submit_round14.sh 14|15}
PBS=scripts/cluster/eval_baseline.pbs

COMMON="TRUTHCHECK_LLM_BASE_URL=http://cb-gpu-server:8001/v1,TRUTHCHECK_LLM_MODEL=qwen2.5-32b-awq"
COMMON="$COMMON,TRUTHCHECK_MIN_RELEVANCE=0.1,TRUTHCHECK_RETRIEVAL_POOL=50,TRUTHCHECK_RERANK_KEEP=10"
# The judge's per-label confidences change which claims get a verdict. The repository does not record
# the value round 13 used (the demo uses 0.82:0.94:0.41); export the same one before running, e.g.
#   TRUTHCHECK_NLI_LABEL_PRECISION=0.82:0.94:0.41 bash scripts/cluster/submit_round14.sh 14
if [ -n "${TRUTHCHECK_NLI_LABEL_PRECISION:-}" ]; then
  COMMON="$COMMON,TRUTHCHECK_NLI_LABEL_PRECISION=$TRUTHCHECK_NLI_LABEL_PRECISION"
else
  echo "WARNING: TRUTHCHECK_NLI_LABEL_PRECISION not set; using the code default (7B precisions)." >&2
fi
case "$ROUND" in
  14) OUT=round14_kb4;    ENV="$COMMON,TRUTHCHECK_NLI_PROMPT=v1" ;;
  15) OUT=round15_kb4_v3; ENV="$COMMON,TRUTHCHECK_NLI_PROMPT=v3" ;;
  *) echo "round must be 14 or 15"; exit 1 ;;
esac

qsub -q Gpu4-40g -v "MODE=graph,N=6,OUT_DIR=evaluation/baselines/$OUT/graph_public_32b_gate01,EXTRA=--public+--details,$ENV" $PBS
qstat -u "$USER"
