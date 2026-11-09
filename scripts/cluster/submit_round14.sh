#!/bin/bash
# Rounds 14-18: coverage experiments on the public benchmark. Round 14 adds knowledge-base batches 4
# and 5 to round 13; rounds 15-18 each add ONE more change to round 14, so each change is measured
# on its own. Keep a change only if coverage rises and balanced accuracy / "true claims called true"
# hold (merge_shards.py prints all three), then run the kept changes together as a final round.
#   14  round 13 (batch 3, gate 0.1, wider retrieval, prompt v1) + KB batches 4-5
#       (MedlinePlus, NCCIH herbs and topics, NCI alternative-cancer-treatment summaries)
#   15  round 14 + plain query lanes        TRUTHCHECK_QUERY_LANES=v2
#   16  round 14 + combined judgment        TRUTHCHECK_NLI_COMBINE=3
#   17  round 14 + judge prompt v3          TRUTHCHECK_NLI_PROMPT=v3
#   18  round 14 + relevance gate 0.05      TRUTHCHECK_MIN_RELEVANCE=0.05
#       (round 7 showed that no gate calls 4 in 10 true claims false; keep 0.05 only if that holds,
#        and remember every threshold here is picked on this same benchmark)
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
# A final combined round: EXTRA_ENV="TRUTHCHECK_QUERY_LANES=v2,TRUTHCHECK_NLI_COMBINE=3" ... 14
set -e
ROUND=${1:?usage: submit_round14.sh 14|15|16|17|18   [EXTRA_ENV=VAR=value,... OUT_NAME=name]}
PBS=scripts/cluster/eval_baseline.pbs

COMMON="TRUTHCHECK_LLM_BASE_URL=http://cb-gpu-server:8001/v1,TRUTHCHECK_LLM_MODEL=qwen2.5-32b-awq"
COMMON="$COMMON,TRUTHCHECK_RETRIEVAL_POOL=50,TRUTHCHECK_RERANK_KEEP=10"
# The judge's per-label confidences change which claims get a verdict. The repository does not record
# the value round 13 used (the demo uses 0.82:0.94:0.41); export the same one before running, e.g.
#   TRUTHCHECK_NLI_LABEL_PRECISION=0.82:0.94:0.41 bash scripts/cluster/submit_round14.sh 14
if [ -n "${TRUTHCHECK_NLI_LABEL_PRECISION:-}" ]; then
  COMMON="$COMMON,TRUTHCHECK_NLI_LABEL_PRECISION=$TRUTHCHECK_NLI_LABEL_PRECISION"
else
  echo "WARNING: TRUTHCHECK_NLI_LABEL_PRECISION not set; using the code default (7B precisions)." >&2
fi

GATE=0.1 PROMPT=v1 MORE=""
case "$ROUND" in
  14) OUT=round14_kb45 ;;
  15) OUT=round15_lanes_v2;   MORE="TRUTHCHECK_QUERY_LANES=v2" ;;
  16) OUT=round16_combine3;   MORE="TRUTHCHECK_NLI_COMBINE=3" ;;
  17) OUT=round17_prompt_v3;  PROMPT=v3 ;;
  18) OUT=round18_gate005;    GATE=0.05 ;;
  *) echo "round must be 14-18"; exit 1 ;;
esac
ENV="$COMMON,TRUTHCHECK_MIN_RELEVANCE=$GATE,TRUTHCHECK_NLI_PROMPT=$PROMPT"
[ -n "$MORE" ] && ENV="$ENV,$MORE"
[ -n "${EXTRA_ENV:-}" ] && ENV="$ENV,$EXTRA_ENV"
OUT=${OUT_NAME:-$OUT}

echo "round $ROUND -> evaluation/baselines/$OUT  ($ENV)"
qsub -q Gpu4-40g -v "MODE=graph,N=6,OUT_DIR=evaluation/baselines/$OUT/graph_public_32b_gate01,EXTRA=--public+--details,$ENV" $PBS
qstat -u "$USER"
