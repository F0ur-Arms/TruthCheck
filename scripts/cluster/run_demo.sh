#!/bin/bash
# Demo: the API and web page on the master node, using the 32B server (vllm_32b.pbs) for the LLM.
# Runs from its own folder so evaluation jobs keep their code, and listens on localhost only
# because the API has no login. Open it through an SSH tunnel:
#   ssh -L 8090:127.0.0.1:8090 bagler@192.168.3.91    then http://localhost:8090
# Usage: bash scripts/cluster/run_demo.sh   (from ~/FoodFactCheck/TruthCheck_demo)
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ~/FoodFactCheck/.conda/truthcheck-eval
cd ~/FoodFactCheck/TruthCheck_demo
export HF_HOME=$HOME/FoodFactCheck/hf_cache HF_HUB_OFFLINE=1
export TRUTHCHECK_LLM_BASE_URL=http://cb-gpu-server:8001/v1 TRUTHCHECK_LLM_MODEL=qwen2.5-32b-awq
export TRUTHCHECK_LLM_API_KEY=none TRUTHCHECK_LLM_TIMEOUT_SECONDS=60
export TRUTHCHECK_NLI_LABEL_PRECISION=0.82:0.94:0.41 TRUTHCHECK_MIN_RELEVANCE=${TRUTHCHECK_MIN_RELEVANCE:-0.1}
# Batch 3 knowledge base (index built by the round 11 job); the demo only reads it.
[ -L data ] && [ "$(readlink data)" != ../TruthCheck_kb3/data ] && ln -sfn ../TruthCheck_kb3/data data
# testing.txt is unsourced, so the demo must not cite it (the team has yet to decide on removing it).
export TRUTHCHECK_EXCLUDE_SOURCES=testing.txt
export TRUTHCHECK_CHECKPOINT_DB=$PWD/demo_checkpoints.sqlite TRUTHCHECK_STORE_DB=$PWD/demo_store.sqlite
exec python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8090
