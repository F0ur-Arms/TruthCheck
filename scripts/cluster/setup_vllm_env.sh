#!/bin/bash
# Creates (or repairs) the conda environment used to serve the local reasoning
# LLM for TruthCheck via vLLM, on the IIITD CB GPU cluster.
#
# Run this ONCE from the cluster master node (cb-cluster.iiitd.edu.in), from
# anywhere -- it does not need PBS/GPU access, only network access to
# download packages. It is idempotent: safe to re-run if it fails partway.
#
# Usage:
#   bash scripts/cluster/setup_vllm_env.sh
#
# Background: the cluster's NVIDIA driver (555.42.02) only supports CUDA up
# to 12.5. Installing "whatever pip resolves by default" for vllm pulls in a
# PyTorch build compiled for CUDA 13, which silently makes
# torch.cuda.is_available() return False, and separately vLLM's own
# compiled extensions require libcudart.so.13. The versions below are the
# specific combination confirmed to work on this cluster. Do not upgrade
# any of them without testing on a GPU node (qsub -I) first.
set -euo pipefail

ENV_PATH="$HOME/FoodFactCheck/.conda/truthcheck-vllm-cu125"

echo "== Creating conda environment at $ENV_PATH (python 3.11) =="
if [ -d "$ENV_PATH" ]; then
  echo "Environment already exists, reusing it."
else
  conda create -p "$ENV_PATH" python=3.11 -y
fi

source ~/anaconda3/etc/profile.d/conda.sh
conda activate "$ENV_PATH"

echo "== Installing PyTorch 2.5.1 (cu121 build) =="
# Must be a cu11x/cu12x-early build. Anything requiring CUDA > 12.5 will
# report torch.cuda.is_available() == False on this cluster's driver.
pip install torch==2.5.1 torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu121

echo "== Installing vLLM 0.7.2 =="
# vLLM >= ~0.8 (and the 0.2x/0.28 line) ships compiled extensions that
# require libcudart.so.13 (CUDA 13), which this cluster cannot provide.
# 0.7.2 is the newest line confirmed to import and run on driver 555.42.02.
pip install vllm==0.7.2

echo "== Pinning transformers to 4.48.2 =="
# vLLM 0.7.2's tokenizer caching code calls
# Qwen2Tokenizer.all_special_tokens_extended, which newer `transformers`
# releases removed/renamed. 4.48.2 is the last version confirmed compatible.
pip install transformers==4.48.2

echo "== Pinning xgrammar to 0.1.11 =="
# vLLM 0.7.2's OpenAI-compatible endpoint uses xgrammar for
# response_format={"type": "json_object"} requests (used throughout
# TruthCheck's claim decomposition/routing prompts), regardless of the
# --guided-decoding-backend flag. xgrammar >= 0.2 renamed
# TokenizerInfo.from_huggingface, which crashes -- and takes the whole
# server process down -- on the first structured-JSON request. 0.1.11 is
# the version vLLM 0.7.2 was actually built against.
pip install xgrammar==0.1.11

echo "== Patching vllm/platforms/cuda.py for MIG UUID support =="
# vLLM (all versions as of this writing; see vllm-project/vllm#13815,
# #17047, #7211, #32569) assumes CUDA_VISIBLE_DEVICES only ever contains
# integer indices, and crashes with
#   ValueError: invalid literal for int() with base 10: 'MIG-...'
# when it's set to a MIG UUID -- which is required on this cluster to
# select a specific GPU slice. CUDA_VISIBLE_DEVICES has already restricted
# the process to exactly one device by the time this code runs, so the
# correct local index is always the requested device_id itself; this patch
# just falls back to that instead of crashing.
CUDA_PY="$ENV_PATH/lib/python3.11/site-packages/vllm/platforms/cuda.py"
python3 - "$CUDA_PY" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path) as f:
    content = f.read()

old = "        physical_device_id = device_ids[device_id]\n        return int(physical_device_id)"
new = (
    "        physical_device_id = device_ids[device_id]\n"
    "        try:\n"
    "            return int(physical_device_id)\n"
    "        except ValueError:\n"
    "            # physical_device_id is a UUID (e.g. \"MIG-...\" or \"GPU-...\").\n"
    "            # CUDA_VISIBLE_DEVICES has already restricted visibility to this\n"
    "            # device, so the local index is device_id itself.\n"
    "            return device_id"
)

if new in content:
    print("Patch already applied, nothing to do.")
elif old in content:
    content = content.replace(old, new)
    with open(path, "w") as f:
        f.write(content)
    print("Patch applied.")
else:
    print(
        "WARNING: could not find the expected function body in "
        f"{path}. vLLM's source may have changed -- check "
        "device_id_to_physical_device_id() manually before relying on MIG "
        "device selection.",
        file=sys.stderr,
    )
    sys.exit(1)
PYEOF

echo ""
echo "== Verifying import (module-level only; run the CUDA check below from a PBS job) =="
python -c "import torch, vllm, transformers, xgrammar; print('torch', torch.__version__); print('vllm', vllm.__version__); print('transformers', transformers.__version__)"

echo ""
echo "Setup complete. IMPORTANT: torch.cuda.is_available() will report False"
echo "here because this is the master node, which has no GPU. Verify CUDA"
echo "works with an interactive job before trusting the environment:"
echo ""
echo "  qsub -I -l select=1:ncpus=8 -l walltime=00:10:00 -q Gpu1-10g"
echo "  export CUDA_VISIBLE_DEVICES=MIG-f3d88dc3-260f-516c-9565-e895709806c8"
echo "  source ~/anaconda3/etc/profile.d/conda.sh"
echo "  conda activate $ENV_PATH"
echo "  python -c \"import torch; print(torch.cuda.is_available())\"  # must print True"
