# Local Reasoning LLM (vLLM on the IIITD CB Cluster)

TruthCheck's claim understanding, decomposition, query generation, and
synthesis are backed by a self-hosted **Qwen2.5-7B-Instruct-AWQ** model,
served through [vLLM](https://github.com/vllm-project/vllm) on the IIITD CB
GPU cluster. It exposes an OpenAI-compatible `/v1/chat/completions` API that
`src/llm_fallback.py`'s `ConfiguredLLMVerifier` (and `src/claims_processor.py`)
already know how to talk to -- no application code changes are needed,
only environment variables.

This is **not** the embedding/NLI/reranker stack (`BAAI/bge-m3`,
`mDeBERTa-v3-base-mnli-xnli`, `BAAI/bge-reranker-v2-m3`), which is unrelated
and unaffected by any of this.

## Architecture

```
TruthCheck (ClaimsProcessor / ConfiguredLLMVerifier)
    |
    | OpenAI-compatible HTTP, base_url + served model name from env vars
    v
vLLM server (PBS job, port 8000, on a GPU compute node)
    |
    v
Qwen2.5-7B-Instruct-AWQ  --  NVIDIA A100 MIG slice (10GB)
```

## Quick start (using an already-running server)

1. Check whether the server is up (see "Operations" below).
2. Copy `.env.example` to `.env` (or export the variables in your shell).
3. If you're not on the cluster network, open a tunnel first:
   ```bash
   ssh -L 8000:cb-gpu-server:8000 <you>@cb-cluster.iiitd.edu.in
   ```
   and set `TRUTHCHECK_LLM_BASE_URL=http://localhost:8000/v1` instead.
4. Run TruthCheck as usual. `ConfiguredLLMVerifier.configured` will be
   `True` and the LLM path will be used automatically.

## Cluster facts this deployment depends on

- Master node (`cb-cluster.iiitd.edu.in`) has **no GPU**. All GPU work goes
  through PBS.
- Queue `Gpu1-10g` maps to MIG slice `MIG-f3d88dc3-260f-516c-9565-e895709806c8`
  (10GB, NVIDIA A100 80GB card). One job per user, 10 CPU cores max,
  **48-hour walltime hard limit** -- PBS kills the job automatically after
  that.
- The compute node the job lands on is `cb-gpu-server`.
- Driver on the compute nodes is `555.42.02`, which supports **up to CUDA
  12.5**. This constrains every package version below.

## Operations

### Check if the server is running
```bash
ssh <you>@cb-cluster.iiitd.edu.in
qstat | grep TRUTHCHECK-VLLM
```
If nothing shows up, it's not running -- see "Starting the server" below.

### Confirm it's actually answering requests
The job can show as `R` (running) in `qstat` while still loading the model
(safetensors load + CUDA graph capture takes ~2-4 minutes after the job
starts). The reliable check is hitting the API directly, from the master
node or a tunnel:
```bash
curl -s http://cb-gpu-server:8000/v1/models
```
A `Connection refused` means it's still starting (or crashed/exited); a JSON
body listing `qwen2.5-7b-awq` means it's ready.

**Do not rely on tailing `vllm_prod.out` / `vllm_prod.err` while the job is
running.** PBS spools output on the compute node and only ever syncs it back
to the paths above when the job *exits* -- for a long-running server that
hasn't crashed, you will see nothing update, even though the server is fine.
If you need to watch startup live, use an interactive job instead:
```bash
qsub -I -l select=1:ncpus=8 -l walltime=00:20:00 -q Gpu1-10g
export CUDA_VISIBLE_DEVICES=MIG-f3d88dc3-260f-516c-9565-e895709806c8
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ~/FoodFactCheck/.conda/truthcheck-vllm-cu125
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --served-model-name qwen2.5-7b-awq --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 --gpu-memory-utilization 0.85 --max-model-len 8192 --quantization awq
```
Exiting the interactive shell (or its 48h walltime elapsing) kills the
server, so this is for debugging only -- use the batch job below for
anything you want to actually stay up.

### Starting the server
```bash
qsub scripts/cluster/vllm_prod.pbs
```
If the conda environment doesn't exist yet (fresh clone, new teammate,
another user's account), run `bash scripts/cluster/setup_vllm_env.sh` first
-- see "Setting up the environment from scratch" below.

### Stopping / restarting
Only one job per user is allowed, so to restart:
```bash
qstat | grep TRUTHCHECK-VLLM    # find the job id
qdel <job-id>
qsub scripts/cluster/vllm_prod.pbs
```

### The 48-hour limit
PBS auto-kills the job after 48 hours regardless of activity. This is **not
a production service that stays up indefinitely** -- treat it as something
that needs to be noticed and resubmitted (`qsub scripts/cluster/vllm_prod.pbs`)
roughly every 2 days. There's no auto-restart configured; if you want one,
a cron job on the master node running `qstat | grep -q TRUTHCHECK-VLLM ||
qsub scripts/cluster/vllm_prod.pbs` would do it, but be aware of the
one-job-per-user cluster policy before automating this for multiple
teammates.

## Setting up the environment from scratch

The conda environment (`~/FoodFactCheck/.conda/truthcheck-vllm-cu125`) is
**not** part of this git repository -- it's a per-user directory on
`/storage`. If you're a new teammate, or the environment gets corrupted, or
you're setting this up under your own cluster account, run:
```bash
bash scripts/cluster/setup_vllm_env.sh
```
This creates the environment and installs the exact package versions
confirmed to work on this cluster's driver (see "Why these specific
versions" below). It's idempotent -- safe to re-run.

**After running it, verify CUDA actually works** (the setup script prints
this same reminder): the master node has no GPU, so `torch.cuda.is_available()`
will report `False` if you check it there -- that's expected and not a
problem. Check from a real GPU job instead:
```bash
qsub -I -l select=1:ncpus=8 -l walltime=00:10:00 -q Gpu1-10g
export CUDA_VISIBLE_DEVICES=MIG-f3d88dc3-260f-516c-9565-e895709806c8
source ~/anaconda3/etc/profile.d/conda.sh
conda activate ~/FoodFactCheck/.conda/truthcheck-vllm-cu125
python -c "import torch; print(torch.cuda.is_available())"   # must print True
exit
```

## Why these specific versions (do not casually upgrade)

Getting this to work required tracking down four separate, stacked
incompatibilities. If you upgrade `torch`, `vllm`, `transformers`, or
`xgrammar` independently, expect to hit these again -- `setup_vllm_env.sh`
pins all of them together for a reason.

1. **PyTorch must be a CUDA 12.1 build (`torch==2.5.1+cu121`), not whatever
   pip resolves by default.** The cluster's driver (555.42.02) only
   supports CUDA up to 12.5. A default `pip install torch` (or letting
   `vllm`'s own dependency resolution pick a torch version) pulls a build
   compiled for CUDA 13, which does *not* error -- it just makes
   `torch.cuda.is_available()` silently return `False`, with a driver
   version warning easy to miss in logs.

2. **vLLM must be `0.7.2`, not a newer release like `0.28.x`.** Newer vLLM
   releases ship a compiled extension (`vllm._C_stable_libtorch`) that
   itself requires `libcudart.so.13` (CUDA 13) regardless of which PyTorch
   build is installed alongside it -- this fails at `import vllm` with
   `ImportError: libcudart.so.13: cannot open shared object file`.

3. **`transformers` must be pinned to `4.48.2`.** vLLM 0.7.2's tokenizer
   caching code calls `tokenizer.all_special_tokens_extended`, an attribute
   later `transformers` releases removed, causing
   `AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended`
   on every request.

4. **`xgrammar` must be pinned to `0.1.11`.** vLLM 0.7.2's OpenAI-compatible
   endpoint uses xgrammar to implement `response_format={"type":
   "json_object"}` -- which is exactly what `ConfiguredLLMVerifier` and
   `ClaimsProcessor._call_llm_json` send on every call. xgrammar `>= 0.2`
   renamed `TokenizerInfo.from_huggingface`; the mismatch crashes the
   *entire* vLLM engine process (not just the one request) with
   `AttributeError: type object 'TokenizerInfo' has no attribute
   'from_huggingface'`.

5. **`vllm/platforms/cuda.py` needs a small source patch for MIG support.**
   vLLM assumes `CUDA_VISIBLE_DEVICES` only ever contains integer indices,
   and crashes with `ValueError: invalid literal for int() with base 10:
   'MIG-...'` when it's set to a MIG UUID -- which is required to select a
   specific GPU slice on this cluster. This is a known, still-open upstream
   issue: [vllm-project/vllm#13815](https://github.com/vllm-project/vllm/issues/13815),
   [#17047](https://github.com/vllm-project/vllm/issues/17047),
   [#7211](https://github.com/vllm-project/vllm/issues/7211),
   [#32569](https://github.com/vllm-project/vllm/issues/32569) (feature
   request, still open). `setup_vllm_env.sh` patches
   `device_id_to_physical_device_id()` to fall back to the local device
   index when the env var isn't a plain integer -- safe, because
   `CUDA_VISIBLE_DEVICES` has already restricted the process to exactly one
   device at the driver level by the time this code runs.

## Known limitations

- **Structured JSON is required and now works, but isn't free.** The first
  request that uses `response_format: json_object` after a fresh server
  start triggers xgrammar's one-time grammar-compilation step and can take
  noticeably longer than subsequent ones. `src/claims_processor.py`'s
  `_call_llm_json` hardcodes a 15-second timeout for this call, separate
  from the configurable `TRUTHCHECK_LLM_TIMEOUT_SECONDS` -- if you see
  timeouts specifically on the *first* claim processed after a restart,
  this is why. Warming the server with one throwaway JSON-mode request
  right after startup avoids this for real users.
- **AWQ quantization is explicitly slower than vLLM's optimized path.**
  vLLM logs `awq quantization is not fully optimized yet` on startup --
  this is expected and not something to "fix"; it's a known tradeoff we
  accepted for the VRAM savings on a 10GB MIG slice (see the
  model-selection rationale below).
- **`--max-model-len 8192`** caps total prompt + completion tokens. This
  was chosen to fit comfortably in the 10GB MIG slice with AWQ
  quantization; if a use case needs longer context, that requires either a
  larger MIG slice (`Gpu4/5/6-40g`) or a shorter-context model, not a flag
  change on this one.
- Not evaluated: multi-turn conversation history length, concurrent-request
  throughput under real traffic, or behavior once VRAM is shared with other
  jobs on the same physical A100 (MIG slices are hardware-isolated from
  each other, so this shouldn't be an issue, but hasn't been load-tested).

## Model / VRAM decision rationale

Qwen2.5-7B-Instruct-AWQ on the 10GB MIG slice (`Gpu1-10g`) was chosen over:
- **Qwen2.5-3B-Instruct** (same queue): smaller and faster, but weaker
  reasoning for multi-clause claim decomposition -- the 7B model's
  structured-output reliability and Hindi/Hinglish handling were judged
  worth the extra VRAM.
- **A larger Qwen model on `Gpu4/5/6-40g` or `Gpu7-80g`**: rejected for this
  use case. Claim decomposition, routing, and query generation don't need
  the larger context/parameter budget those models offer, and staying on a
  10GB MIG slice avoids resource contention with the long-running
  molecular-dynamics jobs that regularly occupy the 40GB queues.

Embeddings (`BAAI/bge-m3`), NLI (`mDeBERTa-v3-base-mnli-xnli`), and
reranking (`BAAI/bge-reranker-v2-m3`) remain on the separate,
already-working `truthcheck-embed` environment and are untouched by any of
this -- this LLM is purely the reasoning/generation layer.
