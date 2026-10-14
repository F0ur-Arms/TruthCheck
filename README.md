# TruthCheck

Multilingual (English / Hinglish / Hindi) health-misinformation verification for Indian healthcare claims. A claim goes in; a verdict, confidence, risk assessment and cited evidence come out.

> **Status: v2 integration in progress.** The legacy pipeline (`main.py`) works end to end. The LangGraph pipeline (`graph/`) is the v2 target, but several of its nodes are still stubs. See "Current status" below before trusting any reported metric.

## Pipeline

```
raw input
  -> HinglishMapper / multilingual preprocessing   (src/preprocessor.py, multilingual/)
  -> route + decompose claim (LLM, heuristic fallback)   (src/claims_processor.py)
       -> medical-advice request  => NOT_A_FACT_CHECK
  -> multi-lane query generation                    (src/hybrid_retriever.py)
  -> BM25 + dense retrieval, RRF fusion, cross-encoder rerank
  -> evidence quality scoring + NLI                 (src/evidence_quality.py, src/RAGpipeline/nli_verifier.py)
  -> confidence calibration                         (src/calibration.py)
  -> two-axis risk (confidence vs. harm)            (src/risk_engine_v2.py)
  -> human review (low confidence / high harm)
  -> response with citations
```

Two entry points exist:

- `main.py` — legacy `TruthCheckPipeline.analyze_query`, fully wired.
- `graph/workflow.py` — LangGraph state machine (7 nodes, 2 conditional edges), served by `app/api/main.py`.

## Layout

| Path | Purpose |
|---|---|
| `main.py` | Legacy end-to-end pipeline |
| `graph/` | LangGraph workflow: `state.py`, `nodes.py`, `workflow.py` |
| `app/api/` | FastAPI app (`POST /api/v1/fact-check`) |
| `web/` | Web UI (home, chat, FAQ) |
| `src/` | Claims processing, retrieval, calibration, risk, evidence quality, LLM client |
| `src/RAGpipeline/` | Knowledge manager and NLI verifier |
| `multilingual/` | Language-ID, transliteration, gloss, semantic-preservation and medical-entity guard layer |
| `pipeline/` | Early OCR / ASR / language-detect stubs (not wired) |
| `evaluation/` | Baselines, golden-set candidates, evaluation scripts |
| `data/` | Verified facts, medical KB (`data/medical_kb/`), evaluation CSV |
| `scripts/cluster/` | PBS / conda scripts for the vLLM server |
| `docs/` | Runbooks (see `docs/LOCAL_LLM_DEPLOYMENT.md`) |
| `tests/` | pytest suite |
| `config.py` | Paths, model names, LLM environment variables |

## Setup

Use **Python 3.11** (`spacy==3.8.4` has no wheel for 3.13).

```bash
python3.11 -m venv venv311
source venv311/bin/activate
pip install -r requirements.txt
```

Models are downloaded from Hugging Face on first use: `BAAI/bge-m3`, `BAAI/bge-reranker-v2-m3`, `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`. The multilingual layer has its own `multilingual/requirements.txt`.

Optional external tools for the image/audio paths: Tesseract OCR and FFmpeg.

## Local reasoning LLM

Claim routing, decomposition, query generation and explanation synthesis use a self-hosted Qwen2.5-7B-Instruct-AWQ served by vLLM on the IIITD CB GPU cluster (OpenAI-compatible API). No code changes are needed, only environment variables:

```bash
cp .env.example .env     # then edit
# TRUTHCHECK_LLM_BASE_URL, TRUTHCHECK_LLM_API_KEY, TRUTHCHECK_LLM_MODEL, TRUTHCHECK_LLM_TIMEOUT_SECONDS
```

From outside the cluster, open a tunnel first (`ssh -L 8000:cb-gpu-server:8000 <you>@cb-cluster.iiitd.edu.in`) and use `http://localhost:8000/v1`. Without these variables the app still runs, using heuristic routing and decomposition. Full runbook, version pins and the 48-hour job limit: [docs/LOCAL_LLM_DEPLOYMENT.md](docs/LOCAL_LLM_DEPLOYMENT.md).

## Run

```bash
# API + web UI
uvicorn app.api.main:app --reload

# Tests
python -m pytest tests -q

# Legacy pipeline
python main.py
```

## Current status

- Built and tested: claim routing/decomposition, medical-advice gate, hybrid retrieval with RRF and rerank, mDeBERTa NLI verifier, two-axis risk engine, multilingual preprocessing layer, FastAPI + web UI.
- Stubbed or not yet wired in the LangGraph path: NLI in `graph/nodes.py` (hardcoded), dense retrieval over the full KB, two-axis risk, human-review interrupt, persistent storage.
- Not started: Postgres/pgvector, trained calibrator, reviewed multilingual golden set, OCR/ASR integration.
- Baseline (legacy pipeline, `evaluation/baselines/phase0/`): accuracy 0.6831, F1 0.6709 on 2,799 claims, run without the LLM connected.
- The numbers produced by `evaluation/generate_evaluation_report.py` defaults and `run_validation_audit.py` are placeholders / smoke checks, not results.
