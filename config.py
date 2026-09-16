import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent

# Paths used by the live pipeline.  All are repository-relative so the project
# can run from any working directory, operating system, or CI runner.
DATA_DIR = BASE_DIR / "data"
PATTERNS_FILE = DATA_DIR / "lifestyle_patterns.jsonl"
FACTS_JSON = DATA_DIR / "verified_facts.json"
LEGACY_FACTS_JSON = DATA_DIR / "final_verifiedfacts.json"
KB_PATH = DATA_DIR / "medical_kb"
EVALUATION_DATASET = DATA_DIR / "final_health_claims.csv"
MODEL_PATH = BASE_DIR / "models" / "language" / "baseline_lr_model.pkl"
VEC_PATH = BASE_DIR / "models" / "language" / "tfidf_vectorizer.pkl"
LOG_PATH = BASE_DIR / "logs.txt"
DEBUG_LOG_PATH = BASE_DIR / "src" / "loginput.txt"

# Model choices.  Caches are model-specific, so changing either identifier
# cannot accidentally reuse vectors produced by a different embedding model.
EMBEDDING_MODEL = "BAAI/bge-m3"
NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"

# The fallback uses an explicit OpenAI-compatible endpoint.  It is disabled
# unless both values are supplied by the deployment environment.
LLM_BASE_URL = os.getenv("TRUTHCHECK_LLM_BASE_URL", "").rstrip("/")
LLM_API_KEY = os.getenv("TRUTHCHECK_LLM_API_KEY", "")
LLM_MODEL = os.getenv("TRUTHCHECK_LLM_MODEL", "")
LLM_TIMEOUT_SECONDS = float(os.getenv("TRUTHCHECK_LLM_TIMEOUT_SECONDS", "30"))
UNREVIEWED_FACTS_INPUT = DATA_DIR / "verified_factsv2.json"
FACT_LABEL_SUGGESTIONS = DATA_DIR / "verified_facts_label_suggestions.json"

# Risk Engine Weights
WEIGHT_FACT  = 0.7
WEIGHT_STYLE = 0.3

# Translation Settings
SOURCE_LANG = "hi"
TARGET_LANG = "en"
