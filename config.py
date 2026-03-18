import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
DATA_DIR      = os.path.join(BASE_DIR, "data")
PATTERNS_FILE = os.path.join(BASE_DIR, "data", "lifestyle_patterns.jsonl")
FACTS_JSON    = os.path.join(BASE_DIR, "data", "verified_facts.json")
KB_PATH       = os.path.join(BASE_DIR, "data", "medical_kb")
MODEL_PATH    = os.path.join(BASE_DIR, "models", "language", "baseline_lr_model.pkl")
VEC_PATH      = os.path.join(BASE_DIR, "models", "language", "tfidf_vectorizer.pkl")

# Risk Engine Weights
WEIGHT_FACT  = 0.7
WEIGHT_STYLE = 0.3

# Translation Settings
SOURCE_LANG = "hi"
TARGET_LANG = "en"