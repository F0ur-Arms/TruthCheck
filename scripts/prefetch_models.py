"""Download the configured self-hosted model weights into the Hugging Face cache."""

import sys
from pathlib import Path

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import EMBEDDING_MODEL, NLI_MODEL


def main():
    print(f"Prefetching embedding model: {EMBEDDING_MODEL}")
    SentenceTransformer(EMBEDDING_MODEL)
    print(f"Prefetching NLI model: {NLI_MODEL}")
    AutoTokenizer.from_pretrained(NLI_MODEL)
    AutoModelForSequenceClassification.from_pretrained(NLI_MODEL)
    print("Model prefetch complete.")


if __name__ == "__main__":
    main()
