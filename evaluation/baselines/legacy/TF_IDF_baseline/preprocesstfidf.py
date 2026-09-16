"""
preprocess.py
─────────────
Text preprocessing pipeline as described in the paper (Section 3.2):
  - Lowercasing
  - Punctuation removal
  - Stopword removal
  - Tokenization
  - Lemmatization
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK data on first run
nltk.download('punkt',       quiet=True)
nltk.download('punkt_tab',   quiet=True)
nltk.download('stopwords',   quiet=True)
nltk.download('wordnet',     quiet=True)
nltk.download('omw-1.4',     quiet=True)

STOP_WORDS  = set(stopwords.words('english'))
LEMMATIZER  = WordNetLemmatizer()


def preprocess(text: str) -> str:
    """
    Full preprocessing pipeline.
    Returns a cleaned string ready for TF-IDF or BERT tokenization.
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # 3. Remove punctuation and non-alphanumeric characters
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # 4. Tokenize
    tokens = word_tokenize(text)

    # 5. Remove stopwords + lemmatize
    tokens = [
        LEMMATIZER.lemmatize(token)
        for token in tokens
        if token not in STOP_WORDS and len(token) > 1
    ]

    return " ".join(tokens)


if __name__ == "__main__":
    samples = [
        "Drinking WARM water on an empty stomach CURES diabetes!!!",
        "Doctors don't want you to know this SECRET remedy for cancer.",
        "Regular exercise improves cardiovascular health.",
    ]
    for s in samples:
        print(f"Original : {s}")
        print(f"Cleaned  : {preprocess(s)}\n")