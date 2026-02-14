import pandas as pd
from pipeline.normalize import normalize_text

def load_dataset(path):
    df = pd.read_csv(path)

    texts = df["text"].apply(normalize_text).tolist()
    labels = df["risk_label"].tolist()

    return texts, labels
