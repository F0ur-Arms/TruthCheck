import pandas as pd
from pipeline.normalize import normalize_text

def load_dataset(path):
    df = pd.read_csv(path)

    if "text" not in df.columns:
        raise KeyError("Dataset must contain a 'text' column.")

    label_col = None
    for candidate in ("risk_label", "label"):
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is None:
        raise KeyError("Dataset must contain either 'risk_label' or 'label'.")

    texts = df["text"].astype(str).apply(normalize_text).tolist()
    labels = df[label_col].astype(int).tolist()

    return texts, labels
