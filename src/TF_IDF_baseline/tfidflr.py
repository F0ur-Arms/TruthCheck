"""
tfidf_lr.py
────────────
TF-IDF + Logistic Regression baseline classifier.
Trains on Kaggle Fake and Real News Dataset.
Tests on your custom health_claims.csv.

Usage:
    python tfidf_lr.py
"""

import os
import kagglehub
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model            import LogisticRegression
from sklearn.metrics                 import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from preprocess import preprocess


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
TEST_DATA_PATH = "data/health_claims.csv"   
MODEL_DIR      = "models/"
TFIDF_MAX_FEAT = 5000
RANDOM_STATE   = 42


def load_kaggle_train_data():
    print("[Step 1] Downloading Kaggle dataset...")
    path = kagglehub.dataset_download("clmentbisaillon/fake-and-real-news-dataset")
    print(f"         Downloaded to: {path}\n")

    fake_path = os.path.join(path, "Fake.csv")
    true_path = os.path.join(path, "True.csv")

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    # Label: 1 = Fake, 0 = Real
    fake_df['label'] = 1
    true_df['label'] = 0

    # Use 'text' column as per paper
    fake_df = fake_df[['text', 'label']].rename(columns={'text': 'claim'})
    true_df = true_df[['text', 'label']].rename(columns={'text': 'claim'})

    df = pd.concat([fake_df, true_df], ignore_index=True).dropna(subset=['claim'])
    df = df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    print(f"[Data]   Train set size : {len(df)}")
    print(f"[Data]   Label distribution:\n{df['label'].value_counts()}\n")

    return df

def load_test_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Test dataset not found at: {path}\n"
            f"Create a CSV with columns: claim, label (0=Real, 1=Fake)"
        )

    df = pd.read_csv(path).dropna(subset=['claim', 'label'])
    df['label'] = df['label'].astype(int)

    print(f"[Data]   Test set size  : {len(df)}")
    print(f"[Data]   Label distribution:\n{df['label'].value_counts()}\n")

    return df


# ─────────────────────────────────────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(y_true, y_pred, model_name="Model"):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\n{'─'*50}")
    print(f"  {model_name} — Results")
    print(f"{'─'*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Real (0)', 'Fake (1)'])}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def plot_confusion_matrix(y_true, y_pred, save_path="results/cm_tfidf_lr.png"):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Real', 'Fake'],
        yticklabels=['Real', 'Fake']
    )
    plt.title('Confusion Matrix — TF-IDF + Logistic Regression')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(save_path)
    print(f"[Plot]   Confusion matrix saved to {save_path}")
    plt.show()

if __name__ == "__main__":
    print("=" * 50)
    print("  TF-IDF + Logistic Regression Pipeline")
    print("=" * 50)

    # Step 1: Load training data from Kaggle
    train_df = load_kaggle_train_data()

    # Step 2: Preprocess training data
    print("[Step 2] Preprocessing training data (this may take a few minutes)...")
    train_df['clean_claim'] = train_df['claim'].apply(preprocess)

    # Step 3: TF-IDF vectorization
    print(f"[Step 3] TF-IDF vectorization (max_features={TFIDF_MAX_FEAT})...")
    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEAT, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(train_df['clean_claim'])
    y_train = train_df['label']

    # Step 4: Train Logistic Regression
    print("[Step 4] Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    print("         Training complete.")

    # Step 5: Save model artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,      os.path.join(MODEL_DIR, "baseline_lr_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    print(f"\n[Saved]  Model      → {MODEL_DIR}baseline_lr_model.pkl")
    print(f"[Saved]  Vectorizer → {MODEL_DIR}tfidf_vectorizer.pkl")

    # Step 6: Load and preprocess your health claims test data
    print(f"\n[Step 5] Loading test data from {TEST_DATA_PATH}...")
    test_df = load_test_data(TEST_DATA_PATH)
    test_df['clean_claim'] = test_df['clean_claim'] = test_df['claim'].apply(preprocess)

    # Step 7: Predict and evaluate
    print("[Step 6] Running predictions on health claims test set...")
    X_test = vectorizer.transform(test_df['clean_claim'])
    y_pred = model.predict(X_test)
    y_true = test_df['label']

    metrics = evaluate(y_true, y_pred, model_name="TF-IDF + LR (trained on news, tested on health claims)")
    plot_confusion_matrix(y_true, y_pred)