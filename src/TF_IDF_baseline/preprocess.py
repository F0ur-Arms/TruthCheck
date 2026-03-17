"""
tfidf_lr.py
────────────
Baseline classifier from the paper (Sections 3.3 + 3.5):
  - TF-IDF vectorization (max 5000 features as per paper)
  - Logistic Regression classifier
  - 80/20 train/test split
  - Full evaluation: accuracy, precision, recall, F1, confusion matrix

Expected CSV format:
    claim,label
    "Warm water cures diabetes",1
    "Exercise improves heart health",0

Labels: 0 = Real/True,  1 = Fake/Misinformation
"""

import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text  import TfidfVectorizer
from sklearn.linear_model             import LogisticRegression
from sklearn.model_selection          import train_test_split, cross_val_score
from sklearn.metrics                  import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)

from preprocess import preprocess


# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DATA_PATH      = "data/health_claims.csv"   # ← update to your CSV path
MODEL_DIR      = "models/"
TFIDF_MAX_FEAT = 5000                        # as specified in paper Section 3.3
TEST_SIZE      = 0.20                        # 80/20 split as per paper Section 3.5
RANDOM_STATE   = 42


def load_data(path: str):
    """Load and validate the dataset."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at: {path}")

    df = pd.read_csv(path)

    # Validate required columns
    if 'claim' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must have 'claim' and 'label' columns.")

    print(f"[Data] Loaded {len(df)} rows")
    print(f"[Data] Label distribution:\n{df['label'].value_counts()}\n")

    # Drop nulls
    df = df.dropna(subset=['claim', 'label'])
    df['label'] = df['label'].astype(int)

    return df


def evaluate(y_true, y_pred, model_name="Model"):
    """Print and return full evaluation metrics."""
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"\n{'─'*50}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'─'*50}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=['Real (0)', 'Fake (1)'])}")

    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def plot_confusion_matrix(y_true, y_pred, model_name="Model", save_path=None):
    """Plot and optionally save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Real', 'Fake'],
        yticklabels=['Real', 'Fake']
    )
    plt.title(f'Confusion Matrix — {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"[Plot] Confusion matrix saved to {save_path}")
    plt.show()


def train_tfidf_lr(df: pd.DataFrame):
    """
    Full TF-IDF + Logistic Regression pipeline.
    Steps:
      1. Preprocess text
      2. TF-IDF vectorization (max 5000 features)
      3. Train/test split (80/20)
      4. Train Logistic Regression
      5. Evaluate + plot
    """
    print("=" * 50)
    print("  TF-IDF + Logistic Regression Pipeline")
    print("=" * 50)

    # Step 1: Preprocess
    print("\n[Step 1] Preprocessing text...")
    df['clean_claim'] = df['claim'].apply(preprocess)

    X = df['clean_claim']
    y = df['label']

    # Step 2: Train/test split
    print("[Step 2] Splitting dataset (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"         Train: {len(X_train)} | Test: {len(X_test)}")

    # Step 3: TF-IDF vectorization
    print(f"[Step 3] TF-IDF vectorization (max_features={TFIDF_MAX_FEAT})...")
    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEAT, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec  = vectorizer.transform(X_test)

    # Step 4: Train Logistic Regression
    print("[Step 4] Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_vec, y_train)

    # Cross-validation score
    cv_scores = cross_val_score(model, X_train_vec, y_train, cv=5, scoring='f1_weighted')
    print(f"         5-Fold CV F1: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    # Step 5: Evaluate
    y_pred = model.predict(X_test_vec)
    metrics = evaluate(y_pred=y_pred, y_true=y_test, model_name="TF-IDF + Logistic Regression")

    # Plot confusion matrix
    os.makedirs("results", exist_ok=True)
    plot_confusion_matrix(y_test, y_pred, "TF-IDF + LR", save_path="results/cm_tfidf_lr.png")

    # Save model artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model,      os.path.join(MODEL_DIR, "baseline_lr_model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl"))
    print(f"\n[Saved] Model → {MODEL_DIR}baseline_lr_model.pkl")
    print(f"[Saved] Vectorizer → {MODEL_DIR}tfidf_vectorizer.pkl")

    return model, vectorizer, metrics


if __name__ == "__main__":
    df = load_data(DATA_PATH)
    model, vectorizer, metrics = train_tfidf_lr(df)