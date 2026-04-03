import os
import joblib

from pipeline.dataset import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


DATA_PATH = "data/train.csv"
MODEL_DIR = "models/language"
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")
MODEL_PATH = os.path.join(MODEL_DIR, "baseline_lr_model.pkl")


def main():
    texts, labels = load_dataset(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=42,
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=12000,
        min_df=2,
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=1500,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train_vec, y_train)

    preds = model.predict(X_test_vec)
    print(classification_report(y_test, preds, target_names=["real/safe", "fake/risky"]))

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)

    print(f"Saved vectorizer to {VECTORIZER_PATH}")
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
