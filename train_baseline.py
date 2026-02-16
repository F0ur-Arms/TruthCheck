import joblib
import os
from pipeline.dataset import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 1. Load data
texts, labels = load_dataset("data/train.csv")

# 2. Split data
X_train, X_test, y_train, y_test = train_test_split(
    texts, labels, test_size=0.2, stratify=labels, random_state=42
)

# 3. Vectorize
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=15000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Train
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 5. Evaluate
preds = model.predict(X_test_vec)
print(classification_report(y_test, preds))

#adding save
os.makedirs('models/language', exist_ok=True)

# Save the Vectorizer (Crucial: You must use the SAME vectorizer for new inputs)
joblib.dump(vectorizer, 'models/language/tfidf_vectorizer.pkl')

# Save the Logistic Regression Model
joblib.dump(model, 'models/language/baseline_lr_model.pkl')

print("✅ Model and Vectorizer saved to models/language/")