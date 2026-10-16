"""Basic evaluation of the category classifier using a held-out test split."""
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

DATA_PATH = Path(__file__).parent / "data" / "qa_dataset.csv"
df = pd.read_csv(DATA_PATH).fillna("")
X = df["question"].astype(str) + " " + df["keywords"].astype(str)
y = df["category"].astype(str)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)
model = Pipeline([
    ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))),
    ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
])
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(f"Dataset size: {len(df)}")
print(f"Training examples: {len(X_train)}")
print(f"Testing examples: {len(X_test)}")
print(f"Accuracy: {accuracy_score(y_test, pred):.3f}")
print("\nClassification report:")
print(classification_report(y_test, pred, zero_division=0))
print("Confusion matrix (labels in alphabetical order):")
print(model.named_steps["classifier"].classes_)
print(confusion_matrix(y_test, pred, labels=model.named_steps["classifier"].classes_))
print("\nNote: with a small educational dataset, scores are only an initial estimate. Add more diverse labelled examples for a reliable evaluation.")
