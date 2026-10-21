
from pathlib import Path
import json
from sklearn.metrics import confusion_matrix
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "SMSSpamCollection"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "message_classifier.joblib"

# 1. Load dataset: label followed by message text
df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "message"],
    encoding="utf-8",
    on_bad_lines="skip",
)

df = df.dropna(subset=["label", "message"])
df = df[df["label"].isin(["ham", "spam"])]
df["message"] = df["message"].astype(str)
df = df.drop_duplicates(subset=["message"])

print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df["label"].value_counts())

# 2. Split data before learning vocabulary
X_train, X_test, y_train, y_test = train_test_split(
    df["message"],
    df["label"],
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)

# 3. Build a single pipeline
model = Pipeline([
    ("tfidf", TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=20000,
        sublinear_tf=True,
    )),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )),
])

# 4. Train
model.fit(X_train, y_train)

# 5. Evaluate on unseen test messages
predictions = model.predict(X_test)

# Generate structured evaluation results
report = classification_report(
    y_test,
    predictions,
    labels=["ham", "spam"],
    target_names=["Legitimate", "Spam"],
    output_dict=True,
    zero_division=0,
)

cm = confusion_matrix(
    y_test,
    predictions,
    labels=["ham", "spam"],
)

evaluation = {
    "accuracy": float(accuracy_score(y_test, predictions)),
    "classification_report": report,
    "confusion_matrix": cm.tolist(),
    "labels": ["Legitimate", "Spam"],
    "test_size": int(len(y_test)),
}

# Save dashboard data
MODEL_DIR.mkdir(parents=True, exist_ok=True)

with open(BASE_DIR / "evaluation_metrics.json", "w") as f:
    json.dump(evaluation, f, indent=4)

print("Evaluation metrics saved to evaluation_metrics.json")


print("\nAccuracy:", round(accuracy_score(y_test, predictions), 4))
print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    labels=["ham", "spam"],
    zero_division=0,
))

ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    labels=["ham", "spam"],
    display_labels=["Legitimate", "Spam"],
)
plt.tight_layout()
plt.savefig(BASE_DIR / "confusion_matrix.png")
plt.close()

# 6. Save the entire pipeline
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")
print("Confusion matrix saved to: confusion_matrix.png")