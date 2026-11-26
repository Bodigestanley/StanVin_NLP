from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# --------------------------------------------------
# PATHS AND CONFIGURATION
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_PATH = ROOT / "data" / "processed" / "returniq_synthetic_15000.csv"
TFIDF_PATH = ROOT / "models" / "tfidf_returniq.joblib"
DISTILBERT_PATH = ROOT / "models" / "distilbert" / "final"
LABELS_PATH = ROOT / "models" / "distilbert" / "labels.json"

OUTPUT_DIR = ROOT / "reports" / "model_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "Sizing",
    "Damaged Item",
    "Wrong Product",
    "Late Delivery"
]

MAX_LENGTH = 96
BATCH_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# --------------------------------------------------
# LOAD TEST DATA
# --------------------------------------------------

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

required_columns = {"review_text", "label", "split"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Dataset must contain: {required_columns}"
    )

test_df = df[
    df["split"].astype(str).str.lower().str.strip() == "test"
].copy()

test_df["review_text"] = (
    test_df["review_text"].fillna("").astype(str).str.strip()
)

test_df["label"] = (
    test_df["label"].fillna("").astype(str).str.strip()
)

test_df = test_df[
    test_df["review_text"].ne("")
    & test_df["label"].isin(CLASSES)
].reset_index(drop=True)

if test_df.empty:
    raise ValueError("No valid test records found.")

y_true = test_df["label"].to_numpy()
texts = test_df["review_text"].tolist()

print("Test records:", len(test_df))
print("Device:", DEVICE)
print("\nTest class distribution:")
print(test_df["label"].value_counts().reindex(CLASSES))

# --------------------------------------------------
# LOAD AND RUN TF-IDF
# --------------------------------------------------

print("\nLoading TF-IDF model...")

bundle = joblib.load(TFIDF_PATH)

if not isinstance(bundle, dict):
    raise ValueError("Expected TF-IDF artifact to be a dictionary.")

vectorizer = bundle["vectorizer"]
classifier = bundle["classifier"]

print("Running TF-IDF predictions...")

X_tfidf = vectorizer.transform(texts)
tfidf_predictions = classifier.predict(X_tfidf)

# --------------------------------------------------
# LOAD AND RUN DISTILBERT
# --------------------------------------------------

print("\nLoading DistilBERT model...")

if not DISTILBERT_PATH.exists():
    raise FileNotFoundError(DISTILBERT_PATH)

if LABELS_PATH.exists():
    labels_data = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    saved_id2label = labels_data.get("id2label", {})
    saved_order = [
        saved_id2label.get(str(i), saved_id2label.get(i))
        for i in range(len(CLASSES))
    ]

    if saved_order != CLASSES:
        raise ValueError(
            f"DistilBERT label order mismatch: {saved_order}"
        )

tokenizer = AutoTokenizer.from_pretrained(
    str(DISTILBERT_PATH),
    local_files_only=True
)

model = AutoModelForSequenceClassification.from_pretrained(
    str(DISTILBERT_PATH),
    local_files_only=True
)

model.to(DEVICE)
model.eval()

print("Running DistilBERT predictions...")

distilbert_predictions = []

with torch.inference_mode():
    for start in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[start:start + BATCH_SIZE]

        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        outputs = model(**inputs)

        predicted_ids = torch.argmax(
            outputs.logits,
            dim=1
        ).cpu().numpy()

        distilbert_predictions.extend(
            [CLASSES[int(i)] for i in predicted_ids]
        )

distilbert_predictions = np.array(distilbert_predictions)

# --------------------------------------------------
# CALCULATE METRICS
# --------------------------------------------------

def evaluate_model(name, predictions):
    return {
        "model": name,
        "accuracy": float(
            accuracy_score(y_true, predictions)
        ),
        "precision_macro": float(
            precision_score(
                y_true, predictions,
                labels=CLASSES,
                average="macro",
                zero_division=0
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true, predictions,
                labels=CLASSES,
                average="macro",
                zero_division=0
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true, predictions,
                labels=CLASSES,
                average="macro",
                zero_division=0
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_true, predictions,
                labels=CLASSES,
                average="weighted",
                zero_division=0
            )
        ),
        "classification_report": classification_report(
            y_true,
            predictions,
            labels=CLASSES,
            output_dict=True,
            zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
            labels=CLASSES
        ).tolist()
    }


print("\nCalculating evaluation metrics...")

tfidf_results = evaluate_model(
    "TF-IDF + Logistic Regression",
    tfidf_predictions
)

distilbert_results = evaluate_model(
    "DistilBERT",
    distilbert_predictions
)

# --------------------------------------------------
# SAVE METRICS
# --------------------------------------------------

comparison = {
    "dataset": "returniq_synthetic_15000.csv",
    "dataset_type": "synthetic",
    "test_records": len(test_df),
    "class_order": CLASSES,
    "models": [
        tfidf_results,
        distilbert_results
    ],
    "warning": (
        "Results are measured on synthetic data and "
        "do not establish real-world performance."
    )
}

(OUTPUT_DIR / "comparison_metrics.json").write_text(
    json.dumps(comparison, indent=2),
    encoding="utf-8"
)

# --------------------------------------------------
# SAVE PREDICTIONS
# --------------------------------------------------

prediction_df = test_df.copy()

prediction_df["tfidf_prediction"] = tfidf_predictions
prediction_df["distilbert_prediction"] = distilbert_predictions

prediction_df["tfidf_correct"] = (
    prediction_df["label"] == prediction_df["tfidf_prediction"]
)

prediction_df["distilbert_correct"] = (
    prediction_df["label"] == prediction_df["distilbert_prediction"]
)

prediction_df.to_csv(
    OUTPUT_DIR / "test_predictions_comparison.csv",
    index=False
)

# --------------------------------------------------
# CONFUSION MATRIX COMPARISON
# --------------------------------------------------

cm_tfidf = np.array(tfidf_results["confusion_matrix"])
cm_distilbert = np.array(distilbert_results["confusion_matrix"])

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

matrices = [
    (cm_tfidf, "TF-IDF + Logistic Regression"),
    (cm_distilbert, "DistilBERT")
]

for ax, (matrix, title) in zip(axes, matrices):
    image = ax.imshow(matrix, cmap="Blues")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")

    ax.set_xticks(range(len(CLASSES)))
    ax.set_yticks(range(len(CLASSES)))

    ax.set_xticklabels(CLASSES, rotation=35, ha="right")
    ax.set_yticklabels(CLASSES)

    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(
                j, i, str(matrix[i, j]),
                ha="center",
                va="center",
                fontsize=11
            )

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

fig.suptitle(
    "ReturnIQ: TF-IDF vs DistilBERT Confusion Matrix",
    fontsize=16,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix_comparison.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# METRICS COMPARISON CHART
# --------------------------------------------------

metric_names = [
    "accuracy",
    "precision_macro",
    "recall_macro",
    "f1_macro",
    "f1_weighted"
]

model_names = [
    "TF-IDF",
    "DistilBERT"
]

values = [
    [tfidf_results[key] for key in metric_names],
    [distilbert_results[key] for key in metric_names]
]

x = np.arange(len(metric_names))
width = 0.35

fig, ax = plt.subplots(figsize=(11, 6))

bars1 = ax.bar(
    x - width / 2,
    values[0],
    width,
    label="TF-IDF"
)

bars2 = ax.bar(
    x + width / 2,
    values[1],
    width,
    label="DistilBERT"
)

ax.set_title("ReturnIQ Model Performance Comparison")
ax.set_ylabel("Score")
ax.set_xticks(x)
ax.set_xticklabels(metric_names, rotation=15)
ax.set_ylim(0, 1.1)
ax.legend()

ax.bar_label(bars1, fmt="%.3f", padding=3)
ax.bar_label(bars2, fmt="%.3f", padding=3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "metrics_comparison.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close()

# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------

summary = pd.DataFrame([
    {
        "Model": result["model"],
        "Accuracy": result["accuracy"],
        "Precision": result["precision_macro"],
        "Recall": result["recall_macro"],
        "Macro F1": result["f1_macro"],
        "Weighted F1": result["f1_weighted"]
    }
    for result in [tfidf_results, distilbert_results]
])

summary.to_csv(
    OUTPUT_DIR / "metrics_comparison.csv",
    index=False
)

print("\n" + "=" * 65)
print("RETURNIQ MODEL COMPARISON COMPLETED")
print("=" * 65)

print("\n", summary.to_string(index=False))

print("\nFiles generated:")
for path in sorted(OUTPUT_DIR.iterdir()):
    print("-", path.relative_to(ROOT))

print(
    "\nNOTE: These results use synthetic test data. "
    "They are development results, not real-world validation."
)
