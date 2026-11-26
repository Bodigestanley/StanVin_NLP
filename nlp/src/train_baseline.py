"""Train ReturnIQ TF-IDF + Logistic Regression on the supplied labelled dataset.
Dataset: data/processed/returniq_synthetic_15000.csv (review_text, label, split).
All reported metrics describe this synthetic dataset only.
"""
from pathlib import Path
import re
import json
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "returniq_synthetic_15000.csv"
OUT = ROOT / "models"
if not DATA.exists():
    raise SystemExit(f"Dataset not found: {DATA}\nPlace the supplied CSV at data/processed/returniq_synthetic_15000.csv")
df = pd.read_csv(DATA)
text_col = "review_text" if "review_text" in df.columns else "text"
if not {text_col, "label"}.issubset(df.columns):
    raise SystemExit("CSV must contain review_text (or text) and label columns")
df = df.dropna(subset=[text_col, "label"]).copy()
label_aliases = {"SIZING": "Sizing", "DAMAGED_ITEM": "Damaged Item", "WRONG_PRODUCT": "Wrong Product", "LATE_DELIVERY": "Late Delivery"}
df["label"] = df["label"].astype(str).str.strip().replace(label_aliases)
df[text_col] = df[text_col].astype(str).str.strip()

def clean_text(value):
    value = str(value).lower().strip()
    value = re.sub(r"https?://\S+|www\.\S+", " ", value)
    value = re.sub(r"[^\w\s']", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value)
    return value.strip()

df[text_col] = df[text_col].map(clean_text)
df = df[df[text_col].str.len() >= 3]
if "split" in df.columns and {"train", "test"}.issubset(set(df["split"].astype(str).str.lower())):
    tr = df[df["split"].astype(str).str.lower() == "train"]
    te = df[df["split"].astype(str).str.lower() == "test"]
else:
    from sklearn.model_selection import train_test_split
    tr, te = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
if tr["label"].nunique() != 4 or te["label"].nunique() != 4:
    raise SystemExit("Both train and test splits must contain all four classes.")
vec = TfidfVectorizer(ngram_range=(1, 2), max_features=30000, sublinear_tf=True, min_df=2)
xtr = vec.fit_transform(tr[text_col].astype(str))
xte = vec.transform(te[text_col].astype(str))
clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
clf.fit(xtr, tr["label"].astype(str))
pred = clf.predict(xte)
OUT.mkdir(parents=True, exist_ok=True)
joblib.dump({"vectorizer": vec, "classifier": clf}, OUT / "tfidf_returniq.joblib")
labels = sorted(df["label"].astype(str).unique())
metrics = {"accuracy": float(accuracy_score(te["label"], pred)),
 "macro_f1": float(f1_score(te["label"], pred, average="macro")),
 "weighted_f1": float(f1_score(te["label"], pred, average="weighted")),
 "classification_report": classification_report(te["label"], pred, labels=labels, output_dict=True, zero_division=0),
 "confusion_matrix": confusion_matrix(te["label"], pred, labels=labels).tolist(),
 "labels": labels, "n_rows": int(len(df)), "train_rows": int(len(tr)), "test_rows": int(len(te)),
 "dataset_source": "synthetic"}
(OUT / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
print(json.dumps({k:v for k,v in metrics.items() if k not in ("classification_report", "confusion_matrix")}, indent=2))
print("NOTE: These metrics measure performance on synthetic data, not verified real customer returns.")
