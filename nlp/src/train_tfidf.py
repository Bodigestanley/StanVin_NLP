"""Train the TF-IDF + Logistic Regression baseline.

Input: data/processed/return_comments.csv with text,label columns.
The included synthetic demo CSV is only for pipeline smoke tests.
"""
from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from preprocess import validate_frame

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "processed" / "return_comments.csv"
if not DATA.exists():
    raise SystemExit("Missing data/processed/return_comments.csv. Download candidates and manually annotate them first; synthetic demo data is not used for training.")
raw = pd.read_csv(DATA)
if "annotation_status" in raw.columns:
    raw = raw[raw["annotation_status"].astype(str).str.lower().eq("reviewed")]
raw = raw[raw["label"].notna() & raw["label"].astype(str).str.strip().ne("")]
df = validate_frame(raw)
counts = df.label.value_counts()
if df.label.nunique() != 4 or counts.min() < 4:
    raise ValueError("For stratified training, provide all four labels with at least 4 reviewed examples per class.")
Xtr, Xte, ytr, yte = train_test_split(df.clean_text, df.label, test_size=.25,
    random_state=42, stratify=df.label)
vec = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_features=50000,
    sublinear_tf=True, strip_accents="unicode")
xtr, xte = vec.fit_transform(Xtr), vec.transform(Xte)
clf = LogisticRegression(max_iter=1500, class_weight="balanced")
clf.fit(xtr,ytr)
pred = clf.predict(xte)
out = ROOT/"models"/"tfidf"
out.mkdir(parents=True,exist_ok=True)
joblib.dump(vec,out/"vectorizer.joblib")
joblib.dump(clf,out/"classifier.joblib")
labels = sorted(df.label.unique())
metrics = {"accuracy":accuracy_score(yte,pred),
    "macro_f1":f1_score(yte,pred,average="macro"),
    "weighted_f1":f1_score(yte,pred,average="weighted"),
    "report":classification_report(yte,pred,labels=labels,output_dict=True,zero_division=0),
    "confusion_matrix":confusion_matrix(yte,pred,labels=labels).tolist(),
    "labels":labels,"n_rows":len(df),"dataset_path":str(DATA.relative_to(ROOT))}
(out/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
print(json.dumps({k:v for k,v in metrics.items() if k not in ("report","confusion_matrix")},indent=2))
print("Training source: reviewed rows only. Check source/licence and annotation quality before reporting these metrics.")
