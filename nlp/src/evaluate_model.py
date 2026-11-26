from pathlib import Path
import json, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import validate_frame
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"data"/"processed"/"return_comments.csv"
if not DATA.exists(): raise SystemExit("Prepare data/processed/return_comments.csv first.")
df=validate_frame(pd.read_csv(DATA))
_,test=train_test_split(df,test_size=.25,random_state=42,stratify=df.label)
vec=joblib.load(ROOT/"models/tfidf"/"vectorizer.joblib")
clf=joblib.load(ROOT/"models/tfidf"/"classifier.joblib")
pred=clf.predict(vec.transform(test.clean_text))
test=test.copy(); test["prediction"]=pred; test["correct"]=test.label==test.prediction
print(classification_report(test.label,pred,zero_division=0))
print("\nMisclassified examples:")
print(test.loc[~test.correct,["text","label","prediction"]].head(10).to_string(index=False))
out=ROOT/"reports"/"error_analysis"; out.mkdir(parents=True,exist_ok=True)
test.loc[~test.correct,["text","label","prediction"]].to_csv(out/"misclassified_examples.csv",index=False)
