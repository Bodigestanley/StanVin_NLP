"""Create dataset profile tables and charts for reviewed return comments."""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
from preprocess import validate_frame

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/"data"/"processed"/"return_comments.csv"
if not DATA.exists():
    DATA=ROOT/"nlp"/"data"/"demo_returns.csv"
df=validate_frame(pd.read_csv(DATA))
out=ROOT/"reports"/"charts"; out.mkdir(parents=True,exist_ok=True)
df["character_length"]=df.text.str.len()
df["word_count"]=df.text.str.split().str.len()
summary={"rows":len(df),"unique_comments":int(df.text.nunique()),
 "class_counts":{str(k):int(v) for k,v in df.label.value_counts().items()},
 "missing_text":int(df.text.isna().sum()),"duplicate_texts":int(df.text.duplicated().sum()),
 "mean_characters":float(df.character_length.mean()),"median_words":float(df.word_count.median()),
 "dataset_file":str(DATA.relative_to(ROOT))}
(out/"dataset_profile.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
df.label.value_counts().reindex(["Sizing","Damaged Item","Wrong Product","Late Delivery"],fill_value=0).plot(kind="bar")
plt.title("Return reason class distribution"); plt.xlabel("Class"); plt.ylabel("Records")
plt.tight_layout(); plt.savefig(out/"class_distribution.png",dpi=160); plt.close()
df.character_length.plot(kind="hist",bins=20)
plt.title("Customer comment length distribution"); plt.xlabel("Characters"); plt.ylabel("Frequency")
plt.tight_layout(); plt.savefig(out/"comment_length.png",dpi=160); plt.close()
print(json.dumps(summary,indent=2))
print("EDA outputs saved to reports/charts. Synthetic demo data must not be used for final claims.")
