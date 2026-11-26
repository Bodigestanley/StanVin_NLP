import re
import pandas as pd

LABELS = {"Sizing", "Damaged Item", "Wrong Product", "Late Delivery"}

def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text

def validate_frame(df: pd.DataFrame) -> pd.DataFrame:
    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df.text.str.len() >= 3) & df.label.isin(LABELS)]
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    df["clean_text"] = df["text"].map(clean_text)
    return df
