from pathlib import Path
import re
import json
import joblib
from functools import lru_cache

import pandas as pd

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text as sql_text
from sqlalchemy.orm import Session

from .config import settings
from .database import Base, engine, get_db
from .db_models import Prediction
from .schemas import PredictRequest, BatchRequest


# ============================================================
# PATHS AND CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = (ROOT / settings.model_dir).resolve()

DISTILBERT_DIR = ROOT / "models" / "distilbert" / "final"

DATA_FILE = ROOT / "data" / "processed" / "returniq_synthetic_15000.csv"

TFIDF_PIPELINE = ROOT / "models" / "tfidf_returniq.joblib"


# ============================================================
# RETURN REASON CLASSES
# ============================================================

REASONS = {
    "Sizing": "Product size or fit did not meet expectations",
    "Damaged Item": "Product arrived damaged, defective or broken",
    "Wrong Product": "Received a product different from the one ordered",
    "Late Delivery": "Product arrived after the promised delivery date",
}

CLASSES = ["Sizing", "Damaged Item", "Wrong Product", "Late Delivery"]
LABEL_ALIASES = {
    "SIZING": "Sizing",
    "DAMAGED_ITEM": "Damaged Item",
    "WRONG_PRODUCT": "Wrong Product",
    "LATE_DELIVERY": "Late Delivery",
}


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="ReturnIQ API",
    version="2.0.0",
    description="Local NLP classification for e-commerce return reasons."
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        settings.frontend_origin,
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# DATABASE STARTUP
# ============================================================

@app.on_event("startup")
def startup():
    try:
        Base.metadata.create_all(bind=engine)

        with engine.connect() as conn:
            conn.execute(sql_text("SELECT 1"))

        print("PostgreSQL connection ready.")

    except Exception as exc:
        print(
            f"PostgreSQL is not ready; API remains available. "
            f"Detail: {exc}"
        )


# ============================================================
# TF-IDF MODEL LOADING
# ============================================================

def get_tfidf():

    if not TFIDF_PIPELINE.exists():
        return None

    try:
        artifact = joblib.load(TFIDF_PIPELINE)

        print(
            "TF-IDF artifact loaded successfully:",
            type(artifact).__name__
        )

        return artifact

    except Exception as exc:
        print(f"Could not load TF-IDF pipeline: {exc}")
        return None


# ============================================================
# DISTILBERT MODEL LOADING
# ============================================================

from functools import lru_cache

@lru_cache(maxsize=1)
def get_distilbert():
    try:
        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification
        )

        tokenizer = AutoTokenizer.from_pretrained(
            str(DISTILBERT_DIR),
            local_files_only=True
        )

        model = AutoModelForSequenceClassification.from_pretrained(
            str(DISTILBERT_DIR),
            local_files_only=True
        )

        model.eval()

        print("✅ DistilBERT loaded successfully")

        return tokenizer, model, torch

    except Exception as exc:
        print(f"❌ DistilBERT loading failed: {repr(exc)}")
        raise RuntimeError(
            f"DistilBERT loading failed: {exc}"
        ) from exc

# ============================================================
# TEXT CLEANING
# ============================================================


def detect_language(text: str) -> str:
    """Detect writing system first, then use conservative romanized-language cues."""
    value = str(text or "").strip()
    if not value:
        return "Unknown"
    if re.search(r"[\u0C00-\u0C7F]", value):
        return "Telugu"
    if re.search(r"[\u0900-\u097F]", value):
        return "Hindi"
    words = set(re.findall(r"[a-z]+", value.lower()))
    telugu_roman = {
        "naaku", "naku", "naa", "vachindi", "vachina", "vachadu",
        "ledu", "kaaledu", "ayyindi", "chesa", "chesanu", "cheyyandi",
        "saripoledu", "pampincharu", "kavali", "undhi", "undi",
        "avvatledu", "inka", "chala", "enduku", "meeru"
    }
    hindi_roman = {
        "mujhe", "mera", "meri", "maine", "mangaaya", "mangaya",
        "nahi", "nahin", "hua", "hui", "chahiye", "bahut", "kyun",
        "bhejo", "bheja", "mila", "mili", "aaya", "aayi", "abhi",
        "tak", "kharab", "toota", "chhota", "bada"
    }
    te_score = len(words & telugu_roman)
    hi_score = len(words & hindi_roman)
    if te_score >= 1 and te_score > hi_score:
        return "Romanized Telugu"
    if hi_score >= 1:
        return "Hinglish / Romanized Hindi"
    return "English"

def clean_for_tfidf(comment: str) -> str:

    value = str(comment).lower().strip()

    value = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        value
    )

    value = re.sub(
        r"[^\w\s']",
        " ",
        value
    )

    value = re.sub(
        r"\s+",
        " ",
        value
    )

    return value.strip()


# ============================================================
# TF-IDF PREDICTION
# ============================================================

def predict_tfidf(comment: str):

    artifact = get_tfidf()

    if artifact is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"TF-IDF model not found at {TFIDF_PIPELINE}. "
                "Run python nlp/src/train_baseline.py"
            )
        )

    # Case 1: Training script saved a dictionary
    if isinstance(artifact, dict):

        vectorizer = artifact.get("vectorizer")
        classifier = artifact.get("classifier")

        if vectorizer is None or classifier is None:
            raise HTTPException(
                status_code=500,
                detail=(
                    "TF-IDF artifact must contain "
                    "'vectorizer' and 'classifier' keys."
                )
            )

        cleaned_text = clean_for_tfidf(comment)

        features = vectorizer.transform([cleaned_text])

        scores = classifier.predict_proba(features)[0]

        model_classes = classifier.classes_

    # Case 2: Artifact is a complete sklearn Pipeline
    else:

        cleaned_text = clean_for_tfidf(comment)

        scores = artifact.predict_proba([cleaned_text])[0]

        model_classes = artifact.classes_

    probabilities = {
        LABEL_ALIASES.get(str(category), str(category)): float(score)
        for category, score in zip(model_classes, scores)
    }

    predicted_category = max(
        probabilities,
        key=probabilities.get
    )

    confidence = probabilities[predicted_category]

    return {
        "predicted_category": predicted_category,
        "predicted_reason": REASONS.get(
            predicted_category,
            "Reason not available"
        ),
        "confidence": confidence,
        "probabilities": probabilities,
        "model_used": "TF-IDF + Logistic Regression",
        "detected_language": detect_language(comment),
        "demo_fallback": False,
        "explanation": (
            "TF-IDF word/bigram features classified "
            "by Logistic Regression."
        )
    }


# ============================================================
# DISTILBERT PREDICTION
# ============================================================
def predict_distilbert(text: str):
    try:
        tokenizer, model, torch = get_distilbert()

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=160
        )

        with torch.no_grad():
            outputs = model(**inputs)
            probabilities = torch.softmax(
                outputs.logits,
                dim=-1
            )[0]

        predicted_index = int(torch.argmax(probabilities).item())

        raw_label = model.config.id2label[predicted_index]
        label = LABEL_ALIASES.get(str(raw_label), str(raw_label))
        confidence = float(probabilities[predicted_index].item())
        all_scores = {
            LABEL_ALIASES.get(str(model.config.id2label[i]), str(model.config.id2label[i])):
                float(probabilities[i].item())
            for i in range(len(probabilities))
        }

        return {
            "predicted_category": label,
            "predicted_reason": REASONS.get(label, "Reason not available"),
            "confidence": confidence,
            "probabilities": all_scores,
            "model_used": "Multilingual DistilBERT",
            "detected_language": detect_language(text),
            "demo_fallback": False,
            "explanation": "Contextual prediction from fine-tuned multilingual DistilBERT."
        }

    except Exception as exc:
        print(f"DistilBERT prediction error: {repr(exc)}")
        raise HTTPException(
            status_code=503,
            detail=f"DistilBERT prediction failed: {str(exc)}"
        )
# ============================================================
# MAIN PREDICTION FUNCTION
# ============================================================

def predict_one(comment: str, model_name: str):

    if not comment or len(comment.strip()) < 3:
        raise HTTPException(
            status_code=422,
            detail="Text must contain at least 3 characters."
        )

    if model_name == "distilbert":
        return predict_distilbert(comment)

    if model_name == "tfidf":
        return predict_tfidf(comment)

    raise HTTPException(
        status_code=400,
        detail="Invalid model. Choose 'tfidf' or 'distilbert'."
    )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "name": "ReturnIQ",
        "version": "2.0.0",
        "docs": "/docs"
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/api/health")
def health():

    try:
        with engine.connect() as conn:
            conn.execute(sql_text("SELECT 1"))

        db_ok = True

    except Exception:
        db_ok = False

    pipeline = get_tfidf()

    distilbert_available = (
        DISTILBERT_DIR.exists()
        and (DISTILBERT_DIR / "config.json").exists()
        and (
            (DISTILBERT_DIR / "model.safetensors").exists()
            or (DISTILBERT_DIR / "pytorch_model.bin").exists()
        )
    )

    return {
        "status": "ok",
        "database_connected": db_ok,
        "tfidf_model_available": pipeline is not None,
        "distilbert_model_available": distilbert_available,
        "classes": CLASSES
    }


# ============================================================
# RETURN CLASSES
# ============================================================

@app.get("/api/classes")
def classes():

    return {
        "classes": CLASSES
    }


# ============================================================
# SINGLE PREDICTION
# ============================================================

@app.post("/api/predict")
def predict(
    payload: PredictRequest,
    db: Session = Depends(get_db)
):

    result = predict_one(
        payload.text,
        payload.model
    )

    try:
        row = Prediction(
            input_text=payload.text,
            predicted_category=result["predicted_category"],
            confidence=result["confidence"],
            model_name=result["model_used"],
            demo_fallback=False
        )

        db.add(row)
        db.commit()

    except Exception as exc:
        db.rollback()

        print(
            f"Prediction persistence failed: {exc}"
        )

    return result


# ============================================================
# BATCH PREDICTION
# ============================================================

@app.post("/api/predict/batch")
def predict_batch(payload: BatchRequest):

    results = []

    for index, comment in enumerate(payload.texts):

        try:
            result = predict_one(
                comment,
                payload.model
            )

            result["row_number"] = index + 1
            results.append(result)

        except HTTPException as exc:
            results.append({
                "row_number": index + 1,
                "predicted_category": None,
                "predicted_reason": None,
                "confidence": 0,
                "probabilities": {},
                "model_used": payload.model,
                "error": exc.detail
            })

        except Exception as exc:
            results.append({
                "row_number": index + 1,
                "predicted_category": None,
                "predicted_reason": None,
                "confidence": 0,
                "probabilities": {},
                "model_used": payload.model,
                "error": str(exc)
            })

    successful = sum(
        1 for item in results
        if item.get("predicted_category") is not None
    )

    return {
        "total": len(payload.texts),
        "successful": successful,
        "failed": len(payload.texts) - successful,
        "results": results
    }


# ============================================================
# RECENT PREDICTIONS
# ============================================================

@app.get("/api/predictions/recent")
def recent_predictions(
    limit: int = 20,
    db: Session = Depends(get_db)
):

    try:
        rows = (
            db.query(Prediction)
            .order_by(Prediction.created_at.desc())
            .limit(min(max(limit, 1), 500))
            .all()
        )

        return {
            "items": [
                {
                    "id": row.id,
                    "text": row.input_text,
                    "category": row.predicted_category,
                    "confidence": row.confidence,
                    "model": row.model_name,
                    "created_at": row.created_at.isoformat()
                }
                for row in rows
            ]
        }

    except Exception as exc:
        db.rollback()

        return {
            "items": [],
            "note": f"Database unavailable: {exc}"
        }


# ============================================================
# DATASET SUMMARY
# ============================================================

@app.get("/api/dataset/summary")
def dataset_summary():

    if not DATA_FILE.exists():
        return {
            "records": 0,
            "class_count": 0,
            "source": (
                "Synthetic dataset not found; place it at "
                "data/processed/returniq_synthetic_15000.csv"
            ),
            "verified": False,
            "distribution": []
        }

    try:
        df = pd.read_csv(DATA_FILE)

        text_col = (
            "review_text"
            if "review_text" in df.columns
            else "text"
        )

        label_col = "label"

        if not {text_col, label_col}.issubset(df.columns):
            raise ValueError(
                "CSV must contain review_text (or text) and label columns"
            )

        counts = df[label_col].value_counts().to_dict()

        total = len(df)

        return {
            "records": total,
            "class_count": len(counts),
            "source": (
                "data/processed/returniq_synthetic_15000.csv"
            ),
            "verified": False,
            "data_source": "synthetic",
            "distribution": [
                {
                    "label": str(key),
                    "count": int(value),
                    "share": value / total if total else 0
                }
                for key, value in counts.items()
            ]
        }

    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not read dataset: {exc}"
        )


# ============================================================
# DATASET QUALITY ANALYSIS
# ============================================================

@app.get("/api/dataset/quality")
def dataset_quality():

    if not DATA_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Dataset file not found."
        )

    try:
        df = pd.read_csv(DATA_FILE)

        total_records = len(df)

        missing_values = {
            str(column): int(count)
            for column, count in df.isnull().sum().items()
        }

        duplicate_records = int(df.duplicated().sum())

        label_distribution = {}

        if "label" in df.columns:
            label_distribution = {
                str(label): int(count)
                for label, count in
                df["label"].value_counts().items()
            }

        source_distribution = {}

        if "data_source" in df.columns:
            source_distribution = {
                str(source): int(count)
                for source, count in
                df["data_source"].value_counts().items()
            }

        annotation_distribution = {}

        if "annotation_status" in df.columns:
            annotation_distribution = {
                str(status): int(count)
                for status, count in
                df["annotation_status"].value_counts().items()
            }

        return {
            "total_records": total_records,
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "missing_values": missing_values,
            "duplicate_records": duplicate_records,
            "label_distribution": label_distribution,
            "source_distribution": source_distribution,
            "annotation_distribution": annotation_distribution,
            "verified": False,
            "dataset_type": "Synthetic"
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Dataset quality analysis failed: {exc}"
        )


# ============================================================
# DATASET RECORD EXPLORER
# ============================================================

@app.get("/api/dataset/records")
def dataset_records(
    search: str = "",
    label: str = "",
    page: int = 1,
    page_size: int = 20
):

    if not DATA_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Dataset file not found."
        )

    if page < 1:
        raise HTTPException(
            status_code=422,
            detail="Page must be at least 1."
        )

    if page_size < 1 or page_size > 100:
        raise HTTPException(
            status_code=422,
            detail="Page size must be between 1 and 100."
        )

    try:
        df = pd.read_csv(DATA_FILE)

        if "review_text" not in df.columns:
            raise ValueError(
                "Dataset must contain review_text column."
            )

        if search.strip():
            df = df[
                df["review_text"]
                .fillna("")
                .astype(str)
                .str.contains(
                    search.strip(),
                    case=False,
                    regex=False
                )
            ]

        if label.strip() and "label" in df.columns:
            df = df[
                df["label"].fillna("").astype(str).str.lower()
                == label.strip().lower()
            ]

        total_filtered = len(df)

        start = (page - 1) * page_size
        end = start + page_size

        records = []

        for _, row in df.iloc[start:end].iterrows():

            record = {}

            for column, value in row.items():

                if pd.isna(value):
                    record[column] = None

                elif hasattr(value, "item"):
                    record[column] = value.item()

                else:
                    record[column] = value

            records.append(record)

        return {
            "total": total_filtered,
            "page": page,
            "page_size": page_size,
            "total_pages": (
                (total_filtered + page_size - 1) // page_size
            ),
            "records": records
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not retrieve dataset records: {exc}"
        )


# ============================================================
# LATEST MODEL EVALUATION
# ============================================================

@app.get("/api/evaluation/latest")
def evaluation_latest():

    path = ROOT / "models" / "baseline_metrics.json"

    transformer_metrics = (
        ROOT / "models" / "distilbert" / "final" / "metrics.json"
    )

    if transformer_metrics.exists():
        path = transformer_metrics

    if not path.exists():
        return {
            "available": False
        }

    try:
        metrics = json.loads(
            path.read_text(encoding="utf-8")
        )

        return {
            "available": True,
            "accuracy": metrics.get("accuracy", 0),
            "macro_f1": metrics.get("macro_f1", 0),
            "weighted_f1": metrics.get("weighted_f1", 0),
            "confusion_matrix": metrics.get("confusion_matrix"),
            "labels": metrics.get("labels", []),
            "report": metrics.get(
                "classification_report",
                metrics.get("report", {})
            )
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read evaluation metrics: {exc}"
        )
