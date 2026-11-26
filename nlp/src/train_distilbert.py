
"""
ReturnIQ - Fast Local DistilBERT Fine-Tuning
Dataset: data/processed/returniq_synthetic_15000.csv

Uses the existing train/test split from the supplied dataset.
This script creates validation data from the training split,
preserves the test split, trains multilingual DistilBERT and saves the model.
"""

from pathlib import Path
import json
import inspect
import time

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)


# ============================================================
# 1. CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data" / "processed" / "returniq_synthetic_15000.csv"

MODEL_ROOT = ROOT / "models" / "distilbert"
CHECKPOINT_DIR = MODEL_ROOT / "checkpoints"
OUTPUT_DIR = MODEL_ROOT / "final"

MODEL_NAME = "distilbert/distilbert-base-multilingual-cased"

CLASSES = [
    "Sizing",
    "Damaged Item",
    "Wrong Product",
    "Late Delivery",
]

LABEL2ID = {
    label: index
    for index, label in enumerate(CLASSES)
}

ID2LABEL = {
    index: label
    for index, label in enumerate(CLASSES)
}

SEED = 42

MAX_LENGTH = 96
EPOCHS = 1

TRAIN_BATCH_SIZE = 8
EVAL_BATCH_SIZE = 16

LEARNING_RATE = 3e-5

# CPU time-saving configuration.
# Set to None to use the entire training dataset.
CPU_TRAIN_LIMIT = 4000
CPU_TEST_LIMIT = 1500

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def stop(message):
    raise SystemExit("\nERROR: " + message)


def show_split(name, frame):

    print(f"{name}: {len(frame):,} rows")

    if frame.empty:
        stop(f"{name} split is empty.")

    missing = set(CLASSES) - set(frame["label"].unique())

    if missing:
        stop(
            f"{name} split is missing classes: "
            f"{sorted(missing)}"
        )


def stratified_limit(frame, limit):

    if limit is None or len(frame) <= limit:
        return frame.copy()

    sampled, _ = train_test_split(
        frame,
        train_size=limit,
        random_state=SEED,
        stratify=frame["label"],
    )

    return sampled.reset_index(drop=True)


# ============================================================
# 3. MAIN TRAINING FUNCTION
# ============================================================

def main():

    started = time.time()

    print("=" * 65)
    print("RETURNIQ - DISTILBERT TRAINING (FAST MODE)")
    print("=" * 65)

    print("Dataset:", DATA)
    print("Model:", MODEL_NAME)
    print("Device:", DEVICE)

    if DEVICE == "cpu":

        print(
            "CPU mode: one epoch and stratified "
            "training sample for faster completion."
        )

    else:

        print("GPU:", torch.cuda.get_device_name(0))

    # --------------------------------------------------------
    # LOAD DATASET
    # --------------------------------------------------------

    if not DATA.exists():
        stop(f"Dataset not found: {DATA}")

    df = pd.read_csv(DATA)

    print("\nOriginal dataset shape:", df.shape)

    if "review_text" in df.columns:
        text_col = "review_text"

    elif "text" in df.columns:
        text_col = "text"

    else:
        stop(
            "CSV must contain a 'review_text' "
            "or 'text' column."
        )

    if "label" not in df.columns:
        stop("CSV must contain a 'label' column.")

    keep = [text_col, "label"]

    if "split" in df.columns:
        keep.append("split")

    df = df[keep].copy()

    df.rename(
        columns={text_col: "text"},
        inplace=True,
    )

    # --------------------------------------------------------
    # CLEAN DATA
    # --------------------------------------------------------

    df["text"] = (
        df["text"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    df["label"] = (
        df["label"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    label_aliases = {
        "SIZING": "Sizing",
        "DAMAGED_ITEM": "Damaged Item",
        "WRONG_PRODUCT": "Wrong Product",
        "LATE_DELIVERY": "Late Delivery",
    }
    df["label"] = df["label"].replace(label_aliases)

    df = df[
        (df["text"].str.len() >= 3)
        & df["label"].isin(CLASSES)
    ].copy()

    df.reset_index(drop=True, inplace=True)

    if df.empty:
        stop("No valid records remain after cleaning.")

    print("\nCleaned dataset shape:", df.shape)

    print("\nClass distribution:")

    print(
        df["label"]
        .value_counts()
        .reindex(CLASSES, fill_value=0)
    )

    missing = set(CLASSES) - set(df["label"].unique())

    if missing:
        stop(
            f"Dataset is missing classes: {sorted(missing)}"
        )

    # --------------------------------------------------------
    # TRAIN / VALIDATION / TEST SPLIT
    # --------------------------------------------------------

    if "split" in df.columns:

        print("\nUsing existing dataset split column.")

        raw_split = (
            df["split"]
            .astype(str)
            .str.lower()
            .str.strip()
        )

        mapping = {
            "train": "train",
            "training": "train",
            "val": "validation",
            "valid": "validation",
            "validation": "validation",
            "test": "test",
            "testing": "test",
        }

        df["split"] = raw_split.map(mapping)

        if df["split"].isna().any():

            bad = sorted(
                raw_split[df["split"].isna()]
                .unique()
                .tolist()
            )

            stop(f"Unexpected split values: {bad}")

        original_train = df[
            df["split"] == "train"
        ].copy()

        test_df = df[
            df["split"] == "test"
        ].copy()

        existing_val = df[
            df["split"] == "validation"
        ].copy()

        if original_train.empty or test_df.empty:
            stop(
                "Existing split must contain "
                "both train and test rows."
            )

        if existing_val.empty:

            print(
                "No validation rows found. "
                "Creating validation from training rows."
            )

            train_df, val_df = train_test_split(
                original_train,
                test_size=0.125,
                random_state=SEED,
                stratify=original_train["label"],
            )

        else:

            train_df = original_train
            val_df = existing_val

    else:

        print(
            "\nNo split column found. "
            "Creating stratified splits."
        )

        trainval_df, test_df = train_test_split(
            df,
            test_size=0.20,
            random_state=SEED,
            stratify=df["label"],
        )

        train_df, val_df = train_test_split(
            trainval_df,
            test_size=0.125,
            random_state=SEED,
            stratify=trainval_df["label"],
        )

    # --------------------------------------------------------
    # VERIFY SPLITS
    # --------------------------------------------------------

    for name, frame in [
        ("Training", train_df),
        ("Validation", val_df),
        ("Testing", test_df),
    ]:

        show_split(name, frame)

    print("\nFull dataset split:")

    print("Training:", len(train_df))
    print("Validation:", len(val_df))
    print("Testing:", len(test_df))

    # --------------------------------------------------------
    # CPU OPTIMIZATION
    # --------------------------------------------------------

    if DEVICE == "cpu":

        train_used = stratified_limit(
            train_df,
            CPU_TRAIN_LIMIT,
        )

        test_used = stratified_limit(
            test_df,
            CPU_TEST_LIMIT,
        )

    else:

        train_used = train_df.copy()
        test_used = test_df.copy()

    print("\nRows used for this training run:")

    print("Training:", len(train_used))
    print("Validation:", len(val_df))
    print("Testing:", len(test_used))

    # --------------------------------------------------------
    # NUMERICAL LABELS
    # --------------------------------------------------------

    for frame in (
        train_used,
        val_df,
        test_used,
    ):

        frame["labels"] = (
            frame["label"]
            .map(LABEL2ID)
            .astype(int)
        )

        frame.reset_index(drop=True, inplace=True)

    # --------------------------------------------------------
    # LOAD TOKENIZER
    # --------------------------------------------------------

    print("\nLoading DistilBERT tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # TOKENIZATION
    # --------------------------------------------------------

    def tokenize(batch):

        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

    def make_dataset(frame):

        dataset = Dataset.from_pandas(
            frame[["text", "labels"]],
            preserve_index=False,
        )

        dataset = dataset.map(
            tokenize,
            batched=True,
        )

        dataset.set_format(
            type="torch",
            columns=[
                "input_ids",
                "attention_mask",
                "labels",
            ],
        )

        return dataset

    print("\nTokenizing datasets...")

    train_ds = make_dataset(train_used)
    val_ds = make_dataset(val_df)
    test_ds = make_dataset(test_used)

    print("Tokenization completed.")

    # --------------------------------------------------------
    # LOAD CLASSIFICATION MODEL
    # --------------------------------------------------------

    print("\nLoading DistilBERT model...")

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(CLASSES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    print("DistilBERT model loaded.")

    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    def compute_metrics(eval_pred):

        logits, labels = eval_pred

        predictions = np.argmax(
            logits,
            axis=-1,
        )

        return {
            "accuracy": accuracy_score(
                labels,
                predictions,
            ),

            "macro_f1": f1_score(
                labels,
                predictions,
                average="macro",
                zero_division=0,
            ),

            "weighted_f1": f1_score(
                labels,
                predictions,
                average="weighted",
                zero_division=0,
            ),
        }

    # --------------------------------------------------------
    # TRAINING ARGUMENTS
    # --------------------------------------------------------

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    params = {
        "output_dir": str(CHECKPOINT_DIR),
        "save_strategy": "no",
        "learning_rate": LEARNING_RATE,
        "per_device_train_batch_size": TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": EVAL_BATCH_SIZE,
        "num_train_epochs": EPOCHS,
        "weight_decay": 0.01,
        "report_to": "none",
        "logging_steps": 50,
        "seed": SEED,
        "fp16": DEVICE == "cuda",
        "dataloader_num_workers": 0,
    }

    ta_params = inspect.signature(
        TrainingArguments.__init__
    ).parameters

    if "eval_strategy" in ta_params:

        params["eval_strategy"] = "epoch"

    else:

        params["evaluation_strategy"] = "epoch"

    training_args = TrainingArguments(
        **params
    )

    # --------------------------------------------------------
    # INITIALIZE TRAINER
    # --------------------------------------------------------

    trainer_params = {
        "model": model,
        "args": training_args,
        "train_dataset": train_ds,
        "eval_dataset": val_ds,
        "compute_metrics": compute_metrics,
    }

    trainer_signature = inspect.signature(
        Trainer.__init__
    ).parameters

    if "processing_class" in trainer_signature:

        trainer_params["processing_class"] = tokenizer

    else:

        trainer_params["tokenizer"] = tokenizer

    trainer = Trainer(
        **trainer_params
    )

    # --------------------------------------------------------
    # TRAIN MODEL
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("STARTING DISTILBERT FINE-TUNING")
    print("=" * 65)

    trainer.train()

    print("\nTraining completed.")

    # --------------------------------------------------------
    # VALIDATION EVALUATION
    # --------------------------------------------------------

    print("\nEvaluating validation dataset...")

    validation_metrics = trainer.evaluate(
        eval_dataset=val_ds
    )

    print("\nValidation metrics:")
    print(validation_metrics)

    # --------------------------------------------------------
    # TEST EVALUATION
    # --------------------------------------------------------

    print("\nEvaluating independent test dataset...")

    test_output = trainer.predict(
        test_ds
    )

    test_predictions = np.argmax(
        test_output.predictions,
        axis=-1,
    )

    test_truth = test_used["labels"].to_numpy()

    accuracy = accuracy_score(
        test_truth,
        test_predictions,
    )

    macro_f1 = f1_score(
        test_truth,
        test_predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        test_truth,
        test_predictions,
        average="weighted",
        zero_division=0,
    )

    conf_matrix = confusion_matrix(
        test_truth,
        test_predictions,
        labels=list(range(len(CLASSES))),
    )

    report = classification_report(
        test_truth,
        test_predictions,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        output_dict=True,
        zero_division=0,
    )

    # --------------------------------------------------------
    # SAVE MODEL
    # --------------------------------------------------------

    print("\nSaving final DistilBERT model...")

    trainer.save_model(
        str(OUTPUT_DIR)
    )

    tokenizer.save_pretrained(
        str(OUTPUT_DIR)
    )

    # --------------------------------------------------------
    # SAVE METRICS
    # --------------------------------------------------------

    metrics = {
        "model": "Multilingual DistilBERT",
        "base_model": MODEL_NAME,

        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),

        "labels": CLASSES,

        "confusion_matrix": conf_matrix.tolist(),

        "classification_report": report,

        "validation_metrics": {
            key: float(value)
            for key, value in validation_metrics.items()
            if isinstance(
                value,
                (int, float, np.number),
            )
        },

        "n_rows": int(len(df)),

        "train_rows": int(len(train_used)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_used)),

        "full_train_rows_available": int(len(train_df)),
        "full_test_rows_available": int(len(test_df)),

        "dataset_path": str(DATA.relative_to(ROOT)),

        "dataset_type": "synthetic_multilingual",
        "languages": sorted(pd.read_csv(DATA)["language"].dropna().astype(str).unique().tolist()) if "language" in pd.read_csv(DATA, nrows=1).columns else [],

        "device": DEVICE,

        "epochs": EPOCHS,

        "max_length": MAX_LENGTH,

        "training_mode": (
            "fast_cpu_sample"
            if DEVICE == "cpu"
            else "full_gpu"
        ),

        "warning": (
            "Metrics are measured on synthetic data "
            "and do not establish real-world performance."
        ),
    }

    (OUTPUT_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    (MODEL_ROOT / "labels.json").write_text(
        json.dumps(
            {
                "label2id": LABEL2ID,
                "id2label": ID2LABEL,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------------

    elapsed = (
        time.time() - started
    ) / 60

    print("\n" + "=" * 65)
    print("DISTILBERT TRAINING COMPLETED")
    print("=" * 65)

    print(
        json.dumps(
            {
                "accuracy": float(accuracy),
                "macro_f1": float(macro_f1),
                "weighted_f1": float(weighted_f1),
                "train_rows_used": len(train_used),
                "validation_rows": len(val_df),
                "test_rows_used": len(test_used),
                "elapsed_minutes": round(elapsed, 2),
                "model_location": str(OUTPUT_DIR),
            },
            indent=2,
        )
    )

    print("\nModel saved at:")
    print(OUTPUT_DIR)

    print("\nMetrics saved at:")
    print(OUTPUT_DIR / "metrics.json")

    print(
        "\nNOTE: Dataset is synthetic. "
        "Treat these metrics as a development check, "
        "not real-world validation."
    )


if __name__ == "__main__":
    main()
