"""Prepare a 30,000-row Amazon + Flipkart product-review corpus.

Sources contain product reviews and sentiment/rating metadata, NOT verified
return reasons. `label` is intentionally blank. `keyword_candidate` is only a
human-review hint and must never be treated as ground truth.
"""
from pathlib import Path
import argparse
import re
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_OUT = ROOT / "data" / "raw" / "returniq_multisource_reviews_30000.csv"
ANNOTATION_OUT = ROOT / "data" / "processed" / "returniq_annotation_template_30000.csv"

SOURCES = {
    "flipkart": "ml-hub/flipkart-reviews-dataset",
    "amazon": "jhan21/amazon-reviews-balanced-text",
}

PATTERNS = [
    ("Sizing", re.compile(r"\b(size|sizing|fit|fitting|too small|too large|too big|tight|loose|shorter than expected)\b", re.I)),
    ("Damaged Item", re.compile(r"\b(damaged|damage|broken|defective|faulty|cracked|not working|stopped working)\b", re.I)),
    ("Wrong Product", re.compile(r"\b(wrong item|wrong product|different product|incorrect item|not what i ordered|received .* instead)\b", re.I)),
    ("Late Delivery", re.compile(r"\b(late delivery|delivered late|delivery delay|arrived late|delayed delivery|not delivered on time)\b", re.I)),
]


def candidate_hint(text: str) -> str:
    hits = [label for label, pattern in PATTERNS if pattern.search(text)]
    return " | ".join(hits) if hits else ""


def load_all_splits(dataset_id: str):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing package. Run: pip install -r requirements.txt") from exc
    loaded = load_dataset(dataset_id)
    frames = []
    for split_name, split in loaded.items():
        frames.append(split.to_pandas().assign(_split=split_name))
    if not frames:
        raise RuntimeError(f"No splits returned by {dataset_id}")
    return pd.concat(frames, ignore_index=True)


def first_existing(frame, names, default=""):
    for name in names:
        if name in frame.columns:
            return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


def normalize_flipkart(frame):
    text = first_existing(frame, ["text", "Review", "review", "review_text"])
    title = first_existing(frame, ["Review", "review_title", "title"], "")
    # The HF Flipkart corpus has `Review` as a short heading and `text` as body.
    body = first_existing(frame, ["text", "review_text", "Review text", "review"])
    combined = (title.fillna("").astype(str).str.strip() + ". " + body.fillna("").astype(str).str.strip()).str.strip(" .")
    return pd.DataFrame({
        "text": combined,
        "rating": first_existing(frame, ["Rate", "rating", "stars"]),
        "platform_sentiment": first_existing(frame, ["labels", "Sentiment", "sentiment"]),
        "product_name": first_existing(frame, ["product_name", "product", "product_title"]),
        "source_platform": "Flipkart",
        "source_dataset": SOURCES["flipkart"],
        "source_record_id": "",
    })


def normalize_amazon(frame):
    body = first_existing(frame, ["text", "review_text", "review_body", "review"])
    title = first_existing(frame, ["title", "review_title", "summary"])
    combined = (title.fillna("").astype(str).str.strip() + ". " + body.fillna("").astype(str).str.strip()).str.strip(" .")
    return pd.DataFrame({
        "text": combined,
        "rating": first_existing(frame, ["rating", "stars", "overall"]),
        "platform_sentiment": first_existing(frame, ["sentiment_text", "sentiment"]),
        "product_name": "",
        "source_platform": "Amazon",
        "source_dataset": SOURCES["amazon"],
        "source_record_id": first_existing(frame, ["__index_level_0__", "review_id", "asin"], ""),
    })


def sample_reviews(frame: pd.DataFrame, target: int, seed: int) -> pd.DataFrame:
    frame = frame.copy()
    frame["text"] = frame["text"].fillna("").astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
    frame = frame[frame["text"].str.len().between(15, 5000)]
    frame = frame.drop_duplicates(subset=["text"])
    frame["rating_num"] = pd.to_numeric(frame["rating"], errors="coerce")
    # Prefer lower-rated reviews as complaint candidates; fill any shortfall randomly.
    low = frame[frame["rating_num"].le(3)]
    high = frame[~frame.index.isin(low.index)]
    chosen = low.sample(n=min(target, len(low)), random_state=seed) if len(low) else frame.iloc[0:0]
    remaining = target - len(chosen)
    if remaining > 0:
        pool = high if len(high) >= remaining else frame[~frame.index.isin(chosen.index)]
        extra = pool.sample(n=min(remaining, len(pool)), random_state=seed)
        chosen = pd.concat([chosen, extra], ignore_index=True)
    return chosen.drop(columns=["rating_num"], errors="ignore").sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=30000)
    parser.add_argument("--amazon", type=int, default=15000)
    parser.add_argument("--flipkart", type=int, default=15000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.amazon + args.flipkart != args.total:
        raise SystemExit("For this project, --amazon + --flipkart must equal --total.")

    print("Downloading Flipkart public review corpus...")
    fk = normalize_flipkart(load_all_splits(SOURCES["flipkart"]))
    print(f"Flipkart usable rows before sampling: {len(fk):,}")
    print("Downloading Amazon public review corpus...")
    am = normalize_amazon(load_all_splits(SOURCES["amazon"]))
    print(f"Amazon English usable rows before sampling: {len(am):,}")

    fk_sample = sample_reviews(fk, args.flipkart, args.seed)
    am_sample = sample_reviews(am, args.amazon, args.seed + 1)
    if len(fk_sample) < args.flipkart or len(am_sample) < args.amazon:
        raise SystemExit(
            "A source returned fewer unique reviews than requested. "
            f"Flipkart={len(fk_sample):,}/{args.flipkart:,}, Amazon={len(am_sample):,}/{args.amazon:,}. "
            "Do not duplicate rows to reach 30,000."
        )

    data = pd.concat([am_sample, fk_sample], ignore_index=True)
    data["keyword_candidate"] = data["text"].map(candidate_hint)
    data["label"] = ""  # Only a human annotator may set the final four-class label.
    data["original_label"] = data["platform_sentiment"].fillna("")
    data["annotation_status"] = "needs_human_review"
    data["record_type"] = "general_product_review_not_verified_return"
    data = data[[
        "text", "label", "keyword_candidate", "source_platform", "source_dataset",
        "source_record_id", "product_name", "rating", "platform_sentiment",
        "original_label", "annotation_status", "record_type"
    ]]
    data = data.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    RAW_OUT.parent.mkdir(parents=True, exist_ok=True)
    ANNOTATION_OUT.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(RAW_OUT, index=False, encoding="utf-8-sig")
    data.to_csv(ANNOTATION_OUT, index=False, encoding="utf-8-sig")

    print("\nDATASET PREPARED")
    print(f"Total rows: {len(data):,}")
    print("Platform counts:")
    print(data["source_platform"].value_counts().to_string())
    print("\nKeyword candidate hints (not ground-truth labels):")
    print(data["keyword_candidate"].value_counts().head(10).to_string())
    print(f"\nRaw corpus: {RAW_OUT}")
    print(f"Annotation template: {ANNOTATION_OUT}")
    print("\nIMPORTANT: `label` is blank intentionally. Review every row, verify it describes a return, and assign one approved class before training.")


if __name__ == "__main__":
    main()
