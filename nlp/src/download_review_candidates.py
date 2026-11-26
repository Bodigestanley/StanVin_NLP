"""Download a bounded sample of real Amazon Fashion review text for manual annotation.

Important: Amazon Reviews 2023 has review text, NOT verified return-reason labels.
This script therefore writes candidate rows with an EMPTY `label` column. A human
must verify that each row actually describes a return and assign one of ReturnIQ's
four labels before model training.
"""
from pathlib import Path
import argparse
import re
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "raw" / "amazon_fashion_return_candidates.csv"
DATASET_ID = "McAuley-Lab/Amazon-Reviews-2023"
CONFIG = "raw_review_Amazon_Fashion"

# Broad discovery terms only. These are not automatic labels.
CANDIDATE_TERMS = re.compile(
    r"\b(return|returned|returning|refund|exchange|wrong size|too small|too big|"
    r"too large|doesn't fit|does not fit|damaged|defective|broken|wrong item|"
    r"wrong product|not what i ordered|late delivery|arrived late|delivery delay)\b",
    re.IGNORECASE,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-reviews", type=int, default=30000,
                        help="Maximum reviews to scan (default: 30000)")
    parser.add_argument("--max-candidates", type=int, default=3000,
                        help="Maximum matching review rows to save (default: 3000)")
    args = parser.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing dependency. Run: pip install -r requirements.txt") from exc

    print(f"Loading {DATASET_ID} / {CONFIG}")
    print("This is a real product-review dataset, not a return-reason dataset.")
    print("The script will save candidate comments for human review; labels remain blank.")

    # Follow the dataset authors' documented Hugging Face loading pattern.
    dataset_dict = load_dataset(DATASET_ID, CONFIG, trust_remote_code=True)
    split = dataset_dict["full"] if "full" in dataset_dict else dataset_dict[next(iter(dataset_dict.keys()))]

    rows = []
    scanned = 0
    for item in split:
        scanned += 1
        text = str(item.get("text") or "").strip()
        title = str(item.get("title") or "").strip()
        combined = (title + ". " + text).strip(" .")
        if len(text) >= 20 and CANDIDATE_TERMS.search(combined):
            rows.append({
                "text": combined,
                "label": "",  # MUST be manually annotated; never auto-fill this.
                "suggested_label": "",  # optional hint to be filled by reviewer
                "source": "Amazon Reviews 2023 - Amazon_Fashion",
                "source_record_id": str(item.get("parent_asin") or item.get("asin") or ""),
                "original_label": "not_provided_by_source",
                "annotation_status": "needs_human_review",
                "rating": item.get("rating", ""),
                "verified_purchase": item.get("verified_purchase", ""),
            })
            if len(rows) >= args.max_candidates:
                break
        if scanned >= args.max_reviews:
            break
        if scanned % 5000 == 0:
            print(f"Scanned {scanned:,}; candidates: {len(rows):,}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows, columns=[
        "text", "label", "suggested_label", "source", "source_record_id",
        "original_label", "annotation_status", "rating", "verified_purchase"
    ])
    frame.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"\nScanned: {scanned:,}")
    print(f"Candidate comments saved: {len(frame):,}")
    print(f"Output: {OUT}")
    print("\nNEXT: Open the CSV, verify each comment really describes a return, and fill `label` with exactly one of:")
    print("Sizing | Damaged Item | Wrong Product | Late Delivery")
    print("Set annotation_status=reviewed only after checking the text. Delete irrelevant rows.")
    print("Do not use suggested_label or rating as ground-truth labels.")
    print("Training scripts now require a reviewed, four-class CSV at data/processed/return_comments.csv.")


if __name__ == "__main__":
    main()
