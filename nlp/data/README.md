# ReturnIQ dataset notes

`demo_returns.csv` is a small synthetic smoke-test fixture only. It is not real customer data and must not be used to claim real-world model performance.

## 30,000-row multi-source review corpus

Run `python src/prepare_multisource_30000.py` from the `nlp` directory. It samples 15,000 Amazon-based product reviews and 15,000 Flipkart product reviews from the public Hugging Face datasets listed below. The raw review text, platform, source dataset, rating and source metadata are preserved.

- Flipkart: https://huggingface.co/datasets/ml-hub/flipkart-reviews-dataset
- Amazon-based: https://huggingface.co/datasets/jhan21/amazon-reviews-balanced-text

These are product reviews, not verified return records. Platform sentiment labels are not ReturnIQ labels. The script leaves `label` empty; `keyword_candidate` is a rule-based hint only. Manually verify that a review describes a return reason and assign one of: Sizing, Damaged Item, Wrong Product, Late Delivery. Keep ambiguous/general product reviews out of the training file.

The generated 30k CSVs are created locally when the script is run; they are not embedded in the ZIP. Dataset cards and licensing/terms should be checked before redistribution or commercial use.
