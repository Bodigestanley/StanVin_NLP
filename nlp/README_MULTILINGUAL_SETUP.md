# ReturnIQ multilingual NLP setup

This project includes a **1,500-row synthetic multilingual development dataset** at
`data/processed/returniq_synthetic_15000.csv`. The filename is retained because the
training scripts expect it; it contains 1,500 rows, not 15,000. Its metrics are
development smoke-test results only, not real-world performance evidence.

Supported input groups in the dataset: English, Telugu, Hindi, Hinglish, and
Romanized Telugu/Hindi. Four labels: Sizing, Damaged Item, Wrong Product, Late Delivery.

## 1. Create the NLP environment (from the repository root)

```powershell
cd C:\Users\Perin\Downloads\ReturnIQ-main-v12
python -m venv nlp\.venv
.\nlp\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r nlp\requirements.txt
```

## 2. Train TF-IDF baseline

```powershell
python nlp\src\train_baseline.py
```

It saves `models/tfidf_returniq.joblib` and `models/baseline_metrics.json`.

## 3. Fine-tune multilingual DistilBERT

The training script uses `distilbert/distilbert-base-multilingual-cased`.
The first run downloads the pretrained tokenizer and weights and needs internet.
CPU training is configured for one epoch and a limited sample to keep the first
run manageable. It may still take a while on a CPU-only laptop.

```powershell
python nlp\src\train_distilbert.py
```

It saves the model and tokenizer under `models/distilbert/final/`, plus
`models/distilbert/final/metrics.json`. The API enables the DistilBERT option
only after the saved model files exist.

## 4. Install backend dependencies and run the API

Use a separate terminal. From the repository root:

```powershell
python -m venv backend\.venv
.\backend\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
python -m uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8000
```

## 5. Run the frontend

Use another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`.

## Notes

- TF-IDF preserves Unicode characters in the API preprocessing path.
- TF-IDF and DistilBERT use the same four UI category names and response schema.
- DistilBERT is a multilingual DistilBERT checkpoint, not the English-only
  `distilbert-base-uncased` checkpoint.
- Installing libraries does not train the model. Complete the DistilBERT training
  command successfully before expecting the UI option to become enabled.
- The sample dataset is synthetic. Replace or supplement it with manually
  verified, consented return-reason examples before claiming real-world accuracy.
