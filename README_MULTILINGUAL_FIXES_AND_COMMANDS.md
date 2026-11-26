# ReturnIQ v12 — multilingual fixes and complete PowerShell commands

## Included fixes
- Correct Telugu and Hindi Unicode-script detection; English no longer gets labelled Telugu due to an incorrectly escaped Unicode regex.
- More conservative Romanized Telugu/Hindi detection. Generic English words like `order` and `size` alone do not trigger Hinglish.
- TF-IDF training text cleaning now matches backend inference cleaning and preserves Unicode letters.
- Both model scripts use the same four labels and the supplied fixed train/test split.
- How It Works now has a seven-stage interactive walkthrough with play/pause, next/previous, examples, outputs and caveats.

## Dataset
The expected file is `data/processed/returniq_synthetic_15000.csv`. Despite the legacy filename, the included file contains **1,500 synthetic multilingual examples**, balanced across the four classes. It is a development dataset, not 15,000 real customer records.

## Run from scratch (Windows PowerShell)
Extract the ZIP, open PowerShell in the extracted `ReturnIQ-main-v12` root folder, and keep your existing `.env` file in that root. The ZIP intentionally excludes `.env` so database credentials are not overwritten or shared.

### 1. One shared Python environment
Using one environment avoids installing the large PyTorch package twice.

```powershell
cd C:\Users\Perin\Downloads\ReturnIQ-main-v12
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r nlp\requirements.txt
python -m pip install -r backend\requirements.txt
```

### 2. Train TF-IDF + Logistic Regression
```powershell
python nlp\src\train_baseline.py
```
Artifacts: `models/tfidf_returniq.joblib` and `models/baseline_metrics.json`.

### 3. Train multilingual DistilBERT
```powershell
python nlp\src\train_distilbert.py
```
The first run downloads `distilbert/distilbert-base-multilingual-cased` and requires internet. CPU training is configured as a one-epoch run; it can still take time. A successful run saves tokenizer, model weights and metrics under `models/distilbert/final/`. The UI enables DistilBERT only when `config.json` and model weights exist there.

### 4. Start backend (from project root, same environment)
```powershell
python -m uvicorn app.main:app --app-dir backend --reload --host 127.0.0.1 --port 8000
```
API docs: `http://127.0.0.1:8000/docs`.

### 5. Start frontend (new PowerShell terminal)
```powershell
cd C:\Users\Perin\Downloads\ReturnIQ-main-v12\frontend
npm install
npm run dev
```
Open `http://localhost:5173`.

## Language checks
- `The shirt is too tight around my waist.` → English
- `నా షర్ట్ చాలా బిగుతుగా ఉంది.` → Telugu
- `मेरी शर्ट बहुत तंग है।` → Hindi
- `Mera order abhi tak nahi aaya.` → Hinglish / Romanized Hindi
- `Naa order inka raledu, chala late ayyindi.` → Romanized Telugu

Detection is heuristic, not machine translation; ambiguous mixed-language input can still be uncertain.

## Why model predictions can differ
TF-IDF + Logistic Regression learns word/bigram weights. DistilBERT uses contextual transformer representations. Both are trained on the same labelled data and fixed test split, but their learned decision boundaries differ. They are not expected to return identical predictions for every comment. Compare each model against reviewed ground-truth labels and the same held-out test set; do not treat one model's prediction as the other's ground truth.

The CSV `label` column in Batch Intelligence is the expected category for checking label agreement. Label agreement is not the same as test-set accuracy.

## Responsible use
The included synthetic dataset is suitable for checking the training and UI pipelines only. Replace it with representative, manually verified customer comments and labels before claiming real-world performance or automating return decisions. Keep human review for uncertain predictions.
