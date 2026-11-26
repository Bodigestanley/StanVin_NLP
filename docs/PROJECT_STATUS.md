# ReturnIQ v6 project status

- Preserves the Sage-themed React + TypeScript + Vite frontend.
- Preserves FastAPI and PostgreSQL configuration.
- Fixes TF-IDF inference to load the single scikit-learn pipeline artifact at `models/tfidf_returniq.joblib`.
- Baseline training writes that exact artifact and `models/baseline_metrics.json`.
- Dataset summary reads `data/processed/returniq_synthetic_15000.csv` (`review_text`, `label`).
- Batch endpoint retains a 500-comment per-request cap; frontend sends batches in chunks.
- DistilBERT remains optional and uses local artifacts only.
- Dataset is synthetic; evaluation metrics are not real-world performance estimates.
