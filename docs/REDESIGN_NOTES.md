# ReturnIQ redesign notes

This revision upgrades the existing React + TypeScript + Vite frontend while preserving the FastAPI, PostgreSQL and local NLP architecture.

## Experience changes
- Adds a public marketing landing page with a CSS 3D animated hero, product problem statement, use cases, platform capabilities and working Explore Demo navigation.
- Reworks the application workspace into an executive-style overview, single prediction studio, CSV-to-intelligence workflow, dataset explorer, model laboratory and methodology page.
- CSV analysis now parses quoted fields and embedded commas/newlines, detects likely text/label columns, permits column selection, and processes every valid review in sequential chunks of 500. This removes the previous frontend `slice(0, 500)` truncation while respecting the existing backend 500-row request limit.
- The batch dashboard uses actual returned model predictions to calculate category counts/shares, per-category average confidence, low-confidence review counts, and label agreement only when a recognized ground-truth label is present.
- Clicking a category opens its review-level drill-down. Dataset Explorer supports search, category filtering and progressive loading.
- Exports include original review, optional ground-truth label, prediction, confidence, model and class scores.

## Important interpretation notes
- Dashboard analytics are descriptive of model predictions, not verified business causes.
- Confidence is a model score, not calibrated certainty.
- Suggested common signal words are illustrative category cues, not corpus-derived keyword frequency analysis.
- If the uploaded CSV has a `label` / ground-truth column, agreement is calculated only for recognized ReturnIQ labels. This does not make synthetic labels real customer evidence.
- The existing backend limits each batch API call to 500 comments. The frontend chunks larger files into multiple calls, so a 1,000-row file is processed in two requests.

## Run locally
1. Start the FastAPI API using the same command currently used by the local project.
2. In `frontend`, run `npm install` if dependencies are not installed.
3. Run `npm run dev`.
4. Open the Vite URL, explore the public landing page, then select Explore Demo.

## Verification status
The source changes are written. A full browser build and live PostgreSQL/API end-to-end test must still be run in the user's Windows environment because the local runtime here does not contain the project's installed frontend dependencies or user's database/model artifacts.
