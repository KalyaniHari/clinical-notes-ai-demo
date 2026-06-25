# Clinical Notes AI Demo

A small end-to-end demo that extracts medications from free-text clinical notes
and predicts 30-day hospital readmission from structured patient features.
Built on **synthetic data only** — no real patient information is used.

## What it does

1. **Medication extraction** — pulls medication mentions from a clinical note.
   Uses a free regex dictionary by default, and an optional LLM-based extractor
   when an `OPENAI_API_KEY` is provided.
2. **Readmission prediction** — an XGBoost model estimates the probability of a
   30-day readmission from patient features (age, sex, comorbidities, glucose,
   medication count, length of stay, etc.).
3. **API** — a FastAPI service exposes both capabilities over HTTP.

## Project Structure

```
.
├── api/
│   └── app.py            # FastAPI service (/extract_meds, /predict)
├── src/
│   ├── make_data.py      # Generate synthetic EHR dataset
│   ├── med_extractor.py  # Regex + optional LLM medication extraction
│   └── train_model.py    # Train and evaluate the XGBoost model
├── data/
│   └── synthetic_ehr.csv # Generated synthetic dataset
├── artifacts/
│   └── model_xgb.joblib  # Trained model
├── plots/                # ROC curve and feature importance plots
└── requirements.txt
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**Regenerate data and retrain (optional):**
```bash
python src/make_data.py
python src/train_model.py
```

**Run the API:**
```bash
uvicorn api.app:app --reload
```

Then call the endpoints:

```bash
# Extract medications from a note
curl -X POST http://localhost:8000/extract_meds \
  -H "Content-Type: application/json" \
  -d '{"note": "Patient started on metformin and lisinopril."}'

# Predict 30-day readmission
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"age":67,"sex":"M","has_diabetes":1,"has_hypertension":1,"prior_admissions":2,"avg_glucose":160.0,"meds_count":5,"length_of_stay":6}'
```

## Optional: LLM extraction

Set an OpenAI key to enable the LLM extractor (falls back to regex otherwise):

```bash
export OPENAI_API_KEY=your_key_here
pip install openai langchain langchain-openai
```

## Model

See `plots/roc_curve.png` and `plots/perm_importance.png` for evaluation
(ROC AUC and permutation feature importance) produced by `train_model.py`.

## Disclaimer

This is an educational demo on synthetic data. It is **not** a medical device
and must not be used for clinical decisions.

## License

Released under the [MIT License](LICENSE).
