from fastapi import FastAPI
from pydantic import BaseModel
import joblib, os

import pandas as pd

from src.med_extractor import extract_medications, llm_enabled

app = FastAPI(title="Clinical Notes Demo")

MODEL_PATH = "artifacts/model_xgb.joblib"

class NoteIn(BaseModel):
    note: str

class FeaturesIn(BaseModel):
    age: int
    sex: str          # "M" or "F"
    has_diabetes: int # 0 or 1
    has_hypertension: int # 0 or 1
    prior_admissions: int
    avg_glucose: float
    meds_count: int
    length_of_stay: int

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)

@app.get("/")
def root():
    return {"ok": True, "message": "Clinical Notes Demo API is running."}

@app.post("/extract_meds")
def extract_meds(body: NoteIn):
    meds = extract_medications(body.note)
    # free path uses regex; we can add LLM later
    return {"engine": "llm" if llm_enabled() else "regex", "medications": meds}


@app.post("/predict")
def predict(body: FeaturesIn):
    model = load_model()
    if model is None:
        return {"error": "Model not found. Run: python src\\train_model.py"}
    X = pd.DataFrame([body.dict()])
    proba = float(model.predict_proba(X)[0, 1])
    return {"readmission_risk": proba, "prediction": int(proba >= 0.5)}
