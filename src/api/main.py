from fastapi import FastAPI
import pandas as pd
import joblib
import json
import os

from src.api.schemas import ProviderFeatures

#====================================Loading ML models and Configs=============================================================
app = FastAPI(title="Healthcare Fraud Detection API")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Load model
model = joblib.load(os.path.join(MODELS_DIR, "fraud_model.joblib"))

# Load config
with open(os.path.join(MODELS_DIR, "model_config.json")) as f:
    config = json.load(f)

THRESHOLD = config["decision_threshold"]


# =====================================Creating End points ====================================================================
@app.post("/predict")
def predict_fraud(features: ProviderFeatures):
    
    # Convert input to DataFrame
    input_df = pd.DataFrame([features.model_dump()])

    # Predict probability
    fraud_prob = model.predict_proba(input_df)[0][1]

    if fraud_prob < 0.3:
        risk = "LOW"
    elif fraud_prob < 0.6:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    # Apply threshold
    fraud_pred = int(fraud_prob >= THRESHOLD)


    return {
        "fraud_probability": round(fraud_prob, 4),
        "fraud_prediction": fraud_pred,
        "decision_threshold": THRESHOLD,
        "risk_level": risk
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}

