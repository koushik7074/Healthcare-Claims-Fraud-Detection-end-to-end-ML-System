from fastapi import FastAPI
import pandas as pd
import joblib
import json
import os

from src.api.schemas import ProviderFeatures
from src.api.cache import redis_client, make_cache_key

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


# =====================================Creating End points ===================================================================
@app.post("/predict")
def predict_fraud(features: ProviderFeatures):

    payload = features.model_dump()
    cache_key = make_cache_key(payload)

    # 🔍 1. Check cache
    cached_response = redis_client.get(cache_key)
    if cached_response:
        print('Returning output from Redis cache!')
        return json.loads(cached_response)
    else:
        print('Returning output from model and adding output to cache')

    # ⏱️ 2. Compute prediction (cache miss)
    input_df = pd.DataFrame([payload])

    fraud_prob = model.predict_proba(input_df)[0][1]

    if fraud_prob < 0.3:
        risk = "LOW"
    elif fraud_prob < 0.6:
        risk = "MEDIUM"
    else:
        risk = "HIGH"

    fraud_pred = int(fraud_prob >= THRESHOLD)

    response = {
        "fraud_probability": round(fraud_prob, 4),
        "fraud_prediction": fraud_pred,
        "decision_threshold": THRESHOLD,
        "risk_level": risk
    }

    # 💾 3. Store in cache (15 min TTL)
    redis_client.setex(
        cache_key,
        900,
        json.dumps(response)
    )

    return response



@app.get("/health")
def health_check():
    return {"status": "ok"}

