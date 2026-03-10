from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# ── lifespan handler (replaces deprecated @app.on_event) ────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_resources()
    yield

app = FastAPI(title="Fraud Detection API", lifespan=lifespan)

MODEL_PATH  = 'models/best_model.joblib'
SCALER_PATH = 'models/scaler.joblib'

model  = None
scaler = None

class Transaction(BaseModel):
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float

def load_resources():
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Model and scaler loaded successfully.")
        return True
    print(f"WARNING: model or scaler not found at {MODEL_PATH} / {SCALER_PATH}")
    return False

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(transaction: Transaction):
    global model, scaler
    if model is None or scaler is None:
        if not load_resources():
            raise HTTPException(status_code=503, detail="Model not available.")
    df      = pd.DataFrame([transaction.model_dump()])
    X_scaled = scaler.transform(df)
    pred     = int(model.predict(X_scaled)[0])
    prob     = float(model.predict_proba(X_scaled)[0][1])
    return {"is_fraud": pred == 1, "fraud_probability": round(prob, 4), "class": pred}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
