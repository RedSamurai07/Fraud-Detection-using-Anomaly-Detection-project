from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import os
import numpy as np

# ── lifespan handler (replaces deprecated @app.on_event) ────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    load_resources()
    yield

app = FastAPI(title="Fraud Detection API", lifespan=lifespan)

MODEL_PATH  = os.path.join('models', 'best_model.joblib')
SCALER_PATH = os.path.join('models', 'scaler.joblib')

# Global references for resources
model  = None
scaler = None

class Transaction(BaseModel):
    Time: float
    V1: float; V2: float; V3: float; V4: float; V5: float
    V6: float; V7: float; V8: float; V9: float; V10: float
    V11: float; V12: float; V13: float; V14: float; V15: float
    V16: float; V17: float; V18: float; V19: float; V20: float
    V21: float; V22: float; V23: float; V24: float; V25: float
    V26: float; V27: float; V28: float
    Amount: float

def engineer_features(df):
    """Integrate same feature engineering as training."""
    df_eng = df.copy()
    
    # Time-based features
    if 'Time' in df_eng.columns:
        df_eng['Hour'] = (df_eng['Time'] // 3600) % 24
        df_eng['Is_Night'] = np.where((df_eng['Hour'] >= 22) | (df_eng['Hour'] <= 5), 1, 0)
        df_eng['Is_Rush_Hour'] = np.where(df_eng['Hour'].between(7, 9) | df_eng['Hour'].between(17, 19), 1, 0)
    
    # Amount-based features
    if 'Amount' in df_eng.columns:
        df_eng['Amount_Log'] = np.log1p(df_eng['Amount'])
        # Simplified ZScore for single-row inference to avoid std=0 errors
        # In production, these should be loaded as fixed parameters from training
        df_eng['Amount_ZScore'] = 0.0 
        df_eng['Is_Round_Amount'] = np.where(df_eng['Amount'] % 1 == 0, 1, 0)
        df_eng['Is_Small_Amount'] = np.where(df_eng['Amount'] < 1.0, 1, 0)
    
    # Interaction features for PCA components
    pca_cols = [c for c in df_eng.columns if c.startswith('V')]
    if 'V17' in pca_cols and 'V14' in pca_cols:
        df_eng['V17_V14_interaction'] = df_eng['V17'] * df_eng['V14']
    if 'V17' in pca_cols and 'Amount' in df_eng.columns:
        df_eng['V17_Amount_ratio'] = df_eng['V17'] / (df_eng['Amount'] + 1e-8)
    if 'V14' in pca_cols and 'V12' in pca_cols:
        df_eng['V14_V12_interaction'] = df_eng['V14'] * df_eng['V12']
    
    return df_eng

def load_resources():
    """Safety-critical loader for model and scaler."""
    global model, scaler
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        try:
            model  = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)
            print("Model and scaler loaded successfully.")
            return True
        except Exception as e:
            print(f"CRITICAL: Failed to load resources: {str(e)}")
            return False
    print(f"WARNING: model or scaler not found at {MODEL_PATH} / {SCALER_PATH}")
    return False

@app.get("/health")
def health():
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.post("/predict")
def predict(transaction: Transaction):
    # Retrieve globals with local references to avoid NoneType confusion in IDE
    m = model
    s = scaler
    
    if m is None or s is None:
        if not load_resources():
            raise HTTPException(status_code=503, detail="Prediction service is currently unavailable.")
        m = model
        s = scaler
        if m is None or s is None:
             raise HTTPException(status_code=503, detail="Failed to initialize model logic.")

    # Convert Pydantic object to DataFrame
    raw_df = pd.DataFrame([transaction.model_dump()])
    
    try:
        # Pre-process: Engineer features and select correct columns
        df_eng = engineer_features(raw_df)
        
        # Precise feature list matching training
        V_COLS = [f"V{i}" for i in range(1, 29)]
        OTHER_COLS = [
            'Amount_Log', 'Hour', 'Is_Night', 'Is_Rush_Hour',
            'Is_Round_Amount', 'Is_Small_Amount', 'Amount_ZScore',
            'V17_V14_interaction', 'V17_Amount_ratio', 'V14_V12_interaction'
        ]
        feature_cols = V_COLS + OTHER_COLS
        
        X_final = df_eng[feature_cols]

        # Scale and predict
        X_scaled = s.transform(X_final)
        pred     = int(m.predict(X_scaled)[0])
        prob_val = float(m.predict_proba(X_scaled)[0][1])
        
        # Round to 4 decimal places using string formatting for type safety
        prob_rounded = float("{:.4f}".format(prob_val))
        
        return {
            "prediction": pred,
            "is_fraud": bool(pred == 1),
            "fraud_probability": prob_rounded,
            "class": pred
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Respect PORT environment variable for Docker/Cloud compatibility
    port = int(os.getenv("PORT", 5000))
    uvicorn.run(app, host="0.0.0.0", port=port)
