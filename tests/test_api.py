import os
import sys
import pytest
import pandas as pd
from fastapi.testclient import TestClient

# Ensure root path is structured cleanly for the runner imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from main import app, engineer_features

client = TestClient(app)

def test_health():
    """Verify health endpoint works cleanly and hits lines 81-89."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()

def test_engineer_features_logic():
    """Explicitly tracks feature engineering data mutation loops."""
    # Create a mock dataframe that contains the features your main.py transforms
    fake_data = {f"V{i}": [0.1] for i in range(1, 29)}
    fake_data["Time"] = [3600] # Trigger first conditional block
    fake_data["Amount"] = [100.0] # Trigger secondary numeric scaling blocks
    
    df = pd.DataFrame(fake_data)
    processed_df = engineer_features(df)
    assert isinstance(processed_df, pd.DataFrame)
    assert "Amount_Log" in processed_df.columns

def test_predict_success():
    """Executes the complete production inference pipeline."""
    # Build complete valid schema input payload
    payload = {f"V{i}": 0.0 for i in range(1, 29)}
    payload["Time"] = 0.0
    payload["Amount"] = 100.0
    
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "fraud_probability" in data

def test_predict_invalid_data():
    """Triggers exception pipelines by sending broken structural payloads."""
    # Empty payload forces validation block checks
    response = client.post("/predict", json={})
    assert response.status_code == 422