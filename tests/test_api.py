from fastapi.testclient import TestClient
import pytest
import os
import sys

# Ensure the project root is in the path for both IDE and CI
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import the app from main.py
try:
    from main import app
except ImportError:
    import sys
    sys.path.append(os.getcwd())
    from main import app

client = TestClient(app)

def test_health():
    """Verify the health endpoint works."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data

@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT_DIR, 'models', 'best_model.joblib')), 
                    reason="Model artifact not found")
def test_predict_success():
    """Verify that prediction works when model is present."""
    # We test with a minimal payload matching the Transaction schema
    payload = {f"V{i}": 0.0 for i in range(1, 29)}
    payload["Time"] = 0.0
    payload["Amount"] = 100.0
    
    response = client.post("/predict", json=payload)
    if response.status_code != 200:
        print(f"Error Response: {response.text}")
    # If model is present, it should return 200
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "fraud_probability" in data
