from fastapi.testclient import TestClient
import pytest
import os
import sys

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from main import app

client = TestClient(app)

SAMPLE_PAYLOAD = {f"V{i}": 0.0 for i in range(1, 29)}
SAMPLE_PAYLOAD["Time"] = 0.0
SAMPLE_PAYLOAD["Amount"] = 100.0


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_health_model_loaded():
    data = client.get("/health").json()
    assert "model_loaded" in data
    assert "scaler_loaded" in data


def test_predict_success():
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "fraud_probability" in data
    assert "is_fraud" in data


def test_predict_returns_valid_probability():
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    prob = response.json()["fraud_probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_prediction_is_int():
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert isinstance(response.json()["prediction"], int)


def test_predict_is_fraud_is_bool():
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert isinstance(response.json()["is_fraud"], bool)


def test_predict_missing_field_returns_422():
    bad_payload = {k: v for k, v in SAMPLE_PAYLOAD.items() if k != "Amount"}
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422


def test_predict_zero_amount():
    payload = {**SAMPLE_PAYLOAD, "Amount": 0.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_predict_night_transaction():
    payload = {**SAMPLE_PAYLOAD, "Time": 23 * 3600.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_predict_large_amount():
    payload = {**SAMPLE_PAYLOAD, "Amount": 25000.0}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200


def test_predict_class_matches_prediction():
    data = client.post("/predict", json=SAMPLE_PAYLOAD).json()
    assert data["prediction"] == data["class"]


def test_predict_is_fraud_matches_prediction():
    data = client.post("/predict", json=SAMPLE_PAYLOAD).json()
    assert data["is_fraud"] == (data["prediction"] == 1)


# ── Coverage boosters: error paths ───────────────────────────────────────────

def test_health_degraded_when_no_model(monkeypatch):
    import main as m
    original_model = m.model
    original_scaler = m.scaler
    m.model = None
    m.scaler = None
    response = client.get("/health")
    assert response.json()["status"] == "degraded"
    m.model = original_model
    m.scaler = original_scaler


def test_load_resources_missing_files(monkeypatch):
    import main as m
    monkeypatch.setattr(m, "MODEL_PATH", "/nonexistent/model.joblib")
    monkeypatch.setattr(m, "SCALER_PATH", "/nonexistent/scaler.joblib")
    m.model = None
    m.scaler = None
    result = m.load_resources()
    assert result is False


def test_predict_503_when_no_model(monkeypatch):
    import main as m
    monkeypatch.setattr(m, "MODEL_PATH", "/nonexistent/model.joblib")
    monkeypatch.setattr(m, "SCALER_PATH", "/nonexistent/scaler.joblib")
    m.model = None
    m.scaler = None
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 503


def test_load_resources_corrupted_file(tmp_path, monkeypatch):
    import main as m
    bad = tmp_path / "bad.joblib"
    bad.write_bytes(b"not-a-valid-joblib-file")
    monkeypatch.setattr(m, "MODEL_PATH", str(bad))
    monkeypatch.setattr(m, "SCALER_PATH", str(bad))
    m.model = None
    m.scaler = None
    result = m.load_resources()
    assert result is False


def test_predict_reloads_model_if_none(monkeypatch):
    """Covers the retry load_resources() branch inside /predict (line 99)."""
    import main as m
    original_model = m.model
    original_scaler = m.scaler
    m.model = None
    m.scaler = None
    response = client.post("/predict", json=SAMPLE_PAYLOAD)
    assert response.status_code == 200
    m.model = original_model
    m.scaler = original_scaler