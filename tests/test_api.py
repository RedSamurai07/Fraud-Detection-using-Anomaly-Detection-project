from fastapi.testclient import TestClient
from main import app
import pytest

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_predict_schema_missing_fields():
    """Missing all V fields → 422 Unprocessable Entity."""
    response = client.post("/predict", json={"Amount": 100.0})
    assert response.status_code == 422


def test_predict_invalid_types():
    """String instead of float → 422."""
    response = client.post("/predict", json={"V1": "bad", "Amount": "nope"})
    assert response.status_code == 422
