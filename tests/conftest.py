import os
import pytest
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

@pytest.fixture(scope="session", autouse=True)
def create_dummy_model():
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(models_dir, 'best_model.joblib')
    scaler_path = os.path.join(models_dir, 'scaler.joblib')

    # 38 features = V1-V28 + 10 engineered features
    # Must match main.py's FEATURE_COLS exactly
    N_FEATURES = 38

    X = np.random.rand(50, N_FEATURES)
    y = np.array([0] * 48 + [1, 1])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=3, random_state=0)
    model.fit(X_scaled, y)

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)