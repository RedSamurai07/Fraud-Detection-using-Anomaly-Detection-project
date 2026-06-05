import os
import pytest
from sklearn.linear_model import LogisticRegression
import joblib

# Automatically build a tiny mock model structure so CI doesn't skip predictions
@pytest.fixture(scope="session", autouse=True)
def create_dummy_model():
    models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    model_path = os.path.join(models_dir, 'best_model.joblib')
    scaler_path = os.path.join(models_dir, 'scaler.joblib')
    
    # Create and save a simple fake model if it doesn't exist in CI
    if not os.path.exists(model_path):
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        
        # Fake training data matching your 28 features
        X = np.random.rand(10, 28)
        y = np.random.randint(0, 2, 10)
        
        scaler = StandardScaler().fit(X)
        model = LogisticRegression().fit(scaler.transform(X), y)
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)