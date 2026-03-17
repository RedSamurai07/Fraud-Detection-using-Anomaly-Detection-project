import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
import joblib
import mlflow
import mlflow.sklearn
import os
import json
from datetime import datetime

# Environment-aware MLflow setup
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns')
EXPERIMENT_NAME = 'fraud-detection'

def engineer_features(df):
    """Clean feature engineering to avoid boolean attribute errors."""
    df_eng = df.copy()
    
    # Time-based features
    if 'Time' in df_eng.columns:
        df_eng['Hour'] = (df_eng['Time'] // 3600) % 24
        df_eng['Is_Night'] = np.where((df_eng['Hour'] >= 22) | (df_eng['Hour'] <= 5), 1, 0)
        df_eng['Is_Rush_Hour'] = np.where(df_eng['Hour'].between(7, 9) | df_eng['Hour'].between(17, 19), 1, 0)
    
    # Amount-based features
    if 'Amount' in df_eng.columns:
        df_eng['Amount_Log'] = np.log1p(df_eng['Amount'])
        df_eng['Amount_ZScore'] = (df_eng['Amount'] - df_eng['Amount'].mean()) / (df_eng['Amount'].std() + 1e-8)
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

def train(data_path='creditcard.csv', model_dir='models'):
    """Main training routine with MLflow tracking."""
    os.makedirs(model_dir, exist_ok=True)
    
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    print(f"[*] Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    print("[*] Engineering features...")
    df_eng = engineer_features(df)
    
    V_COLS = [f"V{i}" for i in range(1, 29)]
    OTHER_COLS = [
        'Amount_Log', 'Hour', 'Is_Night', 'Is_Rush_Hour',
        'Is_Round_Amount', 'Is_Small_Amount', 'Amount_ZScore',
        'V17_V14_interaction', 'V17_Amount_ratio', 'V14_V12_interaction'
    ]
    feature_cols = V_COLS + OTHER_COLS
    
    X = df_eng[feature_cols].fillna(0)
    y = df_eng['Class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    run_name = f"RF_Train_{datetime.now().strftime('%m%d_%H%M')}"
    with mlflow.start_run(run_name=run_name):
        params = {'n_estimators': 100, 'max_depth': 10, 'class_weight': 'balanced', 'random_state': 42}
        mlflow.log_params(params)
        
        clf = RandomForestClassifier(**params)
        clf.fit(X_train_scaled, y_train)
        
        # Predict & Evaluate
        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
        
        metrics = {
            'roc_auc': roc_auc_score(y_test, y_prob),
            'pr_auc': average_precision_score(y_test, y_prob)
        }
        mlflow.log_metrics(metrics)
        print(f"[+] Metrics: {metrics}")
        
        # Save locally for deployment
        joblib.dump(clf, os.path.join(model_dir, 'best_model.joblib'))
        joblib.dump(scaler, os.path.join(model_dir, 'scaler.joblib'))
        
        # Log to MLflow
        mlflow.sklearn.log_model(clf, "model")
        print(f"[+] Model saved to {model_dir}/")

if __name__ == "__main__":
    if os.path.exists('creditcard.csv'):
        train()
    else:
        print("Error: creditcard.csv not found.")
