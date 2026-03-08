import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)
import joblib
import mlflow
import mlflow.sklearn
import os
import json
import pkg_resources
from datetime import datetime

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns')
EXPERIMENT_NAME = 'fraud-detection'

def engineer_features(df):
    """Feature engineering pipeline"""
    df_eng = df.copy()
    
    # Time-based features
    df_eng['Hour'] = (df_eng['Time'] // 3600) % 24
    df_eng['Is_Night'] = ((df_eng['Hour'] >= 22) | (df_eng['Hour'] <= 5)).astype(int)
    df_eng['Is_Rush_Hour'] = ((df_eng['Hour'].between(7, 9)) | 
                              (df_eng['Hour'].between(17, 19))).astype(int)
    
    # Amount-based features
    df_eng['Amount_Log'] = np.log1p(df_eng['Amount'])
    df_eng['Amount_ZScore'] = (df_eng['Amount'] - df_eng['Amount'].mean()) / df_eng['Amount'].std()
    df_eng['Is_Round_Amount'] = (df_eng['Amount'] % 1 == 0).astype(int)
    df_eng['Is_Small_Amount'] = (df_eng['Amount'] < 1.0).astype(int)
    
    # Interaction features
    if 'V17' in df_eng.columns and 'V14' in df_eng.columns:
        df_eng['V17_V14_interaction'] = df_eng['V17'] * df_eng['V14']
    
    if 'V17' in df_eng.columns and 'Amount' in df_eng.columns:
        df_eng['V17_Amount_ratio'] = df_eng['V17'] / (df_eng['Amount'] + 1e-8)
    
    if 'V14' in df_eng.columns and 'V12' in df_eng.columns:
        df_eng['V14_V12_interaction'] = df_eng['V14'] * df_eng['V12']
    
    return df_eng

def train_model(data_path, model_output_dir='./models', experiment_name=EXPERIMENT_NAME):
    """
    Train fraud detection model with MLflow tracking
    
    Args:
        data_path: Path to creditcard.csv
        model_output_dir: Directory to save model artifacts
        experiment_name: MLflow experiment name
    """
    
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_name)
    
    print("="*60)
    print("FRAUD DETECTION MODEL TRAINING WITH MLFLOW")
    print("="*60)
    
    # Load data
    print(f"\n1. Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    print(f"   Dataset shape: {df.shape}")
    print(f"   Fraud rate: {df['Class'].mean()*100:.4f}%")
    
    # Feature engineering
    print("\n2. Engineering features...")
    df_eng = engineer_features(df)
    
    # Define feature columns
    feature_cols = ([c for c in df_eng.columns if c.startswith('V')] +
                   ['Amount_Log', 'Hour', 'Is_Night', 'Is_Rush_Hour',
                    'Is_Round_Amount', 'Is_Small_Amount', 'Amount_ZScore',
                    'V17_V14_interaction', 'V17_Amount_ratio', 'V14_V12_interaction'])
    
    X = df_eng[feature_cols].fillna(0)
    y = df_eng['Class']
    
    print(f"   Feature count: {len(feature_cols)}")
    
    # Train-test split
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"   Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Scale features
    print("\n4. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"rf_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log parameters
        print("\n5. Training Random Forest model...")
        
        model_params = {
            'n_estimators': 100,
            'max_depth': 20,
            'min_samples_split': 10,
            'min_samples_leaf': 4,
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        }
        
        mlflow.log_params(model_params)
        mlflow.log_param('feature_count', len(feature_cols))
        mlflow.log_param('train_size', len(X_train))
        mlflow.log_param('test_size', len(X_test))
        mlflow.log_param('fraud_rate', df['Class'].mean())
        
        # Train model
        model = RandomForestClassifier(**model_params)
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        print("\n6. Evaluating model...")
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # Calculate metrics
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        metrics = {
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
            'roc_auc': roc_auc_score(y_test, y_proba),
            'pr_auc': average_precision_score(y_test, y_proba),
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn)
        }
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
        
        # Print metrics
        print("\n   Model Performance:")
        print(f"   - Accuracy:  {metrics['accuracy']:.4f}")
        print(f"   - Precision: {metrics['precision']:.4f}")
        print(f"   - Recall:    {metrics['recall']:.4f}")
        print(f"   - F1 Score:  {metrics['f1_score']:.4f}")
        print(f"   - ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"   - PR-AUC:    {metrics['pr_auc']:.4f}")
        
        print("\n   Confusion Matrix:")
        print(f"   - True Positives:  {tp}")
        print(f"   - True Negatives:  {tn}")
        print(f"   - False Positives: {fp}")
        print(f"   - False Negatives: {fn}")
        
        # Save model artifacts
        print(f"\n7. Saving model artifacts to {model_output_dir}...")
        os.makedirs(model_output_dir, exist_ok=True)
        
        # Save with joblib (for Flask app)
        joblib.dump(model, os.path.join(model_output_dir, 'fraud_model.pkl'))
        joblib.dump(scaler, os.path.join(model_output_dir, 'scaler.pkl'))
        
        # Save feature columns
        with open(os.path.join(model_output_dir, 'feature_columns.txt'), 'w') as f:
            f.write('\n'.join(feature_cols))
        
        # Save metadata
        metadata = {
            'model_type': 'RandomForestClassifier',
            'trained_on': datetime.now().isoformat(),
            'feature_count': len(feature_cols),
            'metrics': metrics,
            'parameters': model_params
        }
        
        with open(os.path.join(model_output_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Log model to MLflow
        print("\n8. Logging model to MLflow...")
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="fraud-detection-rf"
        )
        
        # Log scaler as artifact
        mlflow.log_artifact(os.path.join(model_output_dir, 'scaler.pkl'))
        mlflow.log_artifact(os.path.join(model_output_dir, 'feature_columns.txt'))
        mlflow.log_artifact(os.path.join(model_output_dir, 'metadata.json'))
        
        # Get run ID
        run_id = mlflow.active_run().info.run_id
        
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("="*60)
        print(f"MLflow Run ID: {run_id}")
        print(f"Model saved to: {model_output_dir}")
        print(f"PR-AUC Score: {metrics['pr_auc']:.4f}")
        print("="*60)
        
        return model, scaler, feature_cols, metrics

if __name__ == '__main__':
    import sys
    
    # Get data path from command line or use default
    data_path = sys.argv[1] if len(sys.argv) > 1 else 'creditcard.csv'
    
    if not os.path.exists(data_path):
        print(f"Error: Data file not found at {data_path}")
        print("Usage: python train_model.py <path_to_creditcard.csv>")
        sys.exit(1)
    
    # Train model
    train_model(data_path)