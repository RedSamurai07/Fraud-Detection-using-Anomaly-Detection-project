from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
import os
import logging
from datetime import datetime
import mlflow
import mlflow.sklearn

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model and scaler
model = None
scaler = None
feature_columns = None

# Model directory
MODEL_DIR = os.getenv('MODEL_DIR', './models')
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns')

def load_model():
    """Load the trained model and scaler"""
    global model, scaler, feature_columns
    
    try:
        # Try loading from MLflow first
        if os.path.exists(os.path.join(MODEL_DIR, 'MLmodel')):
            logger.info("Loading model from MLflow format...")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            model = mlflow.sklearn.load_model(MODEL_DIR)
        else:
            # Fall back to joblib
            logger.info("Loading model from joblib...")
            model = joblib.load(os.path.join(MODEL_DIR, 'fraud_model.pkl'))
        
        scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
        
        # Load feature columns if available
        feature_file = os.path.join(MODEL_DIR, 'feature_columns.txt')
        if os.path.exists(feature_file):
            with open(feature_file, 'r') as f:
                feature_columns = [line.strip() for line in f.readlines()]
        else:
            # Default features
            feature_columns = [f'V{i}' for i in range(1, 29)] + ['Amount']
        
        logger.info(f"Model loaded successfully with {len(feature_columns)} features")
        return True
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return False

def engineer_features(transaction_df):
    """
    Engineer features similar to training process
    """
    df = transaction_df.copy()
    
    # Time-based features
    if 'Time' in df.columns:
        df['Hour'] = (df['Time'] // 3600) % 24
        df['Is_Night'] = ((df['Hour'] >= 22) | (df['Hour'] <= 5)).astype(int)
        df['Is_Rush_Hour'] = ((df['Hour'].between(7, 9)) | 
                              (df['Hour'].between(17, 19))).astype(int)
    
    # Amount-based features
    if 'Amount' in df.columns:
        df['Amount_Log'] = np.log1p(df['Amount'])
        df['Amount_ZScore'] = (df['Amount'] - df['Amount'].mean()) / (df['Amount'].std() + 1e-8)
        df['Is_Round_Amount'] = (df['Amount'] % 1 == 0).astype(int)
        df['Is_Small_Amount'] = (df['Amount'] < 1.0).astype(int)
    
    # Interaction features (if V columns exist)
    if 'V17' in df.columns and 'V14' in df.columns:
        df['V17_V14_interaction'] = df['V17'] * df['V14']
    
    if 'V17' in df.columns and 'Amount' in df.columns:
        df['V17_Amount_ratio'] = df['V17'] / (df['Amount'] + 1e-8)
    
    if 'V14' in df.columns and 'V12' in df.columns:
        df['V14_V12_interaction'] = df['V14'] * df['V12']
    
    return df

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'active',
        'service': 'Fraud Detection API',
        'version': '1.0.0',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health():
    """Detailed health check"""
    return jsonify({
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'features_count': len(feature_columns) if feature_columns else 0,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict fraud probability for a single transaction
    
    Expected JSON format:
    {
        "Time": 12345,
        "V1": -1.23,
        "V2": 2.34,
        ...
        "V28": 0.12,
        "Amount": 149.62
    }
    """
    try:
        # Get JSON data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame([data])
        
        # Engineer features
        df_engineered = engineer_features(df)
        
        # Select only required features
        if feature_columns:
            # Ensure all required columns exist
            missing_cols = set(feature_columns) - set(df_engineered.columns)
            if missing_cols:
                return jsonify({
                    'error': f'Missing required features: {list(missing_cols)}'
                }), 400
            
            X = df_engineered[feature_columns]
        else:
            X = df_engineered
        
        # Scale features
        X_scaled = scaler.transform(X)
        
        # Make prediction
        fraud_probability = model.predict_proba(X_scaled)[0][1]
        prediction = int(model.predict(X_scaled)[0])
        
        # Determine risk level
        if fraud_probability < 0.3:
            risk_level = 'LOW'
        elif fraud_probability < 0.7:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'HIGH'
        
        # Build response
        response = {
            'prediction': prediction,
            'fraud_probability': round(float(fraud_probability), 4),
            'risk_level': risk_level,
            'is_fraud': bool(prediction == 1),
            'timestamp': datetime.now().isoformat(),
            'transaction_amount': float(data.get('Amount', 0))
        }
        
        logger.info(f"Prediction made: {response}")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict fraud probability for multiple transactions
    
    Expected JSON format:
    {
        "transactions": [
            {"Time": 12345, "V1": -1.23, ..., "Amount": 149.62},
            {"Time": 12346, "V1": -2.34, ..., "Amount": 249.85}
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'transactions' not in data:
            return jsonify({'error': 'No transactions provided'}), 400
        
        transactions = data['transactions']
        
        if not isinstance(transactions, list):
            return jsonify({'error': 'Transactions must be a list'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(transactions)
        
        # Engineer features
        df_engineered = engineer_features(df)
        
        # Select features
        if feature_columns:
            X = df_engineered[feature_columns]
        else:
            X = df_engineered
        
        # Scale and predict
        X_scaled = scaler.transform(X)
        predictions = model.predict(X_scaled)
        probabilities = model.predict_proba(X_scaled)[:, 1]
        
        # Build results
        results = []
        for idx, (pred, prob) in enumerate(zip(predictions, probabilities)):
            if prob < 0.3:
                risk = 'LOW'
            elif prob < 0.7:
                risk = 'MEDIUM'
            else:
                risk = 'HIGH'
            
            results.append({
                'transaction_id': idx,
                'prediction': int(pred),
                'fraud_probability': round(float(prob), 4),
                'risk_level': risk,
                'is_fraud': bool(pred == 1)
            })
        
        response = {
            'total_transactions': len(results),
            'fraud_detected': sum(r['is_fraud'] for r in results),
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Batch prediction: {len(results)} transactions processed")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get model information"""
    try:
        info = {
            'model_type': str(type(model).__name__) if model else None,
            'features_count': len(feature_columns) if feature_columns else 0,
            'feature_names': feature_columns if feature_columns else [],
            'scaler_type': str(type(scaler).__name__) if scaler else None,
            'model_loaded': model is not None
        }
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Load model on startup
    if not load_model():
        logger.warning("Model not loaded - API will run but predictions will fail")
    
    # Get port from environment or default to 5000
    port = int(os.getenv('PORT', 5000))
    
    # Run app
    app.run(host='0.0.0.0', port=port, debug=False)