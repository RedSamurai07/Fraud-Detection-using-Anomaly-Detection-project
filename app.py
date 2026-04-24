import gradio as gr
import joblib
import pandas as pd
import numpy as np
import os
from PIL import Image

# --- Load Model and Scaler ---
MODEL_PATH = os.path.join('models', 'best_model.joblib')
SCALER_PATH = os.path.join('models', 'scaler.joblib')

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

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
        # Amount_ZScore is tricky for single row. 
        # Ideally we'd have the mean/std from training, but for now we use 0 or a placeholder.
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

def predict_fraud(*args):
    # Map args to features
    feature_names = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    input_data = dict(zip(feature_names, args))
    
    raw_df = pd.DataFrame([input_data])
    df_eng = engineer_features(raw_df)
    
    # Select features in correct order
    V_COLS = [f"V{i}" for i in range(1, 29)]
    OTHER_COLS = [
        'Amount_Log', 'Hour', 'Is_Night', 'Is_Rush_Hour',
        'Is_Round_Amount', 'Is_Small_Amount', 'Amount_ZScore',
        'V17_V14_interaction', 'V17_Amount_ratio', 'V14_V12_interaction'
    ]
    feature_cols = V_COLS + OTHER_COLS
    X_final = df_eng[feature_cols]
    
    # Scale and predict
    X_scaled = scaler.transform(X_final)
    prob = model.predict_proba(X_scaled)[0][1]
    prediction = "FRAUD" if prob > 0.5 else "LEGITIMATE"
    
    color = "red" if prediction == "FRAUD" else "green"
    
    return f"### Prediction: <span style='color:{color}'>{prediction}</span>\n\n**Fraud Probability:** {prob:.4f}"

# --- UI Setup ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    # Header Image
    header_img_path = "fraud_detection_header.png"
    if os.path.exists(header_img_path):
        gr.Image(header_img_path, show_label=False, interactive=False, height=250)
    
    gr.Markdown("# 🛡️ AI-Powered Fraud Detection Interface")
    gr.Markdown("Input transaction details to analyze the risk of fraud using our advanced anomaly detection model.")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Transaction Metadata")
            time_input = gr.Number(label="Time (Seconds from first transaction)", value=0)
            amount_input = gr.Number(label="Transaction Amount ($)", value=100.0)
            
            with gr.Accordion("PCA Components (V1 - V28)", open=False):
                gr.Markdown("These are anonymized features representing transaction characteristics.")
                v_inputs = [gr.Number(label=f"V{i}", value=0.0) for i in range(1, 29)]
        
        with gr.Column():
            gr.Markdown("### Analysis Results")
            output = gr.Markdown(label="Result")
            predict_btn = gr.Button("Analyze Transaction", variant="primary")
            
            gr.Markdown("---")
            gr.Markdown("### Sample Data")
            gr.Examples(
                examples=[
                    [0, -1.3598, -0.0727, 2.5363, 1.3781, -0.3383, 0.4623, 0.2395, 0.0986, 0.3637, 0.0907, -0.5515, -0.6178, -0.9913, -0.3111, 1.4681, -0.4704, 0.2079, 0.0257, 0.4039, 0.2514, -0.0183, 0.2778, -0.1104, 0.0669, 0.1285, -0.1891, 0.1335, -0.0210, 149.62],
                    [1, 1.1918, 0.2661, 0.1664, 0.4481, 0.0600, -0.0823, -0.0788, 0.0851, -0.2554, -0.1669, 1.6127, 1.0652, 0.4890, -0.1437, 0.6355, 0.4639, -0.1148, -0.1833, -0.1457, -0.0690, -0.2257, -0.6386, 0.1012, -0.3398, 0.1671, 0.1258, -0.0089, 0.0147, 2.69]
                ],
                inputs=[time_input] + v_inputs + [amount_input]
            )

    predict_btn.click(
        fn=predict_fraud,
        inputs=[time_input] + v_inputs + [amount_input],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch()
