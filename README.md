# Fraud Detection using Anomaly Detection project

## Table of contents
- [Project Overview](#project-overview)
- [Executive Summary](#executive-summary)
- [Goal](goal)
- [Data Structure](data-structure)
- [Tools](tools)
- [Analysis](#analysis)
- [Insights](insights)
- [Recommendations](recommendations)

### Project Overview
The project focuses on building a robust machine learning system to identify fraudulent credit card transactions. Given the sensitive nature of financial data, the features provided are principal components (V1–V28) resulting from a PCA transformation, along with the transaction time and amount. The primary challenge addressed is the extreme class imbalance, where fraudulent transactions represent a tiny fraction of the total dataset.

### Executive Summary

The analysis demonstrates that traditional metrics like "Accuracy" are misleading for fraud detection due to the Accuracy Paradox—a model predicting all transactions as legitimate would achieve 99.83% accuracy but fail to catch any fraud. To counter this, the project evaluates models based on Precision-Recall (PR) AUC and F-beta scores, prioritizing the identification of fraudulent cases while managing the operational costs of false alarms. The proposed solution includes a multi-tiered decisioning framework (Approve/Review/Block) to balance financial loss with customer experience.

### Goal

- Primary Objective: Develop a predictive model to classify transactions as fraudulent or legitimate with high precision and recall.

- Financial Goal: Minimize the total economic impact, which includes the average fraud loss ($122.21 per transaction) and the cost of investigating false positives ($15.00 for customer service and $5.00 for analyst review).

- Operational Goal: Create an automated system capable of making real-time decisions to block or flag transactions for manual investigation.

### Data structure and initial checks
[Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

 - The initial checks of your transactions.csv dataset reveal the following:

| Features | Description | Data types |
| -------- | -------- | -------- | 
| Time | Seconds elapsed between the transaction and the first transaction in the dataset. | float64 | 
| V1 – V28 | Numerical features resulting from a PCA transformation to protect user identity and sensitive information. | float64 | 
| Amount | The transaction amount, which can be used for cost-sensitive learning. | float64 | 
| Class | The target variable: 1 for fraudulent transactions and 0 for legitimate ones. | int64 | 

### Tools

1). Excel/CSV: Initial data inspection and output storage.

2). SQL: Used for production-ready queries including Cohort Analysis, Window Functions for Pareto thresholds, and Rolling Retention.

3). Python: Used for data cleaning, advanced feature engineering, and machine learning. Libraries: Pandas, Numpy, Scikit-learn (K-Means, GMM, Agglomerative), Scipy (Stats), Matplotlib, Seaborn.

4). Tableau: Data Visualization, Feature Engineering

### Analysis
**Python**

Laoding all the necessay libraries
``` python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
```
``` python
import os
import time
import json
from datetime import datetime, timedelta
from collections import Counter
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
os.makedirs('output', exist_ok=True)
```
Preprocessing
``` python
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, learning_curve)
from sklearn.pipeline import Pipeline
```
Imbalance Handling
``` python 
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler, TomekLinks
    from imblearn.combine import SMOTETomek
    from imblearn.pipeline import Pipeline as ImbPipeline
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    print("Install: pip install imbalanced-learn")
```
Importing Machine learning model libraries
``` python
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               IsolationForest, VotingClassifier)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
```
Deep Learning Models upload and check

``` python
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except (ImportError, TypeError, Exception) as e:
    TF_AVAILABLE = False
    print(f"TensorFlow not available: {e}")

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except (ImportError, Exception) as e:
    XGB_AVAILABLE = False
    print(f"XGBoost not available: {e}")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except (ImportError, Exception) as e:
    LGB_AVAILABLE = False
    print(f"LightGBM not available: {e}")
```
``` python

# Evaluation Metrics
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, auc,
    average_precision_score, roc_auc_score,
    f1_score, precision_score, recall_score,
    matthews_corrcoef, fbeta_score
)
```
``` python
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Install: pip install shap")

# Stats models
from scipy import stats
from scipy.stats import mannwhitneyu, ks_2samp

# Cost constants (realistic fraud costs)
FRAUD_LOSS_AVG        = 122.21   # Average transaction amount in dataset
FALSE_POSITIVE_COST   = 15.00    # Customer service cost for blocked legit txn
FALSE_NEGATIVE_COST   = FRAUD_LOSS_AVG  # Avg fraud amount lost
INVESTIGATION_COST    = 5.00     # Cost to investigate flagged transaction

print("Setup complete. All imports loaded.")
```
Loading data and Initial exploration
``` python
print("\n" + "="*60)
print("SECTION 1: LOAD DATA & INITIAL EXPLORATION")
print("="*60)

df = pd.read_csv('creditcard.csv')

print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nDuplicate rows: {df.duplicated().sum()}")
print(f"\nClass distribution:")
print(df['Class'].value_counts())
print(f"\nFraud rate: {df['Class'].mean()*100:.4f}%")
print(f"\nStatistical summary:")
print(df[['Time', 'Amount', 'Class']].describe())
```
<img width="850" height="466" alt="image" src="https://github.com/user-attachments/assets/04332ba0-8f9a-4db8-b784-b6ae7e73bdb4" /><img width="252" height="112" alt="image" src="https://github.com/user-attachments/assets/d55ff77e-cfe2-4fb0-b3b7-132b012b3c99" /><img width="132" height="466" alt="image" src="https://github.com/user-attachments/assets/aa13aedd-4c87-4b58-8d2e-7b85ca972c58" /><img width="135" height="97" alt="image" src="https://github.com/user-attachments/assets/536cacbb-881b-4b88-866b-ac6f0cdc5368" /><img width="108" height="481" alt="image" src="https://github.com/user-attachments/assets/85419ae6-f8b3-4406-aac6-17134699570c" /><img width="299" height="413" alt="image" src="https://github.com/user-attachments/assets/2dff3432-bada-4c02-a25d-519e47011f9d" />

Demonstration of the accuracy paradox explicitly

``` python
print("\n" + "="*60)
print("SECTION 2: THE ACCURACY PARADOX DEMONSTRATION")
print("="*60)

total         = len(df)
fraud_count   = df['Class'].sum()
legit_count   = total - fraud_count
fraud_rate    = fraud_count / total

print(f"\nTotal transactions:     {total:,}")
print(f"Fraud transactions:     {fraud_count:,}")
print(f"Legitimate transactions:{legit_count:,}")
print(f"Fraud rate:             {fraud_rate*100:.4f}%")

# A naive model that ALWAYS predicts 'Not Fraud'
naive_accuracy = legit_count / total
naive_precision = 0   # Never predicts fraud, so precision undefined
naive_recall    = 0   # Catches 0 fraud cases

print(f"\n── Naive Model (always predicts NOT FRAUD) ──────────────")
print(f"Accuracy:  {naive_accuracy*100:.2f}%  ← MISLEADINGLY HIGH")
print(f"Precision: {naive_precision*100:.2f}%  ← Catches NO fraud")
print(f"Recall:    {naive_recall*100:.2f}%  ← Catches NO fraud")
print(f"F1 Score:  0.00%")
print(f"Financial Loss: £{fraud_count * FALSE_NEGATIVE_COST:,.2f} (all fraud undetected)")

print(f"\n── Why We Use Precision-Recall AUC Instead ──────────────")
print(f"Precision = Of all flagged transactions, how many are truly fraud?")
print(f"Recall    = Of all fraud transactions, how many did we catch?")
print(f"PR-AUC    = Area under Precision-Recall curve — accounts for imbalance")
print(f"F-beta    = Weighted F score — when recall matters more than precision")

# Visualize the imbalance
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Class distribution
axes[0].bar(['Legitimate', 'Fraud'],
            [legit_count, fraud_count],
            color=['#3498DB', '#E74C3C'])
axes[0].set_title('Class Distribution (Raw Count)')
axes[0].set_ylabel('Number of Transactions')
for i, v in enumerate([legit_count, fraud_count]):
    axes[0].text(i, v + 100, f'{v:,}', ha='center', fontweight='bold')

# Log scale
axes[1].bar(['Legitimate', 'Fraud'],
            [legit_count, fraud_count],
            color=['#3498DB', '#E74C3C'])
axes[1].set_yscale('log')
axes[1].set_title('Class Distribution (Log Scale)')
axes[1].set_ylabel('Count (log scale)')

# Accuracy paradox comparison
metrics = ['Accuracy', 'Precision', 'Recall', 'F1']
naive   = [99.83, 0, 0, 0]
axes[2].bar(metrics, naive, color=['green', 'red', 'red', 'red'])
axes[2].set_title('Naive Model Metrics — The Accuracy Paradox')
axes[2].set_ylabel('Score (%)')
axes[2].set_ylim(0, 110)
for i, v in enumerate(naive):
    axes[2].text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('output/accuracy_paradox.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: accuracy_paradox.png")
```

<img width="513" height="365" alt="image" src="https://github.com/user-attachments/assets/fe4771df-8c6e-4959-9fc7-2a3e0bd1efda" /><img width="513" height="365" alt="image" src="https://github.com/user-attachments/assets/5f9bd930-e1b8-47b9-b4a2-7ef2578d4cb2" />

Exploratory Data Analysis

``` python
# EDA
print("\n" + "="*60)
print("SECTION 3: EXPLORATORY DATA ANALYSIS")
print("="*60)

# Amount distribution by class 
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Amount distribution
fraud = df[df['Class'] == 1]['Amount']
legit = df[df['Class'] == 0]['Amount']
axes[0, 0].hist(legit, bins=100, alpha=0.6, color='blue',
                label=f'Legitimate (n={len(legit):,})', density=True)
axes[0, 0].hist(fraud, bins=50, alpha=0.8, color='red',
                label=f'Fraud (n={len(fraud):,})', density=True)
axes[0, 0].set_xlabel('Transaction Amount (£)')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title('Amount Distribution: Fraud vs Legitimate')
axes[0, 0].legend()
axes[0, 0].set_xlim(0, 1000)

# Amount statistics comparison
amount_stats = df.groupby('Class')['Amount'].agg(
    ['mean', 'median', 'std', 'max', 'min']
).round(2)
amount_stats.index = ['Legitimate', 'Fraud']
print("\nAmount Statistics by Class:")
print(amount_stats)

# Time distribution
axes[0, 1].hist(df[df['Class']==0]['Time']/3600, bins=100,
                alpha=0.6, color='blue', label='Legitimate', density=True)
axes[0, 1].hist(df[df['Class']==1]['Time']/3600, bins=50,
                alpha=0.8, color='red', label='Fraud', density=True)
axes[0, 1].set_xlabel('Time (hours from start)')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('Transaction Time: Fraud vs Legitimate')
axes[0, 1].legend()

# Fraud rate by time window
df['Hour'] = (df['Time'] // 3600) % 24
hourly_fraud = df.groupby('Hour').agg(
    Total=('Class', 'count'),
    Fraud=('Class', 'sum')
)
hourly_fraud['Fraud_Rate'] = hourly_fraud['Fraud'] / hourly_fraud['Total'] * 100

axes[1, 0].bar(hourly_fraud.index, hourly_fraud['Fraud_Rate'],
               color='#E74C3C', alpha=0.8)
axes[1, 0].set_xlabel('Hour of Day')
axes[1, 0].set_ylabel('Fraud Rate (%)')
axes[1, 0].set_title('Fraud Rate by Hour of Day')

# Correlation of features with fraud
feature_cols = [c for c in df.columns if c.startswith('V')]
correlations = df[feature_cols + ['Class']].corr()['Class'].drop('Class').sort_values()
top_corr = pd.concat([correlations.head(7), correlations.tail(7)])
colors = ['#E74C3C' if x < 0 else '#2ECC71' for x in top_corr.values]
axes[1, 1].barh(top_corr.index, top_corr.values, color=colors)
axes[1, 1].set_xlabel('Correlation with Fraud (Class=1)')
axes[1, 1].set_title('Top 14 Features Correlated with Fraud')
axes[1, 1].axvline(0, color='black', linewidth=0.8)

plt.tight_layout()
plt.savefig('output/eda_overview.png', dpi=150, bbox_inches='tight')
plt.show()

# KS Test: Are features statistically different between classes
print("\n── KS Test: Feature Distribution Differences (Fraud vs Legit) ──")
ks_results = []
for col in feature_cols + ['Amount']:
    fraud_vals = df[df['Class'] == 1][col]
    legit_vals = df[df['Class'] == 0][col]
    stat, p = ks_2samp(fraud_vals, legit_vals)
    ks_results.append({'Feature': col, 'KS_Statistic': stat, 'P_Value': p,
                       'Significant': p < 0.05})

ks_df = pd.DataFrame(ks_results).sort_values('KS_Statistic', ascending=False)
print(ks_df.to_string(index=False))
print(f"\nFeatures significantly different between classes: "
      f"{ks_df['Significant'].sum()} / {len(ks_df)}")
```
<img width="420" height="145" alt="image" src="https://github.com/user-attachments/assets/a23a2d0c-90d2-462c-a9db-a1fa78cfd199" />
<img width="420" height="145" alt="image" src="https://github.com/user-attachments/assets/9f62a65b-b9e4-43ca-a3e8-728f94b43084" />
<img width="447" height="488" alt="image" src="https://github.com/user-attachments/assets/6849e025-9a85-4780-9c43-45f378283e40" /><img width="403" height="65" alt="image" src="https://github.com/user-attachments/assets/aec71c4c-bcc3-437d-81f2-69712c5e45b1" />

Feature Engineering and Behavioural Analysis
``` python
# Feature Engineering
print("\n" + "="*60)
print("SECTION 4: ADVANCED FEATURE ENGINEERING")
print("="*60)

df_eng = df.copy()

# Time-based features 
df_eng['Hour']          = (df_eng['Time'] // 3600) % 24
df_eng['Is_Night']      = ((df_eng['Hour'] >= 22) | (df_eng['Hour'] <= 5)).astype(int)
df_eng['Is_Rush_Hour']  = ((df_eng['Hour'].between(7, 9)) |
                            (df_eng['Hour'].between(17, 19))).astype(int)
df_eng['Day_Number']    = (df_eng['Time'] // 86400).astype(int)  # Day 0 or 1

# Amount-based features
df_eng['Amount_Log']    = np.log1p(df_eng['Amount'])
df_eng['Amount_Squared']= df_eng['Amount'] ** 2
df_eng['Is_Round_Amount']= (df_eng['Amount'] % 1 == 0).astype(int)
df_eng['Is_Small_Amount']= (df_eng['Amount'] < 1).astype(int)  # Micro-test transactions

# ── Statistical aggregation features (rolling z-score of amount) ──
# Sort by time first
df_eng = df_eng.sort_values('Time').reset_index(drop=True)
df_eng['Amount_Rolling_Mean'] = df_eng['Amount'].rolling(window=100, min_periods=1).mean()
df_eng['Amount_Rolling_Std']  = df_eng['Amount'].rolling(window=100, min_periods=1).std().fillna(1)
df_eng['Amount_ZScore']       = (
    (df_eng['Amount'] - df_eng['Amount_Rolling_Mean'])
    / df_eng['Amount_Rolling_Std']
)

# PCA component interactions (top correlated) 
# From KS test, V17, V14, V12, V10 most differentiating
df_eng['V17_V14_interaction'] = df_eng['V17'] * df_eng['V14']
df_eng['V17_Amount_ratio']    = df_eng['V17'] / (df_eng['Amount'] + 1)
df_eng['V14_V12_interaction'] = df_eng['V14'] * df_eng['V12']

print(f"Original features:     {df.shape[1]}")
print(f"Engineered features:   {df_eng.shape[1]}")
print(f"\nNew features added:")
new_features = [c for c in df_eng.columns if c not in df.columns]
for f in new_features:
    print(f"  + {f}")

# Validate new features separate classes
print("\n── New Feature Statistics by Class ──")
for feat in ['Amount_Log', 'Is_Night', 'Is_Small_Amount', 'Amount_ZScore']:
    fraud_mean = df_eng[df_eng['Class']==1][feat].mean()
    legit_mean = df_eng[df_eng['Class']==0][feat].mean()
    print(f"  {feat}: Fraud={fraud_mean:.3f}, Legit={legit_mean:.3f}, "
          f"Diff={abs(fraud_mean-legit_mean):.3f}")
```
<img width="423" height="446" alt="image" src="https://github.com/user-attachments/assets/809ee09f-1e4d-49e5-ba21-fd37c172c63c" />

First detection philosophy: Explicit business rules

``` python
print("\n" + "="*60)
print("SECTION 5: RULE-BASED DETECTION — PHILOSOPHY 1")
print("="*60)

def rule_based_detector(row):
    """
    Explicit business rule fraud detector.
    Returns 1 (fraud) or 0 (legit) based on domain rules.
    """
    flags = 0
    reasons = []

    # Rule 1: Small test amount followed by larger transaction pattern
    if row['Amount'] < 1.0:
        flags += 1
        reasons.append("Micro-test transaction (<£1)")

    # Rule 2: Suspicious hour
    hour = int((row['Time'] // 3600) % 24)
    if hour >= 23 or hour <= 4:
        flags += 1
        reasons.append(f"Night-time transaction (hour={hour})")

    # Rule 3: High-value transaction
    if row['Amount'] > 1000:
        flags += 1
        reasons.append(f"High-value transaction (£{row['Amount']:.2f})")

    # Rule 4: Key PCA features outside normal range
    if row['V14'] < -5:
        flags += 2
        reasons.append(f"V14 anomaly ({row['V14']:.2f})")

    if row['V17'] < -5:
        flags += 2
        reasons.append(f"V17 anomaly ({row['V17']:.2f})")

    if row['V10'] < -5:
        flags += 1
        reasons.append(f"V10 anomaly ({row['V10']:.2f})")

    return 1 if flags >= 2 else 0

print("Applying rule-based detection...")
t0 = time.time()
df_eng['Rule_Pred'] = df_eng.apply(rule_based_detector, axis=1)
t_rule = time.time() - t0

# Evaluate rules
rule_pr_auc = average_precision_score(df_eng['Class'], df_eng['Rule_Pred'])
rule_cm     = confusion_matrix(df_eng['Class'], df_eng['Rule_Pred'])
rule_recall = recall_score(df_eng['Class'], df_eng['Rule_Pred'])
rule_prec   = precision_score(df_eng['Class'], df_eng['Rule_Pred'],
                               zero_division=0)

print(f"\nRule-Based Detector Results:")
print(f"  Precision-Recall AUC: {rule_pr_auc:.4f}")
print(f"  Recall (fraud caught): {rule_recall*100:.1f}%")
print(f"  Precision:             {rule_prec*100:.1f}%")
print(f"  Processing time:       {t_rule:.2f}s")
print(f"  Confusion Matrix:\n{rule_cm}")

# Financial cost of rule-based
rule_fn = rule_cm[1][0]  # Missed fraud (false negatives)
rule_fp = rule_cm[0][1]  # False alarms (false positives)
rule_cost = (rule_fn * FALSE_NEGATIVE_COST) + (rule_fp * FALSE_POSITIVE_COST)
print(f"\n  Financial Cost (Rule-Based):")
print(f"  Missed fraud: {rule_fn} × £{FALSE_NEGATIVE_COST:.2f} = "
      f"£{rule_fn*FALSE_NEGATIVE_COST:,.2f}")
print(f"  False alarms: {rule_fp} × £{FALSE_POSITIVE_COST:.2f} = "
      f"£{rule_fp*FALSE_POSITIVE_COST:,.2f}")
print(f"  TOTAL COST:   £{rule_cost:,.2f}")
print(f"\nLimitation: Rules are static, brittle, and miss novel fraud patterns")
```
<img width="491" height="353" alt="image" src="https://github.com/user-attachments/assets/60f6458f-b82b-4514-a7bf-2f13afe1af6f" />

Data Preparation for Machine Learning
``` python
print("\n" + "="*60)
print("SECTION 6: DATA PREPARATION — SCALING & SPLITTING")
print("="*60)

feature_cols_ml = ([c for c in df_eng.columns if c.startswith('V')] +
                   ['Amount_Log', 'Hour', 'Is_Night', 'Is_Rush_Hour',
                    'Is_Round_Amount', 'Is_Small_Amount', 'Amount_ZScore',
                    'V17_V14_interaction', 'V17_Amount_ratio', 'V14_V12_interaction'])

X = df_eng[feature_cols_ml].fillna(0)
y = df_eng['Class']

# Stratified split preserves fraud ratio in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape} | Fraud: {y_train.sum()} ({y_train.mean()*100:.3f}%)")
print(f"Test set:     {X_test.shape}  | Fraud: {y_test.sum()} ({y_test.mean()*100:.3f}%)")

# RobustScaler is better than StandardScaler for fraud data
# — less sensitive to outliers in Amount
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\nUsing RobustScaler (not StandardScaler) — reason:")
print(f"Amount has extreme outliers (max=£{df['Amount'].max():.0f})")
print(f"RobustScaler uses median and IQR instead of mean and std")
print(f"→ Outliers have less influence on scaling")
```
<img width="419" height="175" alt="image" src="https://github.com/user-attachments/assets/a2760e6c-59d1-4f50-8196-33749a312429" />

Handling Data Imbalance
``` python
print("\n" + "="*60)
print("SECTION 7: IMBALANCE HANDLING STRATEGY COMPARISON")
print("="*60)

if IMBLEARN_AVAILABLE:
    strategies = {
        'No Resampling': (X_train_scaled, y_train),
    }

    # SMOTE
    sm = SMOTE(random_state=42, k_neighbors=5)
    X_sm, y_sm = sm.fit_resample(X_train_scaled, y_train)
    strategies['SMOTE'] = (X_sm, y_sm)

    # BorderlineSMOTE
    bsm = BorderlineSMOTE(random_state=42)
    X_bsm, y_bsm = bsm.fit_resample(X_train_scaled, y_train)
    strategies['BorderlineSMOTE'] = (X_bsm, y_bsm)

    # ADASYN
    ada = ADASYN(random_state=42)
    X_ada, y_ada = ada.fit_resample(X_train_scaled, y_train)
    strategies['ADASYN'] = (X_ada, y_ada)

    # SMOTETomek (combined over+under sampling)
    smt = SMOTETomek(random_state=42)
    X_smt, y_smt = smt.fit_resample(X_train_scaled, y_train)
    strategies['SMOTETomek'] = (X_smt, y_smt)

    print("\nResampled dataset sizes:")
    for name, (Xs, ys) in strategies.items():
        fraud_n = ys.sum() if hasattr(ys, 'sum') else sum(ys)
        total_n = len(ys)
        print(f"  {name}: Total={total_n:,} | Fraud={fraud_n:,} "
              f"({fraud_n/total_n*100:.1f}%)")

    # Quick evaluation of each strategy with Logistic Regression
    print("\nLogistic Regression PR-AUC by resampling strategy:")
    strategy_results = {}
    for name, (Xs, ys) in strategies.items():
        lr = LogisticRegression(max_iter=500, random_state=42,
                                class_weight='balanced')
        lr.fit(Xs, ys)
        proba = lr.predict_proba(X_test_scaled)[:, 1]
        pr_auc = average_precision_score(y_test, proba)
        strategy_results[name] = pr_auc
        print(f"  {name}: PR-AUC = {pr_auc:.4f}")

    best_strategy = max(strategy_results, key=strategy_results.get)
    print(f"\nBest resampling strategy: {best_strategy}")

    # Use SMOTE for remaining analysis
    X_train_res, y_train_res = strategies.get('SMOTE', (X_train_scaled, y_train))
else:
    X_train_res, y_train_res = X_train_scaled, y_train
    print("Using class_weight='balanced' as imbalance handling")
```
<img width="383" height="251" alt="image" src="https://github.com/user-attachments/assets/0e1f94ed-1d34-4b2b-8f38-90f023eeab29" />

Tarditional Machine learning detection

``` python
print("\n" + "="*60)
print("SECTION 8: TRADITIONAL ML DETECTION — PHILOSOPHY 2")
print("="*60)

# Unsupervised Anomaly Detection
print("\n── Unsupervised Anomaly Detectors ────────────────────────")

# Use only legitimate training data for unsupervised methods
X_train_legit = X_train_scaled[y_train == 0]

models_unsupervised = {}

# Isolation Forest
t0 = time.time()
iso = IsolationForest(n_estimators=200, contamination=fraud_rate,
                      random_state=42, n_jobs=-1)
iso.fit(X_train_legit)
iso_scores = -iso.score_samples(X_test_scaled)  # Higher = more anomalous
iso_preds  = (iso.predict(X_test_scaled) == -1).astype(int)
t_iso = time.time() - t0
models_unsupervised['Isolation Forest'] = {
    'scores': iso_scores, 'preds': iso_preds, 'time': t_iso
}

# Local Outlier Factor
t0 = time.time()
lof = LocalOutlierFactor(n_neighbors=20, contamination=fraud_rate,
                          novelty=True, n_jobs=-1)
lof.fit(X_train_legit)
lof_scores = -lof.score_samples(X_test_scaled)
lof_preds  = (lof.predict(X_test_scaled) == -1).astype(int)
t_lof = time.time() - t0
models_unsupervised['Local Outlier Factor'] = {
    'scores': lof_scores, 'preds': lof_preds, 'time': t_lof
}

# One-Class SVM (sample for speed)
sample_idx = np.random.choice(len(X_train_legit),
                               min(5000, len(X_train_legit)), replace=False)
t0 = time.time()
ocsvm = OneClassSVM(kernel='rbf', gamma='auto', nu=fraud_rate)
ocsvm.fit(X_train_legit[sample_idx])
ocsvm_scores = -ocsvm.score_samples(X_test_scaled)
ocsvm_preds  = (ocsvm.predict(X_test_scaled) == -1).astype(int)
t_ocsvm = time.time() - t0
models_unsupervised['One-Class SVM'] = {
    'scores': ocsvm_scores, 'preds': ocsvm_preds, 'time': t_ocsvm
}

print(f"\n{'Model':<25} {'PR-AUC':>8} {'Recall':>8} {'Precision':>10} {'Time':>8}")
print("-" * 65)
unsup_results = {}
for name, m in models_unsupervised.items():
    pr_auc = average_precision_score(y_test, m['scores'])
    rec    = recall_score(y_test, m['preds'])
    prec   = precision_score(y_test, m['preds'], zero_division=0)
    t      = m['time']
    print(f"{name:<25} {pr_auc:>8.4f} {rec:>8.4f} {prec:>10.4f} {t:>6.2f}s")
    unsup_results[name] = {'pr_auc': pr_auc, 'recall': rec, 'precision': prec}

# Supervised ML Models 
print("\n── Supervised ML Models ──────────────────────────────────")

models_supervised = {}

# Logistic Regression
t0 = time.time()
lr  = LogisticRegression(max_iter=500, class_weight='balanced', random_state=42)
lr.fit(X_train_res, y_train_res)
lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
t_lr = time.time() - t0
models_supervised['Logistic Regression'] = {
    'model': lr, 'proba': lr_proba, 'time': t_lr
}

# Random Forest
t0 = time.time()
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced',
                             max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train_res, y_train_res)
rf_proba = rf.predict_proba(X_test_scaled)[:, 1]
t_rf = time.time() - t0
models_supervised['Random Forest'] = {
    'model': rf, 'proba': rf_proba, 'time': t_rf
}

# XGBoost
if XGB_AVAILABLE:
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    t0 = time.time()
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.05,
        scale_pos_weight=scale_pos, eval_metric='aucpr',
        random_state=42, n_jobs=-1, verbosity=0
    )
    xgb_model.fit(X_train_scaled, y_train,
                  eval_set=[(X_test_scaled, y_test)],
                  verbose=False)
    xgb_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
    t_xgb = time.time() - t0
    models_supervised['XGBoost'] = {
        'model': xgb_model, 'proba': xgb_proba, 'time': t_xgb
    }

# LightGBM
if LGB_AVAILABLE:
    t0 = time.time()
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        class_weight='balanced', random_state=42, n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_train_scaled, y_train)
    lgb_proba = lgb_model.predict_proba(X_test_scaled)[:, 1]
    t_lgb = time.time() - t0
    models_supervised['LightGBM'] = {
        'model': lgb_model, 'proba': lgb_proba, 'time': t_lgb
    }

print(f"\n{'Model':<25} {'PR-AUC':>8} {'ROC-AUC':>9} {'Time':>8}")
print("-" * 55)
sup_results = {}
for name, m in models_supervised.items():
    pr_auc  = average_precision_score(y_test, m['proba'])
    roc_auc = roc_auc_score(y_test, m['proba'])
    t       = m['time']
    print(f"{name:<25} {pr_auc:>8.4f} {roc_auc:>9.4f} {t:>6.2f}s")
    sup_results[name] = {'pr_auc': pr_auc, 'roc_auc': roc_auc}
```
<img width="446" height="354" alt="image" src="https://github.com/user-attachments/assets/2de07bb4-defa-4aeb-9f6e-337eb3a5ce91" />

Deep Learning Architecture using Autoencoders
``` python
print("\n" + "="*60)
print("SECTION 9: AUTOENCODER ANOMALY DETECTION — PHILOSOPHY 3")
print("="*60)

if TF_AVAILABLE:
    input_dim = X_train_scaled.shape[1]

    # Build Autoencoder
    def build_autoencoder(input_dim, encoding_dim=14):
        inputs  = keras.Input(shape=(input_dim,))
        # Encoder
        encoded = layers.Dense(32, activation='relu')(inputs)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(encoding_dim, activation='relu')(encoded)
        # Decoder
        decoded = layers.Dense(32, activation='relu')(encoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(input_dim, activation='linear')(decoded)

        autoencoder = keras.Model(inputs, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        return autoencoder

    autoencoder = build_autoencoder(input_dim)
    autoencoder.summary()

    # Train ONLY on legitimate transactions
    X_train_legit_ae = X_train_scaled[y_train == 0]

    history = autoencoder.fit(
        X_train_legit_ae, X_train_legit_ae,
        epochs=30, batch_size=256,
        validation_split=0.1,
        verbose=1,
        callbacks=[
            keras.callbacks.EarlyStopping(
                patience=5, restore_best_weights=True
            )
        ]
    )

    # Reconstruction error = anomaly score
    X_test_reconstructed = autoencoder.predict(X_test_scaled, verbose=0)
    reconstruction_error = np.mean(np.power(X_test_scaled - X_test_reconstructed, 2), axis=1)

    # ROC and PR curves
    ae_pr_auc  = average_precision_score(y_test, reconstruction_error)
    ae_roc_auc = roc_auc_score(y_test, reconstruction_error)

    print(f"\nAutoencoder Results:")
    print(f"  PR-AUC:  {ae_pr_auc:.4f}")
    print(f"  ROC-AUC: {ae_roc_auc:.4f}")

    # Training loss plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(history.history['loss'],     label='Train Loss')
    axes[0].plot(history.history['val_loss'], label='Val Loss')
    axes[0].set_title('Autoencoder Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].legend()

    # Reconstruction error distribution
    axes[1].hist(reconstruction_error[y_test == 0], bins=100,
                 alpha=0.6, color='blue', label='Legitimate', density=True)
    axes[1].hist(reconstruction_error[y_test == 1], bins=50,
                 alpha=0.8, color='red', label='Fraud', density=True)
    axes[1].set_xlabel('Reconstruction Error (MSE)')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Reconstruction Error: Fraud vs Legitimate')
    axes[1].legend()
    axes[1].set_xlim(0, np.percentile(reconstruction_error, 99))

    plt.tight_layout()
    plt.savefig('output/autoencoder_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Optimal threshold for autoencoder
    precisions, recalls, thresholds = precision_recall_curve(
        y_test, reconstruction_error
    )
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_thresh_idx = np.argmax(f1_scores)
    best_thresh_ae  = thresholds[best_thresh_idx]
    ae_preds = (reconstruction_error >= best_thresh_ae).astype(int)

    print(f"\nOptimal Threshold: {best_thresh_ae:.4f}")
    print(classification_report(y_test, ae_preds,
          target_names=['Legitimate', 'Fraud']))
else:
    print("TensorFlow not available — install: pip install tensorflow")
    ae_pr_auc = 0
```
<img width="537" height="551" alt="image" src="https://github.com/user-attachments/assets/ae6e6606-ec8c-4117-b66e-7edafc71cc5b" /><img width="518" height="201" alt="image" src="https://github.com/user-attachments/assets/b2685de8-425c-4831-8a43-9e6cd5108745" /><img width="502" height="496" alt="image" src="https://github.com/user-attachments/assets/cdfbcaed-b154-407f-bbfa-3ac05cb39534" />
<img width="1289" height="390" alt="image" src="https://github.com/user-attachments/assets/f777e059-1b96-4e98-a919-9e76de1287ea" />
<img width="383" height="171" alt="image" src="https://github.com/user-attachments/assets/fef4875b-5058-42ed-9782-585409118c8f" />

To Find optimal threshold for business cost minimization

``` python
print("\n" + "="*60)
print("SECTION 10: THRESHOLD OPTIMIZATION — COST-MINIMIZING")
print("="*60)

# Use best supervised model (default RF for demonstration)
best_model_name = max(sup_results, key=lambda k: sup_results[k]['pr_auc'])
best_proba      = models_supervised[best_model_name]['proba']

print(f"\nUsing: {best_model_name}")

thresholds_to_test = np.arange(0.01, 1.0, 0.01)
threshold_results  = []

for thresh in thresholds_to_test:
    preds = (best_proba >= thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()

    cost = (fn * FALSE_NEGATIVE_COST +
            fp * FALSE_POSITIVE_COST +
            (tp + fp) * INVESTIGATION_COST)

    threshold_results.append({
        'Threshold': thresh,
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'Precision':  precision_score(y_test, preds, zero_division=0),
        'Recall':     recall_score(y_test, preds),
        'F1':         f1_score(y_test, preds, zero_division=0),
        'F2':         fbeta_score(y_test, preds, beta=2, zero_division=0),
        'MCC':        matthews_corrcoef(y_test, preds),
        'Total_Cost': cost
    })

thresh_df   = pd.DataFrame(threshold_results)
best_cost_thresh = thresh_df.loc[thresh_df['Total_Cost'].idxmin(), 'Threshold']
best_f2_thresh   = thresh_df.loc[thresh_df['F2'].idxmax(), 'Threshold']
best_f1_thresh   = thresh_df.loc[thresh_df['F1'].idxmax(), 'Threshold']

print(f"\nOptimal Thresholds:")
print(f"  Cost-minimizing threshold: {best_cost_thresh:.2f}")
print(f"  F2-maximizing threshold:   {best_f2_thresh:.2f}")
print(f"  F1-maximizing threshold:   {best_f1_thresh:.2f}")
print(f"  Default threshold (0.5):   0.50")

# Compare at each threshold
for thresh_name, thresh_val in [
    ('Default (0.5)',   0.5),
    ('F1-optimal',      best_f1_thresh),
    ('F2-optimal',      best_f2_thresh),
    ('Cost-optimal',    best_cost_thresh)
]:
    row = thresh_df[thresh_df['Threshold'].round(2) == round(thresh_val, 2)]
    if not row.empty:
        row = row.iloc[0]
        print(f"\n  [{thresh_name}] threshold={thresh_val:.2f}")
        print(f"    Recall={row['Recall']:.3f} | Precision={row['Precision']:.3f} | "
              f"F2={row['F2']:.3f} | Cost=£{row['Total_Cost']:,.2f}")

# Plot threshold analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(thresh_df['Threshold'], thresh_df['Recall'],    label='Recall', color='blue')
axes[0, 0].plot(thresh_df['Threshold'], thresh_df['Precision'], label='Precision', color='green')
axes[0, 0].plot(thresh_df['Threshold'], thresh_df['F1'],        label='F1', color='purple')
axes[0, 0].plot(thresh_df['Threshold'], thresh_df['F2'],        label='F2 (recall-weighted)', color='orange')
axes[0, 0].axvline(best_f2_thresh, color='orange', linestyle='--', alpha=0.7)
axes[0, 0].axvline(0.5, color='grey', linestyle=':', label='Default 0.5')
axes[0, 0].set_xlabel('Decision Threshold')
axes[0, 0].set_ylabel('Score')
axes[0, 0].set_title('Metrics vs Threshold')
axes[0, 0].legend()

axes[0, 1].plot(thresh_df['Threshold'], thresh_df['Total_Cost']/1000, color='red')
axes[0, 1].axvline(best_cost_thresh, color='red', linestyle='--',
                    label=f'Min Cost @ {best_cost_thresh:.2f}')
axes[0, 1].set_xlabel('Decision Threshold')
axes[0, 1].set_ylabel('Total Financial Cost (£000s)')
axes[0, 1].set_title('Financial Cost vs Threshold')
axes[0, 1].legend()

axes[1, 0].plot(thresh_df['Threshold'], thresh_df['FP'], label='False Positives', color='orange')
axes[1, 0].plot(thresh_df['Threshold'], thresh_df['FN'], label='False Negatives', color='red')
axes[1, 0].set_xlabel('Decision Threshold')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title('FP vs FN Trade-off by Threshold')
axes[1, 0].legend()

axes[1, 1].plot(thresh_df['Threshold'], thresh_df['MCC'], color='purple')
axes[1, 1].axvline(thresh_df.loc[thresh_df['MCC'].idxmax(), 'Threshold'],
                    color='purple', linestyle='--')
axes[1, 1].set_xlabel('Decision Threshold')
axes[1, 1].set_ylabel('Matthews Correlation Coefficient')
axes[1, 1].set_title('MCC vs Threshold (handles imbalance well)')

plt.tight_layout()
plt.savefig('output/threshold_optimization.png', dpi=150, bbox_inches='tight')
plt.show()
```

<img width="435" height="388" alt="image" src="https://github.com/user-attachments/assets/aeeca585-ea51-4327-a003-8394f733fe15" /><img width="1390" height="989" alt="image" src="https://github.com/user-attachments/assets/1566f891-b6f6-453b-918d-3bd3d5bcd35c" />

Cost Sensitive Evaluation Framework

``` python
print("\n" + "="*60)
print("SECTION 11: COST-SENSITIVE CONFUSION MATRIX")
print("="*60)

def cost_sensitive_report(y_true, y_pred, y_proba, model_name,
                           fn_cost=FALSE_NEGATIVE_COST,
                           fp_cost=FALSE_POSITIVE_COST,
                           inv_cost=INVESTIGATION_COST):
    """Full cost-sensitive evaluation report."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    total = len(y_true)

    # Financial calculations
    fraud_loss     = fn * fn_cost
    fp_cost_total  = fp * fp_cost
    inv_cost_total = (tp + fp) * inv_cost
    total_cost     = fraud_loss + fp_cost_total + inv_cost_total

    # Maximum possible loss (catching nothing)
    max_possible_loss = y_true.sum() * fn_cost

    # Savings vs doing nothing
    savings = max_possible_loss - fraud_loss

    pr_auc  = average_precision_score(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    mcc     = matthews_corrcoef(y_true, y_pred)
    f2      = fbeta_score(y_true, y_pred, beta=2, zero_division=0)

    print(f"\n{'='*55}")
    print(f"MODEL: {model_name}")
    print(f"{'='*55}")
    print(f"\n  CLASSIFICATION METRICS:")
    print(f"    PR-AUC:     {pr_auc:.4f}  ← Primary metric")
    print(f"    ROC-AUC:    {roc_auc:.4f}")
    print(f"    F2 Score:   {f2:.4f}  ← Recall-weighted")
    print(f"    MCC:        {mcc:.4f}  ← Handles imbalance")
    print(f"    Precision:  {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"    Recall:     {recall_score(y_true, y_pred):.4f}")

    print(f"\n  CONFUSION MATRIX (Counts):")
    print(f"    True Negatives  (TN): {tn:,}  — Correctly blocked fraud")
    print(f"    False Positives (FP): {fp:,}  — Legitimate blocked incorrectly")
    print(f"    False Negatives (FN): {fn:,}  — MISSED FRAUD ← Critical")
    print(f"    True Positives  (TP): {tp:,}  — Fraud correctly caught")

    print(f"\n  FINANCIAL IMPACT:")
    print(f"    Fraud missed (FN × £{fn_cost:.2f}):     £{fraud_loss:>10,.2f}")
    print(f"    False alarms (FP × £{fp_cost:.2f}):      £{fp_cost_total:>10,.2f}")
    print(f"    Investigation cost:               £{inv_cost_total:>10,.2f}")
    print(f"    ─────────────────────────────────────────────")
    print(f"    TOTAL COST:                       £{total_cost:>10,.2f}")
    print(f"    Max possible loss (no detection): £{max_possible_loss:>10,.2f}")
    print(f"    SAVINGS vs no detection:          £{savings:>10,.2f}")
    print(f"    Fraud detection rate:             {tp/(tp+fn)*100:.1f}%")

    return {
        'model': model_name, 'tn': tn, 'fp': fp, 'fn': fn, 'tp': tp,
        'pr_auc': pr_auc, 'roc_auc': roc_auc, 'f2': f2, 'mcc': mcc,
        'total_cost': total_cost, 'savings': savings
    }

# Apply cost-sensitive evaluation to all supervised models
all_cost_results = []
for name, m in models_supervised.items():
    optimal_thresh = best_f2_thresh
    preds = (m['proba'] >= optimal_thresh).astype(int)
    result = cost_sensitive_report(y_test, preds, m['proba'], name)
    all_cost_results.append(result)

cost_df = pd.DataFrame(all_cost_results)
print(f"\n\nFINAL COST COMPARISON (using F2-optimal threshold):")
print(cost_df[['model', 'pr_auc', 'f2', 'mcc', 'total_cost', 'savings']].to_string(index=False))

# Visualize cost comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['#2ECC71' if s == cost_df['savings'].max() else '#3498DB'
          for s in cost_df['savings']]
axes[0].barh(cost_df['model'], cost_df['savings']/1000, color=colors)
axes[0].set_xlabel('Financial Savings (£000s)')
axes[0].set_title('Financial Savings vs No Detection')

axes[1].barh(cost_df['model'], cost_df['pr_auc'], color='#9B59B6')
axes[1].set_xlabel('Precision-Recall AUC')
axes[1].set_title('PR-AUC by Model')

plt.tight_layout()
plt.savefig('output/cost_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
```
<img width="446" height="360" alt="image" src="https://github.com/user-attachments/assets/b18ceca6-f926-4c97-b747-b59d2a8a2c9f" /><img width="433" height="452" alt="image" src="https://github.com/user-attachments/assets/65a344a3-ff71-4e6b-98a8-3f27a8b0db77" /><img width="427" height="453" alt="image" src="https://github.com/user-attachments/assets/0112097a-801e-4771-b365-319fdb8b865a" /><img width="449" height="455" alt="image" src="https://github.com/user-attachments/assets/432e082b-ea3a-4d40-a098-8d57a463fd5b" /><img width="444" height="282" alt="image" src="https://github.com/user-attachments/assets/604dec9e-0805-459d-8e52-7a2707d1402e" />
<img width="444" height="282" alt="image" src="https://github.com/user-attachments/assets/18c923b5-3188-4b99-a3d9-157ee2e291b5" />

PR curve and ROC curve for all Models Comparisons
``` python
print("\n" + "="*60)
print("SECTION 12: PRECISION-RECALL & ROC CURVES")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
colors_list = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12',
               '#9B59B6', '#1ABC9C', '#E67E22']

# PR Curves
for i, (name, m) in enumerate(models_supervised.items()):
    prec_c, rec_c, _ = precision_recall_curve(y_test, m['proba'])
    pr_auc = average_precision_score(y_test, m['proba'])
    axes[0].plot(rec_c, prec_c, color=colors_list[i % len(colors_list)],
                 label=f"{name} (AUC={pr_auc:.3f})", linewidth=1.8)

# Add unsupervised models to PR curve
for i, (name, m) in enumerate(models_unsupervised.items()):
    prec_c, rec_c, _ = precision_recall_curve(y_test, m['scores'])
    pr_auc = average_precision_score(y_test, m['scores'])
    axes[0].plot(rec_c, prec_c, linestyle='--',
                 color=colors_list[(i+4) % len(colors_list)],
                 label=f"{name} (AUC={pr_auc:.3f})", linewidth=1.2)

axes[0].axhline(fraud_rate, color='grey', linestyle=':', label='Random classifier')
axes[0].set_xlabel('Recall')
axes[0].set_ylabel('Precision')
axes[0].set_title('Precision-Recall Curves (All Models)')
axes[0].legend(fontsize=8)
axes[0].set_xlim([0, 1])
axes[0].set_ylim([0, 1])

# ROC Curves
for i, (name, m) in enumerate(models_supervised.items()):
    fpr, tpr, _ = roc_curve(y_test, m['proba'])
    roc_auc = roc_auc_score(y_test, m['proba'])
    axes[1].plot(fpr, tpr, color=colors_list[i % len(colors_list)],
                 label=f"{name} (AUC={roc_auc:.3f})", linewidth=1.8)

axes[1].plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.500)')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curves (Supervised Models Only)')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig('output/pr_roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()
print("\nKey insight: PR curve is more informative than ROC for imbalanced data")
print("ROC-AUC can be misleadingly high because TN dominates the denominator")
```
<img width="422" height="70" alt="image" src="https://github.com/user-attachments/assets/2d006e9a-e25e-4542-b19e-f77de102f9ad" /><img width="507" height="64" alt="image" src="https://github.com/user-attachments/assets/09383984-c36f-4ac6-b7e4-f76d83108bd3" /><img width="507" height="64" alt="image" src="https://github.com/user-attachments/assets/deac11f6-7389-4500-a6c8-e7a7b7d5b7c6" />

Rule based aspects Machine Learning vs Deep Learning
``` python
print("\n" + "="*60)
print("SECTION 13: DETECTION PHILOSOPHY COMPARISON")
print("="*60)

comparison = {
    'Aspect': [
        'Training Data Required',
        'Interpretability',
        'Novel Fraud Detection',
        'Latency (real-time)',
        'Maintenance',
        'False Positive Rate',
        'Best Use Case',
        'When to Use'
    ],
    'Rule-Based': [
        'None',
        'Full (explicit rules)',
        'Poor (misses new patterns)',
        'Microseconds',
        'High (manual rule updates)',
        'High (rigid thresholds)',
        'Compliance/Regulatory context',
        'When you need full explainability'
    ],
    'ML (Isolation Forest/RF)': [
        'Historical labeled/unlabeled',
        'Medium (feature importance)',
        'Good (learns patterns)',
        'Milliseconds',
        'Medium (periodic retraining)',
        'Tunable via threshold',
        'Production fraud systems',
        'When you have labeled data'
    ],
    'Deep Learning (Autoencoder)': [
        'Large unlabeled data',
        'Low (black box)',
        'Excellent (learns representations)',
        'Tens of milliseconds',
        'Low (self-learns)',
        'Lower (better representation)',
        'High-volume card fraud',
        'When labels are scarce'
    ]
}

comp_df = pd.DataFrame(comparison)
print(comp_df.to_string(index=False))
```
<img width="835" height="215" alt="image" src="https://github.com/user-attachments/assets/9885105e-ebd4-46e0-a95c-8fc1e30a6b25" />

Feature Importance and SHAP Explainability
``` python
print("\n" + "="*60)
print("SECTION 14: FEATURE IMPORTANCE & EXPLAINABILITY")
print("="*60)

# Random Forest Feature Importance
rf_importance = pd.Series(
    rf.feature_importances_, index=feature_cols_ml
).sort_values(ascending=False)

print("\nTop 20 Most Important Features (Random Forest):")
print(rf_importance.head(20))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))

rf_importance.head(20).plot(kind='barh', ax=axes[0], color='steelblue')
axes[0].set_title('Random Forest Feature Importance (Top 20)')
axes[0].set_xlabel('Importance')
axes[0].invert_yaxis()

# SHAP values (if available)
if SHAP_AVAILABLE:
    explainer  = shap.TreeExplainer(rf)
    # Use small sample for speed
    sample_idx = np.random.choice(len(X_test_scaled), 200, replace=False)
    shap_values = explainer.shap_values(X_test_scaled[sample_idx])

    # shap_values[1] = fraud class
    shap_fraud = shap_values[1] if isinstance(shap_values, list) else shap_values

    # SHAP summary plot
    plt.sca(axes[1])
    shap.summary_plot(
        shap_fraud,
        pd.DataFrame(X_test_scaled[sample_idx], columns=feature_cols_ml),
        max_display=15,
        show=False,
        plot_type='bar'
    )
    axes[1].set_title('SHAP Feature Importance (Fraud Class)')
else:
    # XGBoost importance as fallback
    if XGB_AVAILABLE:
        xgb_imp = pd.Series(
            xgb_model.feature_importances_, index=feature_cols_ml
        ).sort_values(ascending=False)
        xgb_imp.head(20).plot(kind='barh', ax=axes[1], color='orange')
        axes[1].set_title('XGBoost Feature Importance (Top 20)')
        axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig('output/feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()
```
<img width="408" height="436" alt="image" src="https://github.com/user-attachments/assets/879333d9-1db0-4013-b205-d8825d54853a" /><img width="408" height="436" alt="image" src="https://github.com/user-attachments/assets/293a7ae5-a355-4937-b518-f212a5d77691" />

Simulation of streaming transaction scoring
``` python
print("\n" + "="*60)
print("SECTION 15: REAL-TIME TRANSACTION SCORING SIMULATION")
print("="*60)

def score_transaction(transaction_dict, model, scaler, feature_cols,
                       threshold=0.5):
    """
    Score a single incoming transaction in real-time.
    Returns risk score, prediction, and decision.
    """
    # Build feature vector
    txn_df = pd.DataFrame([transaction_dict])

    # Engineer features
    txn_df['Hour']          = (txn_df['Time'] // 3600) % 24
    txn_df['Is_Night']      = ((txn_df['Hour'] >= 22) | (txn_df['Hour'] <= 5)).astype(int)
    txn_df['Is_Rush_Hour']  = txn_df['Hour'].between(7, 9).astype(int)
    txn_df['Amount_Log']    = np.log1p(txn_df['Amount'])
    txn_df['Is_Round_Amount']   = (txn_df['Amount'] % 1 == 0).astype(int)
    txn_df['Is_Small_Amount']   = (txn_df['Amount'] < 1).astype(int)
    txn_df['Amount_ZScore']     = 0  # Would need rolling stats in production
    txn_df['V17_V14_interaction']= txn_df['V17'] * txn_df['V14']
    txn_df['V17_Amount_ratio']   = txn_df['V17'] / (txn_df['Amount'] + 1)
    txn_df['V14_V12_interaction']= txn_df['V14'] * txn_df['V12']
    txn_df['Day_Number']         = 0

    X_txn = txn_df[feature_cols].fillna(0)
    X_txn_scaled = scaler.transform(X_txn)

    t_start    = time.time()
    proba      = model.predict_proba(X_txn_scaled)[0][1]
    latency_ms = (time.time() - t_start) * 1000

    decision = 'BLOCK'   if proba >= threshold * 1.5 else \
               'REVIEW'  if proba >= threshold        else \
               'APPROVE'

    return {
        'fraud_probability': round(proba, 4),
        'decision':          decision,
        'latency_ms':        round(latency_ms, 3),
        'risk_tier':         'HIGH' if proba > 0.7 else
                             'MEDIUM' if proba > 0.3 else 'LOW'
    }

# Simulate 20 incoming transactions (mix of real test set examples)
print("\nSimulating real-time transaction scoring...\n")
print(f"{'TXN#':>5} {'Amount':>8} {'Hour':>5} {'Actual':>8} "
      f"{'Fraud_Prob':>11} {'Decision':>10} {'Risk':>7} {'Latency':>9}")
print("-" * 75)

# Sample some transactions with known labels for demo
sample_legit = df_eng[df_eng['Class'] == 0].sample(15, random_state=42)
sample_fraud  = df_eng[df_eng['Class'] == 1].sample(5, random_state=42)
sim_batch     = pd.concat([sample_legit, sample_fraud]).sample(frac=1, random_state=99)

sim_results = []
v_cols = [c for c in df.columns if c.startswith('V')]
for i, (_, row) in enumerate(sim_batch.iterrows()):
    txn = row[v_cols + ['Amount', 'Time']].to_dict()
    result = score_transaction(txn, rf, scaler, feature_cols_ml,
                                threshold=best_f2_thresh)
    actual = 'FRAUD' if row['Class'] == 1 else 'LEGIT'
    hour   = int((row['Time'] // 3600) % 24)

    status = ('✅ CORRECT' if (result['decision'] in ['BLOCK','REVIEW']
                               and actual == 'FRAUD') or
                              (result['decision'] == 'APPROVE'
                               and actual == 'LEGIT') else '❌ WRONG')

    print(f"{i+1:>5} {row['Amount']:>8.2f} {hour:>5} {actual:>8} "
          f"{result['fraud_probability']:>11.4f} {result['decision']:>10} "
          f"{result['risk_tier']:>7} {result['latency_ms']:>7.2f}ms  {status}")

    sim_results.append({**result, 'actual': actual, 'amount': row['Amount']})

sim_df = pd.DataFrame(sim_results)
print(f"\nSimulation Summary:")
print(f"  Avg latency:        {sim_df['latency_ms'].mean():.2f}ms per transaction")
print(f"  Max latency:        {sim_df['latency_ms'].max():.2f}ms")
print(f"  APPROVE decisions:  {(sim_df['decision']=='APPROVE').sum()}")
print(f"  REVIEW decisions:   {(sim_df['decision']=='REVIEW').sum()}")
print(f"  BLOCK decisions:    {(sim_df['decision']=='BLOCK').sum()}")
```
<img width="566" height="477" alt="image" src="https://github.com/user-attachments/assets/9cb4da93-aced-47ea-a7ef-8c0b29f36aab" /><img width="331" height="106" alt="image" src="https://github.com/user-attachments/assets/a7770b45-c6ea-4dbe-8d04-4c4456f8acee" />

Cross Validation with Startified KFold Cross Validation
``` python
print("\n" + "="*60)
print("SECTION 16: STRATIFIED CROSS-VALIDATION")
print("="*60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_models = {
    'Logistic Regression': LogisticRegression(
        max_iter=500, class_weight='balanced', random_state=42),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
}
if XGB_AVAILABLE:
    cv_models['XGBoost'] = xgb.XGBClassifier(
        n_estimators=100, scale_pos_weight=scale_pos,
        random_state=42, verbosity=0, n_jobs=-1)

print(f"\n5-Fold Stratified Cross-Validation (PR-AUC):")
print(f"{'Model':<25} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
print("-" * 60)

for name, model in cv_models.items():
    scores = cross_val_score(
        model, X_train_scaled, y_train,
        cv=skf, scoring='average_precision', n_jobs=-1
    )
    print(f"{name:<25} {scores.mean():>8.4f} {scores.std():>8.4f} "
          f"{scores.min():>8.4f} {scores.max():>8.4f}")
```
<img width="428" height="189" alt="image" src="https://github.com/user-attachments/assets/6b1b7722-afa6-4897-a52c-6ef73e2d2bb0" />

Learning cruves of Bias and Variance Tradeoff
``` python
print("\n" + "="*60)
print("SECTION 17: LEARNING CURVES — BIAS-VARIANCE ANALYSIS")
print("="*60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i, (name, model) in enumerate(list(cv_models.items())[:2]):
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train_scaled, y_train,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='average_precision',
        train_sizes=np.linspace(0.1, 1.0, 8),
        n_jobs=-1
    )
    axes[i].plot(train_sizes, train_scores.mean(axis=1),
                  'o-', color='blue', label='Train PR-AUC')
    axes[i].fill_between(train_sizes,
                          train_scores.mean(axis=1) - train_scores.std(axis=1),
                          train_scores.mean(axis=1) + train_scores.std(axis=1),
                          alpha=0.15, color='blue')
    axes[i].plot(train_sizes, test_scores.mean(axis=1),
                  'o-', color='green', label='Val PR-AUC')
    axes[i].fill_between(train_sizes,
                          test_scores.mean(axis=1) - test_scores.std(axis=1),
                          test_scores.mean(axis=1) + test_scores.std(axis=1),
                          alpha=0.15, color='green')
    axes[i].set_xlabel('Training Set Size')
    axes[i].set_ylabel('PR-AUC')
    axes[i].set_title(f'Learning Curve — {name}')
    axes[i].legend()

plt.tight_layout()
plt.savefig('output/learning_curves.png', dpi=150, bbox_inches='tight')
plt.show()
```
<img width="414" height="70" alt="image" src="https://github.com/user-attachments/assets/27a17385-5274-4ff9-bb71-f00c87302fbd" /><img width="1389" height="490" alt="image" src="https://github.com/user-attachments/assets/26f077e2-01ea-4646-86e3-30a73b340a2b" />

Fraud Amount Analysis
``` python
print("\n" + "="*60)
print("SECTION 18: FRAUD AMOUNT DEEP ANALYSIS")
print("="*60)

fraud_df = df[df['Class'] == 1].copy()
legit_df = df[df['Class'] == 0].copy()

print(f"\nFraud Amount Statistics:")
print(fraud_df['Amount'].describe())
print(f"\nLegitimate Amount Statistics:")
print(legit_df['Amount'].describe())

# Mann-Whitney test — are fraud amounts significantly different?
stat, p = mannwhitneyu(fraud_df['Amount'], legit_df['Amount'])
print(f"\nMann-Whitney U Test (Fraud vs Legit Amount):")
print(f"  Statistic: {stat:.2f}")
print(f"  P-value:   {p:.6f}")
print(f"  Result: {'Significantly different' if p < 0.05 else 'Not significant'}")

# Fraud amount distribution
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].hist(fraud_df['Amount'], bins=50, color='#E74C3C', edgecolor='white')
axes[0].set_title('Fraud Transaction Amount Distribution')
axes[0].set_xlabel('Amount (£)')
axes[0].set_ylabel('Count')

# Fraud by amount bucket
buckets = [0, 1, 10, 50, 100, 500, 1000, float('inf')]
labels  = ['<£1', '£1-10', '£10-50', '£50-100', '£100-500', '£500-1K', '>£1K']
fraud_df['Amount_Bucket'] = pd.cut(fraud_df['Amount'], bins=buckets, labels=labels)
legit_df['Amount_Bucket'] = pd.cut(legit_df['Amount'], bins=buckets, labels=labels)

fraud_bucket = fraud_df['Amount_Bucket'].value_counts().sort_index()
legit_bucket = legit_df['Amount_Bucket'].value_counts().sort_index()

x = np.arange(len(labels))
axes[1].bar(x - 0.2, fraud_bucket.values / fraud_bucket.sum() * 100,
            0.4, label='Fraud', color='#E74C3C', alpha=0.8)
axes[1].bar(x + 0.2, legit_bucket.values / legit_bucket.sum() * 100,
            0.4, label='Legitimate', color='#3498DB', alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels, rotation=30)
axes[1].set_ylabel('% of Transactions')
axes[1].set_title('Amount Distribution: Fraud vs Legit (%)')
axes[1].legend()

# Hourly fraud amount
hourly_amount = df[df['Class']==1].groupby('Hour')['Amount'].mean()
axes[2].bar(hourly_amount.index, hourly_amount.values, color='#E74C3C', alpha=0.8)
axes[2].set_xlabel('Hour of Day')
axes[2].set_ylabel('Avg Fraud Amount (£)')
axes[2].set_title('Average Fraud Amount by Hour of Day')

plt.tight_layout()
plt.savefig('output/fraud_amount_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
```
<img width="412" height="495" alt="image" src="https://github.com/user-attachments/assets/497e5d90-f8c9-4435-81ff-74e4ca7a1d29" /><img width="1589" height="490" alt="image" src="https://github.com/user-attachments/assets/2d5d1e37-3e57-4ca3-9ec8-4fa32309cbf5" />

**Hypothesis testing**
``` python
print("\n" + "="*60)
print("SECTION 19: HYPOTHESIS TESTING — FRAUD PATTERN VALIDATION")
print("="*60)

# Test 1: Is night-time significantly more fraudulent?
night_fraud = df_eng[df_eng['Is_Night'] == 1]['Class']
day_fraud   = df_eng[df_eng['Is_Night'] == 0]['Class']
stat1, p1   = mannwhitneyu(night_fraud, day_fraud, alternative='greater')
night_rate  = night_fraud.mean() * 100
day_rate    = day_fraud.mean() * 100

print(f"\n── Test 1: Is night-time fraud rate higher than daytime? ──")
print(f"  Night fraud rate: {night_rate:.4f}%")
print(f"  Day fraud rate:   {day_rate:.4f}%")
print(f"  Mann-Whitney p-value: {p1:.6f}")
print(f"  Result: {'REJECT H0 — Night significantly more fraudulent' if p1 < 0.05 else 'FAIL TO REJECT H0'}")

# Test 2: Do small amounts (<£1) have higher fraud rates?
small_fraud  = df_eng[df_eng['Is_Small_Amount'] == 1]['Class']
normal_fraud = df_eng[df_eng['Is_Small_Amount'] == 0]['Class']
stat2, p2    = mannwhitneyu(small_fraud, normal_fraud, alternative='greater')
small_rate   = small_fraud.mean() * 100
normal_rate  = normal_fraud.mean() * 100

print(f"\n── Test 2: Do micro-transactions (<£1) indicate fraud testing? ──")
print(f"  Small amount fraud rate: {small_rate:.4f}%")
print(f"  Normal amount fraud rate:{normal_rate:.4f}%")
print(f"  Ratio: {small_rate/normal_rate:.1f}x more fraudulent")
print(f"  Mann-Whitney p-value: {p2:.6f}")
print(f"  Result: {'REJECT H0 — Micro-transactions significantly more fraudulent' if p2 < 0.05 else 'FAIL TO REJECT H0'}")

# Test 3: KS test on V14 — most discriminative feature
v14_fraud = df[df['Class'] == 1]['V14']
v14_legit = df[df['Class'] == 0]['V14']
stat3, p3 = ks_2samp(v14_fraud, v14_legit)

print(f"\n── Test 3: KS Test — V14 distribution (Fraud vs Legit) ──")
print(f"  Fraud V14 mean: {v14_fraud.mean():.4f}")
print(f"  Legit V14 mean: {v14_legit.mean():.4f}")
print(f"  KS Statistic:   {stat3:.4f}")
print(f"  P-value:        {p3:.10f}")
print(f"  Result: {'REJECT H0 — V14 distributions are significantly different' if p3 < 0.05 else 'FAIL TO REJECT H0'}")
print(f"  V14 is the single most discriminative feature for fraud detection")
```
<img width="485" height="416" alt="image" src="https://github.com/user-attachments/assets/99e81d70-6609-4034-b446-a479b078b2d6" />

Final Model Summary
``` python
print("\n" + "="*60)
print("SECTION 20: FINAL MODEL RECOMMENDATION REPORT")
print("="*60)

best_model_result = cost_df.loc[cost_df['savings'].idxmax()]

print(f"""
══════════════════════════════════════════════════════════════
         FRAUD DETECTION — EXECUTIVE SUMMARY REPORT          
══════════════════════════════════════════════════════════════
Dataset:       284,807 transactions | 492 fraud cases      
Fraud Rate:    0.17% (severely imbalanced)                 
══════════════════════════════════════════════════════════════
RECOMMENDED MODEL: {best_model_result['model']:<38}
PR-AUC:        {best_model_result['pr_auc']:.4f}                                    
F2 Score:      {best_model_result['f2']:.4f}                                    
MCC:           {best_model_result['mcc']:.4f}                                    
Financial Savings vs No Detection: £{best_model_result['savings']:,.2f}              
══════════════════════════════════════════════════════════════
KEY DESIGN DECISIONS:                                       
1. Used PR-AUC (not accuracy) — handles imbalance          
2. Used RobustScaler (not StandardScaler) — outliers       
3. Used SMOTE for training resampling                       
4. Optimized threshold for F2 (recall > precision)         
5. Cost matrix: FN=£{FALSE_NEGATIVE_COST:.2f}, FP=£{FALSE_POSITIVE_COST:.2f}                          
══════════════════════════════════════════════════════════════
PRODUCTION RECOMMENDATIONS:                                 
• Deploy with F2-optimal threshold: {best_f2_thresh:.2f}               
• Retrain monthly on new fraud patterns                     
• Monitor data drift on V1-V28 features                    
• Implement 3-tier decision: APPROVE/REVIEW/BLOCK          
• Log all REVIEW decisions for analyst investigation       
══════════════════════════════════════════════════════════════
""")

# Save outputs
df_eng[feature_cols_ml + ['Class']].to_csv('output/engineered_features.csv', index=False)
thresh_df.to_csv('output/threshold_analysis.csv', index=False)
cost_df.to_csv('output/model_cost_comparison.csv', index=False)
sim_df.to_csv('output/realtime_simulation.csv', index=False)

print("\nAll outputs saved to /output/ directory")
print("Files: engineered_features.csv, threshold_analysis.csv,")
print("       model_cost_comparison.csv, realtime_simulation.csv")
print("\n" + "="*60)
print("FRAUD DETECTION ANALYSIS COMPLETE")
print("="*60)
```

<img width="432" height="492" alt="image" src="https://github.com/user-attachments/assets/bfca6468-7977-487f-8689-4cb2a31897c7" /><img width="435" height="135" alt="image" src="https://github.com/user-attachments/assets/2af246b8-0092-4e57-a795-9ad6daa1a72a" />

**SQL**

When you are dealing with big datasize, we have found one of the best way to upload the complete datasize in BIg Query is mentioned below:

- It is by converting the `csv` file into `7Z` or GZ file format to compress it. In our case, with `creditcard.csv` we have converted the file into `creditcard.csv.gz` b y using the below command on the VS code terminal or you can navigate it on Power shell command.

  ``` python
  zcat large_file.gz | split -l 1000000 --filter='gzip > $FILE.gz' - chunk_
  ```
  
- As the data size if huge, we need to split the dataset in small chunks of datafiles for storage issue as mentioned below.
``` python
import pandas as pd
# Define the file name and the number of rows per chunk
file_name = 'creditcard.csv.gz'
chunk_size = 40000  # Adjust this number based on your RAM

# Create an iterator to read the compressed CSV in chunks
data_iterator = pd.read_csv(file_name, compression='gzip', chunksize=chunk_size)

# Loop through the chunks and save them as separate files
for i, chunk in enumerate(data_iterator):
    output_name = f'creditcard_part_{i}.csv'
    chunk.to_csv(output_name, index=False)
    print(f'Saved: {output_name}')
```

- Now, we upload the each tables on undet the project name specified on [Big Query Google cloud](https://cloud.google.com/bigquery) and also on the table format pane under additional settings, we choose column headers as `V2` in order to remove unnecessary spaces and skip rows as 1 for file formating purpose.

- Finally, we need to combine all the files as one table as mentioned below.
``` sql
CREATE TABLE `fraud-project-489006.fraud_detect.complete_table` AS
SELECT * FROM `fraud-project-489006.fraud_detect.table_*`
WHERE _TABLE_SUFFIX BETWEEN '1' AND '6'
```

- Now, let's haver a final check on the data for the complete table we have created.
``` sql
SELECT
*
from
`fraud_detect.complete_table`
```
<img width="1599" height="822" alt="image" src="https://github.com/user-attachments/assets/7b976d9a-da33-4598-b0d5-162c3b294483" /><img width="1148" height="383" alt="image" src="https://github.com/user-attachments/assets/a6ddc690-ce82-4637-982d-ecb3c3cf1032" /><img width="1160" height="396" alt="image" src="https://github.com/user-attachments/assets/5ad65573-8083-45d5-829b-309a13828fdb" /><img width="683" height="394" alt="image" src="https://github.com/user-attachments/assets/0a94793c-adfe-43b7-af98-b8bb68fce25a" />

1). Accuarcy Paradox
``` sql
WITH class_counts AS (
    SELECT
        CAST(Class AS STRING) AS Class_ID, -- Cast to string to allow "N/A" or Model names later
        COUNT(*)                                 AS Transaction_Count,
        COUNT(*) * 100.0 / SUM(COUNT(*)) OVER () AS Pct_Of_Total
    FROM `fraud_detect.complete_table`
    WHERE Class IS NOT NULL  -- Removes the Row 1 'null' from your image
    GROUP BY Class
)

SELECT
    Class_ID,
    CASE Class_ID WHEN '0' THEN 'Legitimate' ELSE 'Fraud' END AS Class_Label,
    Transaction_Count,
    ROUND(Pct_Of_Total, 4) AS Pct_Of_Total
FROM class_counts

UNION ALL

-- Summary Row: Replacing NULLs with descriptive placeholders or 0
SELECT
    'SUMMARY' AS Class_ID, 
    'Naive Model Accuracy' AS Class_Label,
    0 AS Transaction_Count, -- Replaces the NULL in Transaction_Count
    ROUND((SELECT COUNT(*) FROM `fraud_detect.complete_table` WHERE Class = 0) * 100.0 
          / (SELECT COUNT(*) FROM `fraud_detect.complete_table`), 4) AS Pct_Of_Total
FROM (SELECT 1)
```
<img width="1006" height="268" alt="image" src="https://github.com/user-attachments/assets/b4d6da68-e8c6-4f75-8c05-5bf7166d2670" />

2).Data Quality Audit
``` sql
SELECT
  COUNT(*) AS Total_Rows,
  SUM(CASE WHEN Amount IS NULL THEN 1 ELSE 0 END) AS Null_Amount,
  SUM(CASE WHEN Class IS NULL THEN 1 ELSE 0 END) AS Null_Class,
  SUM(CASE WHEN V1 IS NULL THEN 1 ELSE 0 END) AS Null_V1,
  MIN(Amount) AS Min_Amount,
  MAX(Amount) AS Max_Amount,
  AVG(Amount) AS Avg_Amount,
  STDDEV(Amount) AS Std_Amount,
  SUM(Class) AS Total_Fraud,
  COUNT(*) - SUM(Class) AS Total_Legit,
  ROUND(AVG(Class) * 100, 4) AS Fraud_Rate_Pct,
  SUM(CASE WHEN Amount < 1 THEN 1 ELSE 0 END) AS Micro_Transactions,
  SUM(CASE WHEN Amount > 1000 THEN 1 ELSE 0 END) AS High_Value_Transactions
FROM `fraud_detect.complete_table`
WHERE Class IS NOT NULL AND Amount IS NOT NULL;
```
<img width="1473" height="172" alt="image" src="https://github.com/user-attachments/assets/6fd4f736-9853-4433-9dda-4782cc3f9a25" /><img width="622" height="65" alt="image" src="https://github.com/user-attachments/assets/e271d35f-d806-45c4-939d-bbab1de4286b" />


3).Fraud Rate by hour of day
``` sql

WITH hourly_data AS (
    SELECT
        MOD(CAST(creditcard_csv / 3600 AS INT64), 24)      AS Hour_Of_Day,
        Class,
        Amount
    FROM `fraud_detect.complete_table`
),

hourly_stats AS (
    SELECT
        Hour_Of_Day,
        COUNT(*)                                 AS Total_Transactions,
        SUM(Class)                               AS Fraud_Count,
        COUNT(*) - SUM(Class)                    AS Legit_Count,
        ROUND(AVG(Class) * 100, 4)               AS Fraud_Rate_Pct,
        ROUND(AVG(Amount), 2)                    AS Avg_Amount,
        ROUND(AVG(CASE WHEN Class=1 THEN Amount END), 2)
                                                 AS Avg_Fraud_Amount,
        ROUND(SUM(CASE WHEN Class=1 THEN Amount ELSE 0 END), 2)
                                                 AS Total_Fraud_Amount
    FROM hourly_data
    GROUP BY Hour_Of_Day
)

SELECT
    Hour_Of_Day,
    CASE
        WHEN Hour_Of_Day BETWEEN 0 AND 5   THEN 'Night (00-05)'
        WHEN Hour_Of_Day BETWEEN 6 AND 11  THEN 'Morning (06-11)'
        WHEN Hour_Of_Day BETWEEN 12 AND 17 THEN 'Afternoon (12-17)'
        ELSE                                    'Evening (18-23)'
    END                                          AS Time_Period,
    Total_Transactions,
    Fraud_Count,
    Fraud_Rate_Pct,
    Avg_Amount,
    Avg_Fraud_Amount,
    Total_Fraud_Amount,
    -- Rank hours by fraud rate
    RANK() OVER (ORDER BY Fraud_Rate_Pct DESC)  AS Fraud_Rate_Rank,
    -- Flag high-risk hours
    CASE WHEN Fraud_Rate_Pct > 0.3 THEN 'HIGH_RISK'
         WHEN Fraud_Rate_Pct > 0.17 THEN 'ELEVATED'
         ELSE 'NORMAL'
    END                                          AS Risk_Flag
FROM hourly_stats
ORDER BY Hour_Of_Day;
```
<img width="939" height="579" alt="image" src="https://github.com/user-attachments/assets/d29571b9-d750-4ce4-a91e-fccddd7a770d" /><img width="715" height="476" alt="image" src="https://github.com/user-attachments/assets/4ee70f2e-5b1d-4ede-af86-8ec87c536e91" />


4).Fraud Rate by Amount Bucket
``` sql
WITH bucketed AS (
    SELECT
        CASE
            WHEN Amount < 1      THEN '1_Micro (<£1)'
            WHEN Amount < 10     THEN '2_Small (£1-10)'
            WHEN Amount < 50     THEN '3_Medium (£10-50)'
            WHEN Amount < 100    THEN '4_Moderate (£50-100)'
            WHEN Amount < 500    THEN '5_High (£100-500)'
            WHEN Amount < 1000   THEN '6_Very High (£500-1K)'
            ELSE                      '7_Premium (>£1K)'
        END                                      AS Amount_Bucket,
        Class,
        Amount
    FROM `fraud_detect.complete_table`
)

SELECT
    Amount_Bucket,
    COUNT(*)                                     AS Total_Transactions,
    SUM(Class)                                   AS Fraud_Count,
    ROUND(AVG(Class) * 100, 4)                   AS Fraud_Rate_Pct,
    ROUND(AVG(Amount), 2)                        AS Avg_Amount,
    ROUND(SUM(CASE WHEN Class=1 THEN Amount ELSE 0 END), 2)
                                                 AS Total_Fraud_Value,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2)
                                                 AS Pct_Of_All_Transactions,
    ROUND(SUM(Class) * 100.0 / SUM(SUM(Class)) OVER(), 2)
                                                 AS Pct_Of_All_Fraud,
    -- Fraud concentration index (actual % fraud / expected % fraud)
    ROUND(
        AVG(Class)
        / (SUM(SUM(Class)) OVER() / SUM(COUNT(*)) OVER()), 2
    )                                            AS Fraud_Concentration_Index
FROM bucketed
GROUP BY Amount_Bucket
ORDER BY Amount_Bucket;
```

<img width="949" height="381" alt="image" src="https://github.com/user-attachments/assets/3e6d4d64-7c5b-4eba-ad3f-f9cacb5350d4" /><img width="625" height="264" alt="image" src="https://github.com/user-attachments/assets/da7a6e40-a0d3-4039-8ca1-f72d529d24f3" />

5).Rolling Window Fraud Rate by 1 hour
``` sql
WITH
  time_windows AS (
    SELECT
      SUM(Class) AS Fraud_In_Window,
      COUNT(*) AS Transactions_In_Window,
      ROUND(AVG(Class) * 100, 4) AS Fraud_Rate_Pct,
      ROUND(SUM(CASE WHEN Class = 1 THEN Amount ELSE 0 END), 2)
        AS Fraud_Value_In_Window,
      ROUND(AVG(Amount), 2) AS Avg_Amount_In_Window
    FROM `fraud_detect.complete_table`
  )
SELECT
  Fraud_In_Window,
  Transactions_In_Window,
  Fraud_Rate_Pct,
  Fraud_Value_In_Window,
  Avg_Amount_In_Window
FROM time_windows;
```
<img width="919" height="130" alt="image" src="https://github.com/user-attachments/assets/4ba1e2e2-5bb1-44ab-997b-ac243f53916b" />

6).Micro Transaction Fraud Analysis
``` sql
WITH micro_analysis AS (
    SELECT
        CASE WHEN Amount < 1 THEN 'Micro_Transaction'
             ELSE 'Normal_Transaction' END        AS Transaction_Type,
        Class,
        Amount,
        V14, V17
    FROM `fraud_detect.complete_table`
)

SELECT
    Transaction_Type,
    COUNT(*)                                     AS Total_Count,
    SUM(Class)                                   AS Fraud_Count,
    ROUND(AVG(Class) * 100, 4)                   AS Fraud_Rate_Pct,
    ROUND(AVG(Amount), 4)                        AS Avg_Amount,
    ROUND(AVG(V14), 4)                           AS Avg_V14,
    ROUND(AVG(V17), 4)                           AS Avg_V17,
    -- Fraud rate index vs overall fraud rate
    ROUND(AVG(Class) / 0.001727, 2)              AS Fraud_Rate_Index
FROM micro_analysis
GROUP BY Transaction_Type;

-- Detailed micro-transaction fraud profile
SELECT
    ROUND(Amount, 2)                             AS Amount,
    COUNT(*)                                     AS Occurrences,
    SUM(Class)                                   AS Fraud_Count,
    ROUND(AVG(Class) * 100, 2)                   AS Fraud_Rate_Pct
FROM `fraud_detect.complete_table`
WHERE Amount < 1
GROUP BY ROUND(Amount, 2)
ORDER BY Fraud_Rate_Pct DESC
LIMIT 20;
```
<img width="702" height="429" alt="image" src="https://github.com/user-attachments/assets/11787dde-6858-436c-9451-d441401309cf" />

7).V14 Distributin Analysis
``` sql

WITH v14_stats AS (
    SELECT
        Class,
        CASE Class WHEN 0 THEN 'Legitimate' ELSE 'Fraud' END
                                                 AS Class_Label,
        COUNT(*)                                 AS Count,
        ROUND(AVG(V14), 4)                       AS Mean_V14,
        ROUND(STDDEV(V14), 4)                    AS Std_V14,
        ROUND(MIN(V14), 4)                       AS Min_V14,
        ROUND(MAX(V14), 4)                       AS Max_V14,
        -- Percentiles using APPROX_QUANTILES (BigQuery)
        APPROX_QUANTILES(V14, 4)[OFFSET(1)]      AS Q1_V14,
        APPROX_QUANTILES(V14, 4)[OFFSET(2)]      AS Median_V14,
        APPROX_QUANTILES(V14, 4)[OFFSET(3)]      AS Q3_V14
    FROM `fraud_detect.complete_table`
    GROUP BY Class
)

SELECT * FROM v14_stats;

-- V14 threshold analysis: at what value does fraud risk spike?
WITH v14_bucketed AS (
    SELECT
        ROUND(V14, 0)                            AS V14_Rounded,
        Class,
        Amount
    FROM `fraud_detect.complete_table`
    WHERE V14 BETWEEN -20 AND 10
)

SELECT
    V14_Rounded,
    COUNT(*)                                     AS Total,
    SUM(Class)                                   AS Fraud_Count,
    ROUND(AVG(Class) * 100, 3)                   AS Fraud_Rate_Pct,
    ROUND(AVG(CASE WHEN Class=1 THEN Amount END), 2)
                                                 AS Avg_Fraud_Amount
FROM v14_bucketed
GROUP BY V14_Rounded
ORDER BY V14_Rounded;
```
<img width="925" height="489" alt="image" src="https://github.com/user-attachments/assets/d003b3f0-49f3-4925-bab1-138fd48341e6" />

8).Running Cummulative Fraud Detection
``` sql
WITH fraud_totals AS (
    SELECT
        COUNT(*)                                 AS Total_Transactions,
        SUM(Class)                               AS Total_Fraud_Count,
        COUNT(*) - SUM(Class)                    AS Total_Legit_Count,
        SUM(CASE WHEN Class=1 THEN Amount ELSE 0 END)
                                                 AS Total_Fraud_Value,
        AVG(CASE WHEN Class=1 THEN Amount END)   AS Avg_Fraud_Amount
    FROM `fraud_detect.complete_table`
)

SELECT
    'Scenario'                                   AS Label,
    'No Detection (Baseline)'                    AS Scenario,
    Total_Fraud_Count                            AS Fraud_Missed,
    0                                            AS False_Alarms,
    ROUND(Total_Fraud_Value, 2)                  AS Financial_Loss_GBP,
    0.00                                         AS Savings_GBP
FROM fraud_totals

UNION ALL

SELECT
    'Scenario',
    'Good Model (90% Recall, 10% FPR)',
    ROUND(Total_Fraud_Count * 0.10)              AS Fraud_Missed,
    ROUND(Total_Legit_Count * 0.10)              AS False_Alarms,
    ROUND(Total_Fraud_Count * 0.10 * 122.21
          + Total_Legit_Count * 0.10 * 15.00
          + Total_Fraud_Count * 0.90 * 5.00, 2) AS Financial_Loss_GBP,
    ROUND(Total_Fraud_Value * 0.90
          - Total_Legit_Count * 0.10 * 15.00
          - Total_Fraud_Count * 0.90 * 5.00, 2) AS Savings_GBP
FROM fraud_totals

UNION ALL

SELECT
    'Scenario',
    'Perfect Model (100% Recall, 0% FPR)',
    0                                            AS Fraud_Missed,
    0                                            AS False_Alarms,
    ROUND(Total_Fraud_Count * 5.00, 2)           AS Financial_Loss_GBP,
    ROUND(Total_Fraud_Value
          - Total_Fraud_Count * 5.00, 2)         AS Savings_GBP
FROM fraud_totals;
```
<img width="878" height="224" alt="image" src="https://github.com/user-attachments/assets/b6b46940-c542-4b10-9cab-3515aa4d02d5" /><img width="312" height="131" alt="image" src="https://github.com/user-attachments/assets/b88c4d18-a775-4523-84e8-825a501dc58a" />


9).Financial Exposure by Detection
``` sql
WITH feature_stats AS (
    SELECT
        'V1'  AS Feature, AVG(CASE WHEN Class=0 THEN V1  END) AS Legit_Mean,
                          AVG(CASE WHEN Class=1 THEN V1  END) AS Fraud_Mean,
                          STDDEV(V1)                           AS Overall_Std
    FROM `fraud_detect.complete_table`
    UNION ALL
    SELECT 'V2',  AVG(CASE WHEN Class=0 THEN V2  END), AVG(CASE WHEN Class=1 THEN V2  END), STDDEV(V2)  FROM `fraud_detect.complete_table`
    UNION ALL
    SELECT 'V3',  AVG(CASE WHEN Class=0 THEN V3  END), AVG(CASE WHEN Class=1 THEN V3  END), STDDEV(V3)  FROM `fraud_detect.complete_table`
    UNION ALL
    SELECT 'V4',  AVG(CASE WHEN Class=0 THEN V4  END), AVG(CASE WHEN Class=1 THEN V4  END), STDDEV(V4)  FROM `fraud_detect.complete_table`
    UNION ALL
    SELECT 'V7',  AVG(CASE WHEN Class=0 THEN V7  END), AVG(CASE WHEN Class=1 THEN V7  END), STDDEV(V7)  FROM `fraud_detect.complete_table`
    UNION ALL
    SELECT 'V10', AVG(CASE WHEN Class=0 THEN V10 END), AVG(CASE WHEN Class=1 THEN V10 END), STDDEV(V10) FROM `fraud_detect.complete_table`
    UNION ALL
    SELECT 'V11', AVG(CASE WHEN Class=0 THEN V11 END), AVG(CASE WHEN Class=1 THEN V11 END), STDDEV(V11) FROM `fraud_detect.complete_table`
    UNION ALL
    SELECT 'V12', AVG(CASE WHEN Class=0 THEN V12 END), AVG(CASE WHEN Class=1 THEN V12 END), STDDEV(V12) FROM `fraud_detect.complete_table`
    UNION ALL
    SELECT 'V14', AVG(CASE WHEN Class=0 THEN V14 END), AVG(CASE WHEN Class=1 THEN V14 END), STDDEV(V14) FROM `fraud_detect.complete_table`
    UNION ALL
    SELECT 'V16', AVG(CASE WHEN Class=0 THEN V16 END), AVG(CASE WHEN Class=1 THEN V16 END), STDDEV(V16) FROM `fraud_detect.complete_table`
    UNION ALL
    SELECT 'V17', AVG(CASE WHEN Class=0 THEN V17 END), AVG(CASE WHEN Class=1 THEN V17 END), STDDEV(V17) FROM `fraud_detect.complete_table`
    UNION ALL
    SELECT 'V18', AVG(CASE WHEN Class=0 THEN V18 END), AVG(CASE WHEN Class=1 THEN V18 END), STDDEV(V18) FROM `fraud_detect.complete_table`
    UNION ALL
    SELECT 'V19', AVG(CASE WHEN Class=0 THEN V19 END), AVG(CASE WHEN Class=1 THEN V19 END), STDDEV(V19) FROM `fraud_detect.complete_table`
    UNION ALL
    SELECT 'Amount', AVG(CASE WHEN Class=0 THEN Amount END), AVG(CASE WHEN Class=1 THEN Amount END), STDDEV(Amount) FROM `fraud_detect.complete_table`
)

SELECT
    Feature,
    ROUND(Legit_Mean, 4)                         AS Legit_Mean,
    ROUND(Fraud_Mean, 4)                         AS Fraud_Mean,
    ROUND(ABS(Fraud_Mean - Legit_Mean), 4)       AS Absolute_Separation,
    -- Effect size (Cohen's d approximation)
    ROUND(ABS(Fraud_Mean - Legit_Mean) / NULLIF(Overall_Std, 0), 4)
                                                 AS Effect_Size_Cohens_D,
    RANK() OVER (ORDER BY ABS(Fraud_Mean - Legit_Mean) / NULLIF(Overall_Std, 0) DESC)
                                                 AS Discriminative_Rank
FROM feature_stats
ORDER BY Discriminative_Rank;
```
<img width="1013" height="474" alt="image" src="https://github.com/user-attachments/assets/11e4d71e-0cf2-4525-886e-cb33658d1eff" /><img width="151" height="365" alt="image" src="https://github.com/user-attachments/assets/8091b149-69f9-47cf-b207-17a29c230e55" />

10). Cost optimized Threshold Evaluation
``` sql
WITH scored AS (
    SELECT
        Class,
        Amount,
        -- Use -V14 as anomaly score (lower V14 = more likely fraud)
        V14 AS Anomaly_Score
    FROM `fraud_detect.complete_table`
),

thresholds AS (
    SELECT threshold / 10.0 AS Threshold
    FROM UNNEST(GENERATE_ARRAY(-30, 30, 1)) AS threshold
),

threshold_metrics AS (
    SELECT
        t.Threshold,
        SUM(CASE WHEN s.Anomaly_Score >= t.Threshold AND s.Class=1 THEN 1 ELSE 0 END) AS TP,
        SUM(CASE WHEN s.Anomaly_Score >= t.Threshold AND s.Class=0 THEN 1 ELSE 0 END) AS FP,
        SUM(CASE WHEN s.Anomaly_Score <  t.Threshold AND s.Class=1 THEN 1 ELSE 0 END) AS FN,
        SUM(CASE WHEN s.Anomaly_Score <  t.Threshold AND s.Class=0 THEN 1 ELSE 0 END) AS TN
    FROM thresholds t
    CROSS JOIN scored s
    GROUP BY t.Threshold
)

SELECT
    Threshold,
    TP, FP, FN, TN,
    ROUND(TP * 100.0 / NULLIF(TP + FN, 0), 2)   AS Recall_Pct,
    ROUND(TP * 100.0 / NULLIF(TP + FP, 0), 2)   AS Precision_Pct,
    ROUND(2.0 * TP / NULLIF(2*TP + FP + FN, 0), 4) AS F1,
    ROUND(5.0 * TP / NULLIF(5*TP + 4*FN + FP, 0), 4) AS F2,
    -- Financial cost at this threshold
    ROUND(FN * 122.21 + FP * 15.00 + (TP+FP) * 5.00, 2)
                                                 AS Total_Cost_GBP,
    -- Savings vs doing nothing
    ROUND((TP + FN) * 122.21 -
          (FN * 122.21 + FP * 15.00 + (TP+FP) * 5.00), 2)
                                                 AS Net_Savings_GBP
FROM threshold_metrics
WHERE TP > 0
ORDER BY Net_Savings_GBP DESC
LIMIT 20;
```
<img width="1449" height="466" alt="image" src="https://github.com/user-attachments/assets/bb4bc187-3b17-42be-8fc7-5eae25486618" /><img width="301" height="371" alt="image" src="https://github.com/user-attachments/assets/360244f8-08dc-406c-b868-1f6f4096e805" />

11). Fraud Hostspot Detection
``` sql
WITH fraud_zones AS (
    SELECT
        CASE
            WHEN V14 < -10 AND V17 < -5          THEN 'Zone_A_Critical'
            WHEN V14 < -5  AND V17 < -3          THEN 'Zone_B_High'
            WHEN V14 < -3  AND V17 < -1          THEN 'Zone_C_Elevated'
            WHEN V14 BETWEEN -3 AND 0            THEN 'Zone_D_Moderate'
            ELSE                                      'Zone_E_Normal'
        END                                      AS Fraud_Zone,
        Class,
        Amount,
        V14, V17
    FROM `fraud_detect.complete_table`
)

SELECT
    Fraud_Zone,
    COUNT(*)                                     AS Total_Transactions,
    SUM(Class)                                   AS Fraud_Count,
    ROUND(AVG(Class) * 100, 4)                   AS Fraud_Rate_Pct,
    ROUND(AVG(Amount), 2)                        AS Avg_Amount,
    ROUND(SUM(CASE WHEN Class=1 THEN Amount ELSE 0 END), 2)
                                                 AS Total_Fraud_Value,
    ROUND(AVG(V14), 3)                           AS Avg_V14,
    ROUND(AVG(V17), 3)                           AS Avg_V17,
    -- Priority flag for rule-based system
    CASE
        WHEN AVG(Class) > 0.5  THEN 'AUTO_BLOCK'
        WHEN AVG(Class) > 0.10 THEN 'AUTO_REVIEW'
        WHEN AVG(Class) > 0.02 THEN 'ENHANCED_MONITORING'
        ELSE                        'STANDARD_PROCESSING'
    END                                          AS Recommended_Action
FROM fraud_zones
GROUP BY Fraud_Zone
ORDER BY Fraud_Rate_Pct DESC;
```
<img width="1535" height="297" alt="image" src="https://github.com/user-attachments/assets/312cf076-89c1-4047-aefe-fe895768b6db" />

12). Recall vs Precision Trade off 
``` sql
WITH base AS (
    SELECT
        SUM(Class)                               AS Total_Fraud,
        COUNT(*) - SUM(Class)                    AS Total_Legit,
        SUM(CASE WHEN Class=1 THEN Amount ELSE 0 END)
                                                 AS Total_Fraud_Value
    FROM `fraud_detect.complete_table`
)

SELECT
    'High Recall Strategy (catch 95% fraud)'     AS Strategy,
    'Recall=95%, Precision=30%'                  AS Model_Performance,
    ROUND(Total_Fraud * 0.05)                    AS Fraud_Missed,
    ROUND(Total_Fraud * 0.95 / 0.30 * 0.70)     AS False_Alarms,
    ROUND(Total_Fraud * 0.05 * 122.21 +
          Total_Fraud * 0.95 / 0.30 * 0.70 * 15, 2)
                                                 AS Total_Cost_GBP,
    'High customer friction, low fraud loss'     AS Business_Impact
FROM base

UNION ALL

SELECT
    'Balanced Strategy (F1 optimal)',
    'Recall=80%, Precision=85%',
    ROUND(Total_Fraud * 0.20),
    ROUND(Total_Fraud * 0.80 / 0.85 * 0.15),
    ROUND(Total_Fraud * 0.20 * 122.21 +
          Total_Fraud * 0.80 / 0.85 * 0.15 * 15, 2),
    'Balanced — good for most use cases'
FROM base

UNION ALL

SELECT
    'High Precision Strategy (minimize FP)',
    'Recall=50%, Precision=99%',
    ROUND(Total_Fraud * 0.50),
    ROUND(Total_Fraud * 0.50 / 0.99 * 0.01),
    ROUND(Total_Fraud * 0.50 * 122.21 +
          Total_Fraud * 0.50 / 0.99 * 0.01 * 15, 2),
    'Low friction, high fraud loss'
FROM base;
```
<img width="1292" height="244" alt="image" src="https://github.com/user-attachments/assets/67208eb6-90b4-46cd-be40-b6973ecb7ae5" />

**Tableau:**
<img width="1285" height="821" alt="image" src="https://github.com/user-attachments/assets/f26e76d0-b376-4517-9e5c-1f5e94ef5e3b" />


### Insights:

- Extreme Class Imbalance: The dataset is highly skewed, with fraud transactions making up only 0.1727% of the total data (492 fraud cases vs. 284,315 legitimate ones).

- The Accuracy Paradox: A naive model that simply classifies every transaction as "Not Fraud" would achieve an 99.83% accuracy but would fail to detect a single fraudulent transaction, resulting in a total financial loss of approximately £60,127 in this dataset.

- Transaction Profile: While the average transaction amount is $88.35, the average loss from a fraudulent transaction is higher at $122.21.

- Handling Skewed Features: The "Amount" feature contains significant outliers (ranging from $0 to over $25,000). The code utilizes RobustScaler to scale this feature, as it is less sensitive to outliers than standard scaling.

- Anonymized Features: The features $V1$ through $V28$ are results of a PCA transformation, meaning they are already decorrelated and scaled, though their physical meaning is hidden for privacy.

- Advanced Metrics Over Accuracy: Because of the imbalance, the notebook shifts focus from accuracy to:

a). Precision-Recall AUC (PR-AUC): A better indicator of performance on imbalanced datasets than the standard ROC-AUC.

b). F-beta Score: Used to weight Recall more heavily than Precision, prioritizing the detection of fraud even if it increases false alarms slightly.

Imbalance Handling Techniques: The code sets up imblearn pipelines to test several strategies, including:

a). SMOTE (Over-sampling): Creating synthetic fraud examples.

b). Random Under-sampling: Reducing the majority class to balance the training set.

- Algorithmic Approach: The notebook prepares a variety of models, from traditional Logistic Regression and Random Forest to advanced gradient boosting (XGBoost, LightGBM) and anomaly detection methods like Isolation Forest.

- The analysis introduces a "Cost Function" to evaluate models based on real-world business impact rather than just statistical error:

a). False Positive Cost ($15.00): The administrative cost of investigating a legitimate transaction that was flagged.

b). False Negative Cost ($122.21): The direct financial loss of missing a fraudulent transaction.

Investigation Cost ($5.00): The standard overhead for any flagged transaction.

### Recommendations

- Optimize for Financial Impact: Instead of selecting a model based on the highest F1-score, select the model and probability threshold that minimizes the Total Financial Cost (False Positives + False Negatives).

- Deployment of Explainable AI: Integrate SHAP (Shapley Additive Explanations) to provide "reason codes" for why a transaction was flagged. This helps investigators understand model decisions and provides transparency if a customer's card is blocked.

- Tiered Response System: Use the model's probability scores to trigger different actions:

a). Low Score: Auto-approve.

b). Medium Score: Trigger Multi-Factor Authentication (MFA) or "Step-up" verification.

c). High Score: Decline transaction and alert a human investigator.

- Continuous Re-sampling: Regularly retrain the model using SMOTETomek or ADASYN to ensure the model stays robust against evolving fraud patterns while maintaining a clean decision boundary.
