# Model Card: Credit Card Fraud Detection using Anomaly Detection

---

## 1. Model Details

| Field | Details |
|---|---|
| **Framework Name** | Credit Card Fraud Detection — Multi-Philosophy ML Framework |
| **Python Version** | 3.10 |
| **Analysis Date** | June 2026 |
| **Model Type** | Ensemble of Supervised Classifiers + Unsupervised Anomaly Detectors |
| **Recommended Model** | Logistic Regression (highest financial savings at F2-optimal threshold) |
| **Primary Metric** | Precision-Recall AUC (PR-AUC) — chosen over accuracy to handle extreme class imbalance |
| **Secondary Metrics** | F2 Score, MCC, ROC-AUC, Financial Savings vs No Detection |

---

## 2. Intended Use

- **Primary Use Case:** Real-time classification of credit card transactions as fraudulent or legitimate via a FastAPI microservice deployed on AWS EC2.
- **Target Users:** Risk & Fraud Analysts, Data Scientists, FinTech Engineering Teams.
- **Out of Scope:** Multi-class fraud categorization, identity theft detection beyond transaction-level signals, or multi-armed bandit / highly correlated cluster-randomized designs.

---

## 3. Dataset

| Property | Value |
|---|---|
| **Total Transactions** | 284,807 |
| **Fraud Cases** | 492 (0.1727%) |
| **Legitimate Cases** | 284,315 (99.83%) |
| **Duplicate Rows Removed** | 1,081 |
| **Missing Values** | 0 across all 31 columns |
| **Features** | V1–V28 (PCA-transformed, anonymized) + `Time` + `Amount` |
| **Time Span** | ~48 hours (172,792 seconds) |
| **Amount Range** | £0.00 – £25,691.16 |
| **Amount Mean** | £88.35 |
| **Amount Median** | £22.00 |

> **Note:** Features V1–V28 are the result of a PCA transformation applied for privacy compliance. Original feature names are not available.

---

## 4. The Accuracy Paradox

A naive model that predicts every transaction as legitimate achieves:

| Metric | Naive Model |
|---|---|
| Accuracy | **99.83%** ← Misleadingly high |
| Precision | 0.00% |
| Recall | 0.00% |
| F1 Score | 0.00 |
| **Financial Loss** | **£60,127.32** (all fraud undetected) |

This is why **PR-AUC** and **F2 Score** are used as primary metrics — they directly measure the ability to catch fraud cases, not the overwhelming majority of legitimate transactions.

---

## 5. Feature Engineering

**Original features:** 32 → **Engineered features:** 45 (13 new features added)

| Feature | Type | Fraud Mean | Legit Mean | Difference |
|---|---|---|---|---|
| `Is_Night` | Time-based (hour ≥ 22 or ≤ 5) | 0.313 | 0.176 | +0.137 |
| `Is_Rush_Hour` | Time-based (7–9am, 5–7pm) | — | — | — |
| `Day_Number` | Time-based (Day 0 or 1) | — | — | — |
| `Amount_Log` | log1p(Amount) | 2.821 | 3.153 | -0.332 |
| `Amount_Squared` | Amount² | — | — | — |
| `Is_Round_Amount` | Amount % 1 == 0 | — | — | — |
| `Is_Small_Amount` | Amount < £1 (micro-test txns) | 0.138 | 0.059 | +0.079 |
| `Amount_Rolling_Mean` | Rolling 100-window mean | — | — | — |
| `Amount_Rolling_Std` | Rolling 100-window std | — | — | — |
| `Amount_ZScore` | Rolling z-score of amount | 0.191 | 0.000 | +0.191 |
| `V17_V14_interaction` | V17 × V14 | — | — | — |
| `V17_Amount_ratio` | V17 / (Amount + 1) | — | — | — |
| `V14_V12_interaction` | V14 × V12 | — | — | — |

---

## 6. Methodology & Pipeline Architecture

```
creditcard.csv
     │
     ▼
Data Quality Audit (duplicates, nulls, class distribution)
     │
     ▼
Accuracy Paradox Demonstration
     │
     ▼
Feature Engineering (32 → 45 features)
     │
     ▼
Train/Test Split (80/20, stratified) ──► RobustScaler
     │                                   (chosen over StandardScaler
     │                                    due to Amount outliers up to £25,691)
     ▼
Imbalance Strategy Comparison (SMOTE, BorderlineSMOTE, ADASYN, SMOTETomek)
     │
     ▼
┌─────────────────────────────────────────────────┐
│  Philosophy 1: Rule-Based Detection             │
│  Philosophy 2: Traditional ML (Supervised +     │
│                Unsupervised Anomaly Detection)   │
│  Philosophy 3: Autoencoder (TF — optional)      │
└─────────────────────────────────────────────────┘
     │
     ▼
Threshold Optimisation (F2-optimal: 0.38)
     │
     ▼
Cost-Sensitive Evaluation (FN=£122.21, FP=£15.00, Inv=£5.00)
     │
     ▼
Cross-Validation (5-fold stratified)
     │
     ▼
Real-Time Simulation (APPROVE / REVIEW / BLOCK)
     │
     ▼
Hypothesis Testing (Mann-Whitney, KS Test)
     │
     ▼
Final Recommendation & MLflow Logging
```

---

## 7. Imbalance Handling — Strategy Comparison

All strategies were evaluated using Logistic Regression PR-AUC on the held-out test set:

| Strategy | PR-AUC |
|---|---|
| No Resampling | **0.7602** ← Best |
| SMOTE | 0.6985 |
| BorderlineSMOTE | 0.7241 |
| ADASYN | 0.6996 |
| SMOTETomek | 0.6985 |

> **Decision:** No resampling used for final training. `class_weight='balanced'` handles imbalance within the model itself without synthetic oversampling.

**Training / Test Split:**
- Training: 227,845 transactions | 394 fraud (0.173%)
- Test: 56,962 transactions | 98 fraud (0.172%)
- Scaler: **RobustScaler** (median + IQR — robust to extreme Amount outliers)

---

## 8. Model Results

### 8a. Rule-Based Detector (Baseline)

| Metric | Value |
|---|---|
| PR-AUC | 0.0955 |
| Recall | 76.0% |
| Precision | 12.5% |
| Missed Fraud (FN) | 118 × £122.21 = £14,420.78 |
| False Alarms (FP) | 2,617 × £15.00 = £39,255.00 |
| **Total Cost** | **£53,675.78** |

> Limitation: Rules are static, brittle, and miss novel fraud patterns.

---

### 8b. Unsupervised Anomaly Detectors

| Model | PR-AUC | Recall | Precision | Train Time |
|---|---|---|---|---|
| Isolation Forest | 0.2204 | 37.76% | 26.24% | 4.17s |
| Local Outlier Factor | 0.0448 | 10.20% | 9.01% | 92.63s |
| One-Class SVM | 0.3798 | **93.88%** | 1.19% | 21.13s |

> One-Class SVM achieves highest recall but at extremely low precision — too many false alarms for production use.

---

### 8c. Supervised ML Models

| Model | PR-AUC | ROC-AUC | Train Time |
|---|---|---|---|
| Logistic Regression | 0.6985 | 0.9712 | 36.28s |
| Random Forest | 0.8319 | 0.9775 | 508.29s |
| XGBoost | 0.8805 | 0.9766 | 11.88s |
| **LightGBM** | **0.8863** | 0.9721 | 4.73s |

---

### 8d. 5-Fold Stratified Cross-Validation (PR-AUC)

| Model | Mean | Std | Min | Max |
|---|---|---|---|---|
| Logistic Regression | 0.7660 | ±0.0381 | 0.6955 | 0.7998 |
| Random Forest | 0.8376 | ±0.0319 | 0.7975 | 0.8943 |
| XGBoost | **0.8454** | ±0.0243 | 0.8135 | 0.8870 |

---

### 8e. Cost-Sensitive Evaluation (F2-optimal threshold = 0.38)

| Model | PR-AUC | F2 Score | MCC | Financial Savings |
|---|---|---|---|---|
| **Logistic Regression** | 0.6985 | **0.1329** | **0.1611** | **£10,998.90** |
| Random Forest | 0.8319 | — | — | — |
| XGBoost | 0.8805 | — | — | — |
| LightGBM | 0.8863 | — | — | — |

> **Recommended model: Logistic Regression** — highest financial savings at the F2-optimal threshold of 0.38, balancing recall (catching fraud) against precision (avoiding false alarms).

**Cost Matrix Used:**

| Cost Type | Value |
|---|---|
| False Negative (missed fraud) | £122.21 per transaction |
| False Positive (blocked legit) | £15.00 per transaction |
| Investigation cost | £5.00 per flagged transaction |
| Max possible loss (no detection) | £60,127.32 |

---

## 9. Hypothesis Testing — Fraud Pattern Validation

| Test | Finding | p-value | Result |
|---|---|---|---|
| **Test 1:** Night-time vs daytime fraud rate | Night: 0.3061% vs Day: 0.1441% | 0.000000 | ✅ REJECT H₀ — Night significantly more fraudulent |
| **Test 2:** Micro-transactions (<£1) fraud rate | Small: 0.4047% vs Normal: 0.1582% (2.6× ratio) | 0.000000 | ✅ REJECT H₀ — Micro-transactions significantly more fraudulent |
| **Test 3:** KS Test on V14 distribution | Fraud mean: -6.9717 vs Legit mean: 0.0121 · KS=0.8428 | 0.0000000000 | ✅ REJECT H₀ — V14 is the single most discriminative feature |

---

## 10. Real-Time Simulation — 3-Tier Decision Framework

Transactions are scored and routed into one of three tiers:

| Decision | Condition | Action |
|---|---|---|
| ✅ **APPROVE** | fraud_probability < lower threshold | Transaction passes automatically |
| 🔍 **REVIEW** | fraud_probability in middle band | Flagged for analyst investigation (£5.00 cost) |
| 🚫 **BLOCK** | fraud_probability ≥ upper threshold | Transaction blocked (£15.00 FP cost if wrong) |

---

## 11. Ethical Considerations & Limitations

- **Class Imbalance Bias:** At 0.17% fraud rate, even small threshold shifts dramatically affect recall vs precision. Threshold must be tuned per operational cost tolerance.
- **PCA Anonymization:** V1–V28 are PCA-transformed for privacy — interpretability is limited without the original feature names. SHAP values can still rank feature importance on the transformed space.
- **Temporal Drift:** The dataset spans only 2 days. Real-world fraud patterns evolve continuously; monthly retraining is strongly recommended.
- **False Positive Cost:** Blocking legitimate transactions directly impacts customer experience (£15/case). The 3-tier APPROVE/REVIEW/BLOCK framework mitigates this.
- **No Demographic Features:** Dataset contains no protected attributes — formal fairness analysis is not applicable, but should be revisited if raw features become available.
- **Night-Time Bias:** Statistically, night-time transactions have 2.1× higher fraud rates, but blocking them wholesale would unfairly impact shift workers and international cardholders.

---

## 12. Infrastructure & Tools

| Category | Tool / Library |
|---|---|
| Language | Python 3.10 |
| ML | Scikit-learn (LR, RF, IsoForest, LOF, OneClassSVM) |
| Boosting | XGBoost, LightGBM |
| Imbalance | imbalanced-learn (SMOTE, ADASYN, BorderlineSMOTE, SMOTETomek) |
| Statistics | SciPy (Mann-Whitney U, KS Test) |
| API Framework | FastAPI + Uvicorn |
| Experiment Tracking | MLflow |
| Testing | Pytest + pytest-cov (≥85% coverage enforced) |
| Coverage Reporting | Codecov |
| CI/CD | GitHub Actions |
| Containerisation | Docker |
| Cloud Infrastructure | AWS EC2 |
| Model Serialisation | Joblib |
| Data Processing | Pandas, NumPy |
| Visualisation | Matplotlib, Seaborn |

---

## 13. Final Decision Summary

```
══════════════════════════════════════════════════════════════
         FRAUD DETECTION — EXECUTIVE SUMMARY REPORT
══════════════════════════════════════════════════════════════
Dataset:       284,807 transactions | 492 fraud cases
Fraud Rate:    0.17% (severely imbalanced)
══════════════════════════════════════════════════════════════
RECOMMENDED MODEL:  Logistic Regression
PR-AUC:             0.6985
F2 Score:           0.1329
MCC:                0.1611
Financial Savings vs No Detection: £10,998.90
══════════════════════════════════════════════════════════════
KEY DESIGN DECISIONS:
1. PR-AUC used (not accuracy) — handles imbalance
2. RobustScaler used (not StandardScaler) — Amount outliers
3. No resampling: class_weight='balanced' outperforms SMOTE
4. Threshold optimised for F2 (recall > precision)
5. Cost matrix: FN=£122.21, FP=£15.00
══════════════════════════════════════════════════════════════
PRODUCTION RECOMMENDATIONS:
• Deploy with F2-optimal threshold: 0.38
• Retrain monthly on new fraud patterns
• Monitor data drift on V1–V28 features
• Implement 3-tier decision: APPROVE / REVIEW / BLOCK
• Log all REVIEW decisions for analyst investigation
══════════════════════════════════════════════════════════════
```
