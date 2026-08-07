
# Telco Customer Churn – Mini Production ML System

## 1. Project Overview

This project implements a mini production-oriented machine learning system for predicting customer churn.

The system demonstrates the complete ML lifecycle:

- Data ingestion
- Data quality validation
- Feature engineering
- Model training
- Offline model evaluation
- Model selection
- Model artifact storage
- FastAPI online inference
- Latency and throughput benchmarking
- Data and feature monitoring
- Retraining trigger logic

The project uses the Telco Customer Churn dataset and implements binary classification to predict whether a customer is likely to churn.

---

## 2. Problem Definition

Customer churn prediction is a binary classification problem.

The model predicts:

- `0` → Customer is unlikely to churn
- `1` → Customer is likely to churn

The system is designed to support customer-service or retention teams by providing a churn probability for an individual customer.

### Primary evaluation metric

**ROC-AUC** is used as the primary model-selection metric because it evaluates the model's ability to distinguish between customers who churn and customers who do not across different classification thresholds.

Additional metrics include:

- Accuracy
- Precision
- Recall
- F1-score

---

## 3. Architecture

The overall system architecture is shown below.

![Telco Churn ML System Architecture](docs/architecture.png)

The pipeline follows:

**Data Source → Ingestion → Data Quality → Feature Engineering → Training → Evaluation → Model Artifact → FastAPI Serving → Monitoring → Retraining**

---

## 4. Dataset

The project uses the Telco Customer Churn dataset.

The dataset contains customer demographic information, account information, service information and the target churn label.

Important fields include:

- Customer ID
- Gender
- Senior citizen status
- Partner and dependent status
- Tenure
- Internet service
- Contract type
- Payment method
- Monthly charges
- Total charges
- Churn

The `Churn` column is the target variable.

---

## 5. Feature Engineering

The project creates additional features from the raw customer attributes.

Examples include:

- `AverageMonthlySpend`
- `ServiceCount`
- `IsMonthToMonth`
- `HasOnlineSecurity`
- `HasTechSupport`
- `IsHighMonthlyCharge`
- `TenureGroup`

Feature preprocessing is shared between training and inference to reduce the risk of training-serving skew.

The preprocessing pipeline handles:

- Numerical imputation
- Numerical scaling
- Categorical encoding
- Unknown categorical values

---

## 6. Model Training

Two models are evaluated:

### Baseline

**Logistic Regression**

### Candidate

**Random Forest**

The training pipeline:

1. Loads the cleaned dataset
2. Creates engineered features
3. Splits the data into training, validation and test sets
4. Applies preprocessing
5. Trains the baseline model
6. Trains the candidate model
7. Compares validation metrics
8. Applies a model promotion rule
9. Evaluates the selected model on the test set
10. Saves the production model artifact

---

## 7. Model Promotion Rule

The candidate model is promoted only when:

- ROC-AUC is at least `0.80`
- Candidate ROC-AUC is not more than `0.01` worse than the baseline

In the experiment:

| Model | Validation ROC-AUC |
|---|---:|
| Logistic Regression | 0.8509 |
| Random Forest | 0.8380 |

The Random Forest candidate was therefore **not promoted**.

The selected production model is:

**Logistic Regression**

---

## 8. Final Test Performance

The selected Logistic Regression model achieved the following test results:

| Metric | Result |
|---|---:|
| Accuracy | 0.7985 |
| Precision | 0.6650 |
| Recall | 0.4875 |
| F1-score | 0.5626 |
| ROC-AUC | 0.8464 |

The model is not intended to be state-of-the-art. The main objective is to demonstrate a reproducible production ML workflow.

---

## 9. Online Inference

The trained model is exposed through a FastAPI service.

### Endpoint

```text
POST /predict
