
import sys
import os

# Add project root to Python path
PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import pandas as pd

from features.preprocess import create_features


def test_feature_creation():

    data = pd.DataFrame({
        "gender": ["Female"],
        "SeniorCitizen": [0],
        "Partner": ["Yes"],
        "Dependents": ["No"],
        "tenure": [12],
        "PhoneService": ["Yes"],
        "MultipleLines": ["No"],
        "InternetService": ["DSL"],
        "OnlineSecurity": ["Yes"],
        "OnlineBackup": ["No"],
        "DeviceProtection": ["Yes"],
        "TechSupport": ["No"],
        "StreamingTV": ["Yes"],
        "StreamingMovies": ["No"],
        "Contract": ["Month-to-month"],
        "PaperlessBilling": ["Yes"],
        "PaymentMethod": ["Electronic check"],
        "MonthlyCharges": [70.0],
        "TotalCharges": [840.0],
        "Churn": ["No"]
    })

    result = create_features(data)

    expected_features = [
        "AverageMonthlySpend",
        "ServiceCount",
        "IsMonthToMonth",
        "HasOnlineSecurity",
        "HasTechSupport",
        "IsHighMonthlyCharge",
        "TenureGroup"
    ]

    for feature in expected_features:
        assert feature in result.columns
