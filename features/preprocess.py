
import pandas as pd
import numpy as np


# Fixed threshold learned from the training data.
# This prevents training-serving skew.
MONTHLY_CHARGE_THRESHOLD = 70.35


def create_features(df):
    """
    Create production-style features for customer churn prediction.

    The same function is used during training and inference.
    """

    data = df.copy()

    # -----------------------------------------
    # Data cleaning
    # -----------------------------------------

    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"],
        errors="coerce"
    )

    data["TotalCharges"] = data["TotalCharges"].fillna(0)

    # Avoid division by zero
    tenure_safe = data["tenure"].replace(0, 1)

    # -----------------------------------------
    # Feature 1:
    # Average Monthly Spend
    # -----------------------------------------

    data["AverageMonthlySpend"] = (
        data["TotalCharges"] / tenure_safe
    )

    # -----------------------------------------
    # Feature 2:
    # Service Count
    # -----------------------------------------

    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies"
    ]

    data["ServiceCount"] = 0

    for column in service_columns:

        data["ServiceCount"] += (
            data[column]
            .isin(["Yes"])
            .astype(int)
        )

    # -----------------------------------------
    # Feature 3:
    # Month-to-month contract
    # -----------------------------------------

    data["IsMonthToMonth"] = (
        data["Contract"] == "Month-to-month"
    ).astype(int)

    # -----------------------------------------
    # Feature 4:
    # Online security
    # -----------------------------------------

    data["HasOnlineSecurity"] = (
        data["OnlineSecurity"] == "Yes"
    ).astype(int)

    # -----------------------------------------
    # Feature 5:
    # Technical support
    # -----------------------------------------

    data["HasTechSupport"] = (
        data["TechSupport"] == "Yes"
    ).astype(int)

    # -----------------------------------------
    # Feature 6:
    # High monthly charge
    # -----------------------------------------

    data["IsHighMonthlyCharge"] = (
        data["MonthlyCharges"]
        > MONTHLY_CHARGE_THRESHOLD
    ).astype(int)

    # -----------------------------------------
    # Feature 7:
    # Tenure group
    # -----------------------------------------

    data["TenureGroup"] = pd.cut(
        data["tenure"],
        bins=[-1, 6, 24, 48, 72],
        labels=[
            "New",
            "Early",
            "Established",
            "Loyal"
        ]
    ).astype(str)

    return data
