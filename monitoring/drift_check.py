
import pandas as pd


def calculate_drift(
    reference_data,
    recent_data,
    feature,
    threshold=0.20
):

    reference_mean = reference_data[feature].mean()
    recent_mean = recent_data[feature].mean()

    # Relative change in mean
    if reference_mean == 0:
        relative_change = 0
    else:
        relative_change = abs(
            recent_mean - reference_mean
        ) / abs(reference_mean)

    drift_detected = (
        relative_change > threshold
    )

    return {
        "feature": feature,
        "reference_mean": round(
            float(reference_mean),
            4
        ),
        "recent_mean": round(
            float(recent_mean),
            4
        ),
        "relative_change": round(
            float(relative_change),
            4
        ),
        "threshold": threshold,
        "drift_detected": drift_detected
    }


if __name__ == "__main__":

    reference = pd.read_csv(
        "data/processed/telco_churn_clean.csv"
    )

    recent = pd.read_csv(
        "data/new_data/daily_batch.csv"
    )

    result = calculate_drift(
        reference,
        recent,
        "MonthlyCharges"
    )

    print("Feature drift result:")
    print(result)
