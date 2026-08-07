
import pandas as pd


def run_data_quality_checks(df):

    issues = []

    # -----------------------------
    # Missing values
    # -----------------------------

    missing_values = df.isnull().sum()

    missing_columns = (
        missing_values[missing_values > 0]
        .to_dict()
    )

    if missing_columns:
        issues.append({
            "type": "missing_values",
            "details": missing_columns
        })

    # -----------------------------
    # Duplicate customer IDs
    # -----------------------------

    duplicate_count = df["customerID"].duplicated().sum()

    if duplicate_count > 0:
        issues.append({
            "type": "duplicate_customer_ids",
            "count": int(duplicate_count)
        })

    # -----------------------------
    # Invalid tenure
    # -----------------------------

    invalid_tenure = (
        (df["tenure"] < 0) |
        (df["tenure"] > 72)
    ).sum()

    if invalid_tenure > 0:
        issues.append({
            "type": "invalid_tenure",
            "count": int(invalid_tenure)
        })

    # -----------------------------
    # Invalid monthly charges
    # -----------------------------

    invalid_charges = (
        df["MonthlyCharges"] < 0
    ).sum()

    if invalid_charges > 0:
        issues.append({
            "type": "invalid_monthly_charges",
            "count": int(invalid_charges)
        })

    # -----------------------------
    # Invalid target values
    # -----------------------------

    if "Churn" in df.columns:

        valid_targets = {"Yes", "No"}

        invalid_targets = (
            ~df["Churn"].isin(valid_targets)
        ).sum()

        if invalid_targets > 0:
            issues.append({
                "type": "invalid_churn_values",
                "count": int(invalid_targets)
            })

    # -----------------------------
    # Overall result
    # -----------------------------

    status = (
        "PASS"
        if len(issues) == 0
        else "WARNING"
    )

    return {
        "status": status,
        "issues": issues
    }


if __name__ == "__main__":

    data = pd.read_csv(
        "data/processed/training_data.csv"
    )

    result = run_data_quality_checks(data)

    print(result)
