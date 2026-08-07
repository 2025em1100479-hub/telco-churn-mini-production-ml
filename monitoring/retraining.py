
def should_retrain(
    new_rows,
    drift_detected,
    recent_auc=None,
    baseline_auc=0.8509,
    auc_drop_threshold=0.05,
    minimum_new_rows=500
):
    """
    Determine whether model retraining should be triggered.

    Retraining occurs if:
    - at least 500 new rows are available, OR
    - feature drift is detected, OR
    - recent ROC-AUC falls by at least 0.05
      from the baseline validation ROC-AUC.
    """

    # Trigger 1: enough new data
    if new_rows >= minimum_new_rows:
        return True, "Enough new data accumulated"

    # Trigger 2: feature drift
    if drift_detected:
        return True, "Significant feature drift detected"

    # Trigger 3: model performance degradation
    if recent_auc is not None:
        if recent_auc <= (
            baseline_auc - auc_drop_threshold
        ):
            return True, "Recent model AUC has degraded"

    return False, "No retraining trigger"


if __name__ == "__main__":

    retrain, reason = should_retrain(
        new_rows=100,
        drift_detected=False,
        recent_auc=0.84
    )

    print("Retrain:", retrain)
    print("Reason:", reason)
