
import os
import json
from datetime import datetime

import pandas as pd


# -----------------------------
# Configuration
# -----------------------------

INPUT_FILE = "data/new_data/daily_batch.csv"
OUTPUT_FILE = "data/processed/training_data.csv"
LOG_FILE = "artifacts/ingestion_log.json"

REQUIRED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn"
]


def validate_schema(data):
    """
    Validate that all required columns are present.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


def ingest_data():
    """
    Read a new batch and append it to the training dataset.
    """

    # Check input file
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    # Read new batch
    new_data = pd.read_csv(INPUT_FILE)

    rows_received = len(new_data)

    # Validate schema
    validate_schema(new_data)

    # Remove duplicate customer IDs within the new batch
    new_data = new_data.drop_duplicates(
        subset=["customerID"]
    )

    rows_after_deduplication = len(new_data)

    # Create output directory
    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    # Append to existing training data
    if os.path.exists(OUTPUT_FILE):

        existing_data = pd.read_csv(
            OUTPUT_FILE
        )

        combined_data = pd.concat(
            [existing_data, new_data],
            ignore_index=True
        )

        # Keep latest record for each customer
        combined_data = combined_data.drop_duplicates(
            subset=["customerID"],
            keep="last"
        )

    else:

        combined_data = new_data

    # Save updated training data
    combined_data.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # Create ingestion log
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "input_file": INPUT_FILE,
        "rows_received": rows_received,
        "rows_after_deduplication": rows_after_deduplication,
        "total_rows_after_ingestion": len(combined_data),
        "status": "success"
    }

    # Load existing logs if available
    if os.path.exists(LOG_FILE):

        with open(LOG_FILE, "r") as file:
            logs = json.load(file)

    else:
        logs = []

    logs.append(log_entry)

    os.makedirs(
        os.path.dirname(LOG_FILE),
        exist_ok=True
    )

    with open(LOG_FILE, "w") as file:
        json.dump(
            logs,
            file,
            indent=4
        )

    print("Ingestion completed successfully.")
    print(f"Rows received: {rows_received}")
    print(
        f"Rows after deduplication: "
        f"{rows_after_deduplication}"
    )
    print(
        f"Total training rows: "
        f"{len(combined_data)}"
    )
    print(f"Log saved to: {LOG_FILE}")


if __name__ == "__main__":
    ingest_data()
