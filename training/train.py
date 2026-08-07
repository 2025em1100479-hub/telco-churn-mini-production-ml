import sys
import os

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from features.preprocess import create_features


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_PATH = "data/processed/telco_churn_clean.csv"
MODEL_PATH = "models/churn_model.pkl"
EVAL_PATH = "artifacts/eval/evaluation.json"

RANDOM_STATE = 42


# --------------------------------------------------
# Evaluation function
# --------------------------------------------------

def evaluate_model(model, X_data, y_data):

    predictions = model.predict(X_data)
    probabilities = model.predict_proba(X_data)[:, 1]

    return {
        "accuracy": accuracy_score(
            y_data,
            predictions
        ),
        "precision": precision_score(
            y_data,
            predictions
        ),
        "recall": recall_score(
            y_data,
            predictions
        ),
        "f1": f1_score(
            y_data,
            predictions
        ),
        "roc_auc": roc_auc_score(
            y_data,
            probabilities
        )
    }


# --------------------------------------------------
# Main training pipeline
# --------------------------------------------------

def train():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Loaded {len(df)} rows.")

    # ----------------------------------------------
    # Feature engineering
    # ----------------------------------------------

    df = create_features(df)

    # ----------------------------------------------
    # Define target
    # ----------------------------------------------

    X = df.drop(
        columns=[
            "customerID",
            "Churn"
        ]
    )

    y = df["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    # ----------------------------------------------
    # Identify feature types
    # ----------------------------------------------

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    # ----------------------------------------------
    # Train / validation / test split
    # ----------------------------------------------

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=y
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=RANDOM_STATE,
        stratify=y_temp
    )

    # ----------------------------------------------
    # Preprocessing
    # ----------------------------------------------

    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numerical_features
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features
            )
        ]
    )

    # ----------------------------------------------
    # Baseline model
    # ----------------------------------------------

    baseline_model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE
                )
            )
        ]
    )

    print("Training Logistic Regression...")

    baseline_model.fit(
        X_train,
        y_train
    )

    # ----------------------------------------------
    # Candidate model
    # ----------------------------------------------

    candidate_model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                    n_jobs=-1
                )
            )
        ]
    )

    print("Training Random Forest...")

    candidate_model.fit(
        X_train,
        y_train
    )

    # ----------------------------------------------
    # Validation evaluation
    # ----------------------------------------------

    baseline_metrics = evaluate_model(
        baseline_model,
        X_val,
        y_val
    )

    candidate_metrics = evaluate_model(
        candidate_model,
        X_val,
        y_val
    )

    print("\nBaseline:")
    print(baseline_metrics)

    print("\nCandidate:")
    print(candidate_metrics)

    # ----------------------------------------------
    # Promotion rule
    # ----------------------------------------------

    minimum_auc = 0.80
    allowed_auc_drop = 0.01

    baseline_auc = baseline_metrics["roc_auc"]
    candidate_auc = candidate_metrics["roc_auc"]

    promote_candidate = (
        candidate_auc >= minimum_auc
        and
        candidate_auc >= (
            baseline_auc - allowed_auc_drop
        )
    )

    if promote_candidate:

        production_model = candidate_model
        production_model_name = "random_forest"

    else:

        production_model = baseline_model
        production_model_name = "logistic_regression"

    print(
        f"\nProduction model: "
        f"{production_model_name}"
    )

    # ----------------------------------------------
    # Final test evaluation
    # ----------------------------------------------

    test_metrics = evaluate_model(
        production_model,
        X_test,
        y_test
    )

    print("\nFinal test metrics:")

    for metric, value in test_metrics.items():
        print(
            f"{metric}: {value:.4f}"
        )

    # ----------------------------------------------
    # Save model
    # ----------------------------------------------

    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True
    )

    joblib.dump(
        production_model,
        MODEL_PATH
    )

    # ----------------------------------------------
    # Save evaluation report
    # ----------------------------------------------

    evaluation_results = {

        "baseline_model":
            "Logistic Regression",

        "candidate_model":
            "Random Forest",

        "baseline_validation_metrics":
            baseline_metrics,

        "candidate_validation_metrics":
            candidate_metrics,

        "production_model":
            production_model_name,

        "promotion_rule": {

            "minimum_candidate_auc":
                minimum_auc,

            "allowed_auc_drop_from_baseline":
                allowed_auc_drop
        },

        "candidate_promoted":
            bool(promote_candidate),

        "final_test_metrics":
            test_metrics
    }

    os.makedirs(
        os.path.dirname(EVAL_PATH),
        exist_ok=True
    )

    with open(
        EVAL_PATH,
        "w"
    ) as file:

        json.dump(
            evaluation_results,
            file,
            indent=4
        )

    print("\nTraining pipeline completed.")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Evaluation saved to: {EVAL_PATH}")


if __name__ == "__main__":
    train()
