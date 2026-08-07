
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

from features.preprocess import create_features


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = "models/churn_model.pkl"
MODEL_VERSION = "v1.0"


# --------------------------------------------------
# Load model
# --------------------------------------------------

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    model = None
    print(f"Warning: Could not load model: {e}")


# --------------------------------------------------
# Create FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-style ML inference API for customer churn prediction",
    version=MODEL_VERSION
)


# --------------------------------------------------
# Request schema
# --------------------------------------------------

class CustomerData(BaseModel):

    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health_check():

    if model is None:
        return {
            "status": "unhealthy",
            "model_version": MODEL_VERSION
        }

    return {
        "status": "healthy",
        "model_version": MODEL_VERSION
    }


# --------------------------------------------------
# Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(customer: CustomerData):

    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Model is not available"
        )

    try:

        # Convert request to DataFrame
        input_data = pd.DataFrame(
            [customer.model_dump()]
        )

        # Add a dummy target column because the shared
        # feature function only transforms predictors
        input_data["Churn"] = "No"

        # Create engineered features
        input_data = create_features(
            input_data
        )

        # Remove target
        input_data = input_data.drop(
            columns=["Churn"]
        )

        # Generate prediction
        prediction = model.predict(
            input_data
        )[0]

        probability = model.predict_proba(
            input_data
        )[0][1]

        churn_label = (
            "Churn"
            if prediction == 1
            else "No Churn"
        )

        return {
            "prediction": churn_label,
            "churn_probability": round(
                float(probability),
                4
            ),
            "model_version": MODEL_VERSION
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
