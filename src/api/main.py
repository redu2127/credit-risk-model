from fastapi import FastAPI
import joblib
import pandas as pd
from src.api.models import CustomerData

app = FastAPI(title="Credit Risk API")

model = joblib.load("models/best_model.pkl")

FEATURE_COLUMNS = [
    "total_transaction_amount",
    "average_transaction_amount",
    "transaction_count",
    "std_transaction_amount",
    "total_value",
    "average_value",
    "average_transaction_hour",
    "average_transaction_day",
    "average_transaction_month",
    "fraud_count",
    "fraud_rate",
    "Recency",
    "Frequency",
    "Monetary",
]


@app.get("/")
def home():
    return {"message": "Credit Risk API is running"}


@app.post("/predict")
def predict(data: CustomerData):
    input_df = pd.DataFrame([data.model_dump()])

    for col in FEATURE_COLUMNS:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[FEATURE_COLUMNS]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "risk_probability": float(probability),
        "risk_class": "High Risk" if prediction == 1 else "Low Risk",
    }
