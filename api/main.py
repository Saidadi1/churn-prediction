"""FastAPI service for churn predictions.

Run:   uvicorn api.main:app --reload
Docs:  http://127.0.0.1:8000/docs
"""

import json
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.config import (
    MODEL_FILE,
    MODEL_METADATA_FILE,
    RAW_FEATURE_COLS,
    RISK_HIGH_FROM,
    RISK_MEDIUM_FROM,
    SHAP_BACKGROUND_FILE,
)
from src.data import clean_data
from src.explain import ChurnExplainer

YesNo = Literal["Yes", "No"]
InternetDependent = Literal["Yes", "No", "No internet service"]

EXAMPLE_CUSTOMER = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No", "tenure": 2,
    "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "No", "Contract": "Month-to-month",
    "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5, "TotalCharges": 171.0,
}


class Customer(BaseModel):
    """One customer, using the same fields as the raw dataset (minus ID and target)."""

    model_config = ConfigDict(json_schema_extra={"example": EXAMPLE_CUSTOMER}, extra="forbid")

    gender: Literal["Female", "Male"]
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetDependent
    OnlineBackup: InternetDependent
    DeviceProtection: InternetDependent
    TechSupport: InternetDependent
    StreamingTV: InternetDependent
    StreamingMovies: InternetDependent
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(gt=0, le=1000)
    TotalCharges: float = Field(ge=0, description="Total billed so far (0 for brand-new customers)")


class Reason(BaseModel):
    feature: str
    shap_value: float
    effect: str


class Prediction(BaseModel):
    churn_probability: float
    churn_prediction: int
    risk_level: Literal["low", "medium", "high"]
    top_reasons: list[Reason]


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=1000)


state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model ONCE at startup (not per request), so each call is fast."""
    if not MODEL_FILE.exists():
        raise RuntimeError(f"{MODEL_FILE} not found. Run `python -m src.train` and `python -m src.evaluate`.")
    state["pipeline"] = joblib.load(MODEL_FILE)
    state["metadata"] = json.loads(MODEL_METADATA_FILE.read_text())
    state["explainer"] = ChurnExplainer(state["pipeline"], joblib.load(SHAP_BACKGROUND_FILE))
    yield
    state.clear()


app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predicts the probability that a telecom customer will churn, with SHAP reasons.",
    version="1.0.0",
    lifespan=lifespan,
)


def risk_level(p: float) -> str:
    if p >= RISK_HIGH_FROM:
        return "high"
    return "medium" if p >= RISK_MEDIUM_FROM else "low"


def predict_frame(customers: list[Customer]) -> list[Prediction]:
    """Score customers with the SAME cleaning + pipeline used in training."""
    df = clean_data(pd.DataFrame([c.model_dump() for c in customers]))[RAW_FEATURE_COLS]
    proba = state["pipeline"].predict_proba(df)[:, 1]
    reasons = state["explainer"].top_reasons(df, k=3)
    threshold = state["metadata"]["threshold"]
    return [
        Prediction(
            churn_probability=round(float(p), 4),
            churn_prediction=int(p >= threshold),
            risk_level=risk_level(float(p)),
            top_reasons=r,
        )
        for p, r in zip(proba, reasons)
    ]


@app.get("/health")
def health() -> dict:
    """Liveness check + which model / threshold is being served."""
    if "pipeline" not in state:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok", "model": state["metadata"]["model_name"],
            "threshold": state["metadata"]["threshold"]}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer) -> Prediction:
    """Predict churn for one customer."""
    return predict_frame([customer])[0]


@app.post("/predict_batch", response_model=list[Prediction])
def predict_batch(request: BatchRequest) -> list[Prediction]:
    """Predict churn for up to 1,000 customers in one call (vectorised, much faster than a loop)."""
    return predict_frame(request.customers)
