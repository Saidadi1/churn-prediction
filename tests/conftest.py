"""Shared test fixtures."""

import copy

import pandas as pd
import pytest

from api.main import EXAMPLE_CUSTOMER
from src.config import MODEL_FILE


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Tiny raw-format dataset (strings, blank TotalCharges) so tests don't need the download."""
    a = {**EXAMPLE_CUSTOMER, "customerID": "0001-A", "TotalCharges": "171.0", "Churn": "Yes"}
    b = {**EXAMPLE_CUSTOMER, "customerID": "0002-B", "tenure": 0, "TotalCharges": " ", "Churn": "No"}
    c = {**EXAMPLE_CUSTOMER, "customerID": "0003-C", "tenure": 60, "Contract": "Two year",
         "InternetService": "No", "OnlineSecurity": "No internet service",
         "OnlineBackup": "No internet service", "DeviceProtection": "No internet service",
         "TechSupport": "No internet service", "StreamingTV": "No internet service",
         "StreamingMovies": "No internet service", "MonthlyCharges": 20.0,
         "TotalCharges": "1200.5", "Churn": "No"}
    return pd.DataFrame([a, b, c])


@pytest.fixture
def example_customer() -> dict:
    return copy.deepcopy(EXAMPLE_CUSTOMER)


requires_model = pytest.mark.skipif(
    not MODEL_FILE.exists(), reason="Trained model not found: run `python -m src.train` first"
)
