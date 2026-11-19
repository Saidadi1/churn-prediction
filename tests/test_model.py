import joblib
import numpy as np
import pandas as pd

from src.config import MODEL_FILE, RAW_FEATURE_COLS
from tests.conftest import requires_model


@requires_model
def test_model_loads_and_predicts_probabilities(example_customer):
    pipeline = joblib.load(MODEL_FILE)
    X = pd.DataFrame([example_customer, {**example_customer, "tenure": 70, "Contract": "Two year",
                                         "TotalCharges": 5985.0}])[RAW_FEATURE_COLS]
    proba = pipeline.predict_proba(X)
    assert proba.shape == (2, 2)
    assert np.all((proba >= 0) & (proba <= 1))
    assert np.allclose(proba.sum(axis=1), 1)
    # Sanity check on business logic: a new month-to-month customer is riskier than a loyal one.
    assert proba[0, 1] > proba[1, 1]
