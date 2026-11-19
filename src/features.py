"""Feature engineering and the preprocessing pipeline.

Everything here is wrapped into one scikit-learn Pipeline, so the exact same steps run
during training, evaluation and inside the API (no "training/serving skew").
"""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import RAW_CATEGORICAL_COLS, RAW_NUMERIC_COLS, SERVICE_COLS

ENGINEERED_NUMERIC_COLS: list[str] = ["avg_monthly_spend", "num_services", "has_long_contract"]
ENGINEERED_CATEGORICAL_COLS: list[str] = ["tenure_bucket"]

NUMERIC_COLS: list[str] = RAW_NUMERIC_COLS + ENGINEERED_NUMERIC_COLS
CATEGORICAL_COLS: list[str] = RAW_CATEGORICAL_COLS + ENGINEERED_CATEGORICAL_COLS


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add business-driven features. Row-wise only (nothing is learned), so it cannot leak."""
    df = df.copy()

    # Lifecycle stage: EDA showed churn is ~47% in year 1 and ~10% after year 4, and the
    # effect is not linear. Buckets let even a linear model capture that shape.
    df["tenure_bucket"] = pd.cut(
        df["tenure"], bins=[-1, 12, 24, 48, np.inf], labels=["0-1y", "1-2y", "2-4y", "4y+"]
    ).astype(str)

    # Average historical bill. Compared with the current MonthlyCharges it reveals price
    # changes. New customers (tenure 0) have no history, so their current bill is the best estimate.
    safe_tenure = df["tenure"].replace(0, np.nan)
    df["avg_monthly_spend"] = (df["TotalCharges"] / safe_tenure).fillna(df["MonthlyCharges"])

    # Engagement / switching cost: the more services a customer uses, the harder it is to leave.
    n_addons = (df[SERVICE_COLS] == "Yes").sum(axis=1)
    df["num_services"] = n_addons + (df["InternetService"] != "No").astype(int)

    # Commitment: one- and two-year contracts churn at 11% and 3% vs 43% for month-to-month.
    df["has_long_contract"] = df["Contract"].isin(["One year", "Two year"]).astype(int)

    return df


def build_preprocessor() -> Pipeline:
    """Feature engineering + scaling for numeric + one-hot for categorical columns."""
    column_transformer = ColumnTransformer(
        transformers=[
            # Scaling matters for Logistic Regression (regularisation + convergence);
            # tree models ignore it, so one shared pipeline works for both.
            ("num", StandardScaler(), NUMERIC_COLS),
            # handle_unknown="ignore": an unseen category in production becomes all zeros
            # instead of crashing the API.
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS),
        ],
        verbose_feature_names_out=False,  # keep readable names like "Contract_Two year"
    )
    return Pipeline(
        steps=[
            ("features", FunctionTransformer(add_features)),
            ("columns", column_transformer),
        ]
    )


def get_feature_names(preprocessor: Pipeline) -> list[str]:
    """Names of the columns produced by a FITTED preprocessor (used by SHAP)."""
    return list(preprocessor.named_steps["columns"].get_feature_names_out())
