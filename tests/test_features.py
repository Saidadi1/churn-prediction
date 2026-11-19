import numpy as np

from src.config import TARGET_COL
from src.data import clean_data
from src.features import CATEGORICAL_COLS, NUMERIC_COLS, add_features, build_preprocessor


def test_add_features_creates_expected_columns(raw_df):
    out = add_features(clean_data(raw_df))
    for col in ["tenure_bucket", "avg_monthly_spend", "num_services", "has_long_contract"]:
        assert col in out.columns
    assert out[["avg_monthly_spend", "num_services"]].isna().sum().sum() == 0


def test_avg_monthly_spend_handles_tenure_zero(raw_df):
    out = add_features(clean_data(raw_df))
    new = out[out["tenure"] == 0].iloc[0]
    # No billing history yet: fall back to the current monthly bill (no division by zero).
    assert new["avg_monthly_spend"] == new["MonthlyCharges"]
    assert np.isfinite(out["avg_monthly_spend"]).all()


def test_engineered_values(raw_df):
    out = add_features(clean_data(raw_df)).set_index(raw_df["customerID"])
    assert out.loc["0003-C", "has_long_contract"] == 1
    assert out.loc["0001-A", "has_long_contract"] == 0
    assert out.loc["0003-C", "tenure_bucket"] == "4y+"
    # 0003-C: phone only, no internet -> 1 service. 0001-A: phone + fiber + StreamingTV -> 3.
    assert out.loc["0003-C", "num_services"] == 1
    assert out.loc["0001-A", "num_services"] == 3


def test_preprocessor_output_shape(raw_df):
    X = clean_data(raw_df).drop(columns=[TARGET_COL])
    pre = build_preprocessor().fit(X)
    Xt = pre.transform(X)
    one_hot_width = sum(len(c) for c in pre.named_steps["columns"].named_transformers_["cat"].categories_)
    assert Xt.shape == (len(X), len(NUMERIC_COLS) + one_hot_width)
    assert not np.isnan(Xt).any()
    assert len(CATEGORICAL_COLS) == 16


def test_preprocessor_ignores_unseen_category(raw_df):
    X = clean_data(raw_df).drop(columns=[TARGET_COL])
    pre = build_preprocessor().fit(X)
    unseen = X.head(1).assign(PaymentMethod="Crypto")  # never seen in training
    assert pre.transform(unseen).shape[0] == 1
