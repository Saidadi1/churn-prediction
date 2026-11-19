import pandas as pd

from src.config import ID_COL, TARGET_COL
from src.data import clean_data, split_data


def test_total_charges_blank_becomes_zero(raw_df):
    clean = clean_data(raw_df)
    assert pd.api.types.is_float_dtype(clean["TotalCharges"])
    # The tenure = 0 customer had a blank TotalCharges: never billed -> 0.
    assert clean.loc[clean["tenure"] == 0, "TotalCharges"].tolist() == [0.0]


def test_no_missing_values_after_cleaning(raw_df):
    assert clean_data(raw_df).isna().sum().sum() == 0


def test_target_is_binary_and_id_dropped(raw_df):
    clean = clean_data(raw_df)
    assert set(clean[TARGET_COL].unique()) <= {0, 1}
    assert ID_COL not in clean.columns


def test_clean_does_not_modify_input(raw_df):
    before = raw_df.copy()
    clean_data(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)


def test_split_is_stratified_60_20_20():
    n = 1000
    df = pd.DataFrame({"x": range(n), TARGET_COL: [1] * 250 + [0] * 750})
    X_tr, X_va, X_te, y_tr, y_va, y_te = split_data(df)
    assert (len(X_tr), len(X_va), len(X_te)) == (600, 200, 200)
    for y in (y_tr, y_va, y_te):
        assert abs(y.mean() - 0.25) < 0.01
    # No customer appears in two splits (no leakage).
    assert not set(X_tr.index) & set(X_te.index)
    assert not set(X_tr.index) & set(X_va.index)
