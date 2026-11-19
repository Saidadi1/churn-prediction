"""Data access: download, load, clean and split the Telco churn dataset."""

import sys
import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_URL,
    ID_COL,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    RAW_DATA_FILE,
    TARGET_COL,
    TEST_SIZE,
    VALID_SIZE,
)

KAGGLE_HINT = (
    "Download failed. Please download 'Telco Customer Churn' from Kaggle "
    "(https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and save "
    f"the CSV as: {RAW_DATA_FILE}"
)


def download_data(url: str = DATA_URL, dest: Path = RAW_DATA_FILE, force: bool = False) -> Path:
    """Download the raw CSV to `dest`. Skips the download if the file already exists."""
    if dest.exists() and not force:
        print(f"Data already present: {dest}")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    # Download to a temp file first so a failed download never leaves a broken CSV behind.
    tmp = dest.with_suffix(".part")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(dest)
    except Exception as exc:  # network errors, 404, etc.
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{KAGGLE_HINT}\nOriginal error: {exc}") from exc

    print(f"Downloaded data to: {dest}")
    return dest


def load_raw_data(path: Path = RAW_DATA_FILE) -> pd.DataFrame:
    """Load the raw CSV exactly as it is on disk (no cleaning)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.data` first.\n{KAGGLE_HINT}")
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix types, encode the target as 0/1 and drop the ID column.

    Works on both training data (with Churn) and scoring data (without Churn).
    """
    df = df.copy()

    # TotalCharges is text with blank strings for tenure = 0 customers who were never billed,
    # so 0 is the true value (not a missing value to impute).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.loc[df["TotalCharges"].isna() & (df["tenure"] == 0), "TotalCharges"] = 0.0
    # Any other unparseable value would be a real data problem: fall back to tenure x monthly.
    df["TotalCharges"] = df["TotalCharges"].fillna(df["tenure"] * df["MonthlyCharges"])

    if TARGET_COL in df.columns and df[TARGET_COL].dtype == object:
        df[TARGET_COL] = df[TARGET_COL].map({"Yes": 1, "No": 0}).astype(int)

    # The ID is unique per row: it carries no signal and the model would only memorise it.
    return df.drop(columns=[ID_COL], errors="ignore")


def split_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Stratified 60/20/20 train/validation/test split.

    Stratifying keeps the ~26.5% churn rate identical in every split, so scores are comparable.
    """
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    # VALID_SIZE is a share of the full data, so rescale it to the remaining 80%.
    valid_share = VALID_SIZE / (1 - TEST_SIZE)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_temp, y_temp, test_size=valid_share, stratify=y_temp, random_state=RANDOM_STATE
    )
    return X_train, X_valid, X_test, y_train, y_valid, y_test


def save_splits(splits: dict[str, pd.DataFrame], out_dir: Path = PROCESSED_DATA_DIR) -> None:
    """Save each split (features + target) as CSV so evaluation uses exactly the same rows."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in splits.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False)


def load_split(name: str, in_dir: Path = PROCESSED_DATA_DIR) -> tuple[pd.DataFrame, pd.Series]:
    """Load a saved split ('train', 'valid' or 'test') as (X, y)."""
    path = in_dir / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.train` first.")
    frame = pd.read_csv(path)
    return frame.drop(columns=[TARGET_COL]), frame[TARGET_COL]


if __name__ == "__main__":
    try:
        download_data()
    except RuntimeError as err:
        print(err, file=sys.stderr)
        sys.exit(1)

    df = load_raw_data()
    print(f"Loaded {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(df.head())
