"""Central configuration: paths, seed and constants.

Keeping every "magic value" here means one place to change it, and every
module (training, API, tests) agrees on the same values.
"""

from pathlib import Path

# --- Paths ---
# Resolve from this file so scripts work no matter which folder you run them from.
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

RAW_DATA_FILE: Path = RAW_DATA_DIR / "Telco-Customer-Churn.csv"
MODEL_FILE: Path = MODELS_DIR / "churn_pipeline.joblib"
# Small sample of training rows used as the SHAP background (baseline) in the API.
SHAP_BACKGROUND_FILE: Path = MODELS_DIR / "shap_background.joblib"
# Chosen threshold + model info, written by evaluate.py and read by the API.
MODEL_METADATA_FILE: Path = MODELS_DIR / "model_metadata.json"

# --- Data source ---
DATA_URL: str = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)

# --- Reproducibility ---
RANDOM_STATE: int = 42

# --- Columns ---
TARGET_COL: str = "Churn"
ID_COL: str = "customerID"

# Raw input columns the model expects (everything except ID and target).
RAW_NUMERIC_COLS: list[str] = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
RAW_CATEGORICAL_COLS: list[str] = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
]
RAW_FEATURE_COLS: list[str] = RAW_NUMERIC_COLS + RAW_CATEGORICAL_COLS

# Add-on services counted in the engineered "num_services" feature.
SERVICE_COLS: list[str] = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]

# --- Data split (60 / 20 / 20, stratified on the target) ---
TEST_SIZE: float = 0.20
VALID_SIZE: float = 0.20  # fraction of the FULL dataset, not of what remains

# --- MLflow ---
MLFLOW_EXPERIMENT_NAME: str = "churn-prediction"
# A local SQLite store is the standard lightweight setup (no server needed).
MLFLOW_TRACKING_URI: str = f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"

# --- Hyperparameter search (kept small so training stays well under 5 minutes) ---
CV_FOLDS: int = 5
N_SEARCH_ITER: int = 30

# --- Risk levels returned by the API (probability cut-offs) ---
RISK_MEDIUM_FROM: float = 0.30
RISK_HIGH_FROM: float = 0.60

# --- Business assumptions (ASSUMPTIONS, not facts — used in Phase 7) ---
# Value of keeping a customer = one year of their monthly bill.
ASSUMPTION_CUSTOMER_VALUE_MONTHS: int = 12
# Cost of giving one customer a retention offer (discount, call, etc.).
ASSUMPTION_OFFER_COST: float = 50.0
# Share of true churners who stay because of the offer.
ASSUMPTION_OFFER_SUCCESS_RATE: float = 0.30
