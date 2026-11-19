"""Train baseline + main models, track them in MLflow, save the best full pipeline.

Run:  python -m src.train
"""

import json
import os
import time

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")  # silence an MLflow info banner
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")

import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import skops.io as sio
from scipy.stats import loguniform, randint, uniform
from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import (
    CV_FOLDS,
    FIGURES_DIR,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_FILE,
    MODEL_METADATA_FILE,
    MODELS_DIR,
    N_SEARCH_ITER,
    PROJECT_ROOT,
    RANDOM_STATE,
    REPORTS_DIR,
    SHAP_BACKGROUND_FILE,
    TARGET_COL,
)
from src.data import clean_data, download_data, load_raw_data, save_splits, split_data
from src.evaluate import compute_metrics
from src.features import build_preprocessor


def make_pipeline(model: ClassifierMixin) -> Pipeline:
    """Preprocessing + model in ONE object: fitted together, saved together, served together."""
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def setup_mlflow() -> None:
    """Use a local SQLite tracking store and keep artifacts inside the project folder."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME) is None:
        mlflow.create_experiment(
            MLFLOW_EXPERIMENT_NAME, artifact_location=(PROJECT_ROOT / "mlruns").as_uri()
        )
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


def curves_figure(y_true: pd.Series, proba: np.ndarray, name: str) -> plt.Figure:
    """ROC + PR curves side by side (validation set), logged to MLflow."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay.from_predictions(y_true, proba, name=name, ax=ax1)
    ax1.plot([0, 1], [0, 1], "k--")
    ax1.set_title("ROC (validation)")
    PrecisionRecallDisplay.from_predictions(y_true, proba, name=name, ax=ax2)
    ax2.axhline(y_true.mean(), color="k", ls="--")
    ax2.set_title("Precision-Recall (validation)")
    fig.tight_layout()
    return fig


def log_run(
    name: str,
    pipeline: Pipeline,
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cv_auc: float,
) -> dict[str, float]:
    """Evaluate a FITTED pipeline on validation and log everything to MLflow."""
    proba = pipeline.predict_proba(X_valid)[:, 1]
    metrics = compute_metrics(y_valid, proba)
    with mlflow.start_run(run_name=name):
        mlflow.log_params(params)
        mlflow.log_metric("cv_roc_auc", cv_auc)
        mlflow.log_metrics({f"valid_{k}": v for k, v in metrics.items()})
        fig = curves_figure(y_valid, proba, name)
        mlflow.log_figure(fig, "validation_curves.png")
        plt.close(fig)
        # MLflow 3 stores models with skops (safer than pickle) and refuses unknown types.
        # We built this pipeline ourselves, so we explicitly trust its custom types.
        trusted = sio.get_untrusted_types(data=sio.dumps(pipeline))
        mlflow.sklearn.log_model(pipeline, name="model", input_example=X_train.head(3),
                                 skops_trusted_types=trusted)
    return metrics


def main() -> None:
    plt.switch_backend("Agg")  # scripts only save figures; no GUI window needed
    start = time.time()
    np.random.seed(RANDOM_STATE)
    for d in (MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # --- Data ---
    download_data()
    df = clean_data(load_raw_data())
    X_train, X_valid, X_test, y_train, y_valid, y_test = split_data(df)
    save_splits({
        "train": X_train.assign(**{TARGET_COL: y_train}),
        "valid": X_valid.assign(**{TARGET_COL: y_valid}),
        "test": X_test.assign(**{TARGET_COL: y_test}),
    })
    print(f"Split sizes  train={len(X_train)}  valid={len(X_valid)}  test={len(X_test)}  "
          f"| churn rate {y_train.mean():.3f} / {y_valid.mean():.3f} / {y_test.mean():.3f}")

    setup_mlflow()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results: dict[str, dict[str, float]] = {}
    fitted: dict[str, Pipeline] = {}

    # --- 1) Dummy baseline: always predicts the majority class (the "do nothing" model) ---
    dummy = make_pipeline(DummyClassifier(strategy="prior"))
    cv_auc = cross_val_score(dummy, X_train, y_train, cv=cv, scoring="roc_auc").mean()
    dummy.fit(X_train, y_train)
    results["Dummy (prior)"] = {"cv_roc_auc": cv_auc, **log_run(
        "dummy", dummy, {"strategy": "prior"}, X_train, y_train, X_valid, y_valid, cv_auc)}
    fitted["Dummy (prior)"] = dummy

    # --- 2) Logistic Regression: simple, interpretable baseline ---
    # class_weight="balanced" up-weights the minority class (churners) so it is not ignored.
    logreg = make_pipeline(LogisticRegression(class_weight="balanced", max_iter=2000,
                                              random_state=RANDOM_STATE))
    cv_auc = cross_val_score(logreg, X_train, y_train, cv=cv, scoring="roc_auc").mean()
    logreg.fit(X_train, y_train)
    results["Logistic Regression"] = {"cv_roc_auc": cv_auc, **log_run(
        "logistic_regression", logreg, {"class_weight": "balanced", "C": 1.0},
        X_train, y_train, X_valid, y_valid, cv_auc)}
    fitted["Logistic Regression"] = logreg

    # --- 3) XGBoost + randomized search ---
    # scale_pos_weight = (#negatives / #positives) is XGBoost's built-in way to balance classes.
    spw = float((y_train == 0).sum() / (y_train == 1).sum())
    xgb = make_pipeline(XGBClassifier(
        objective="binary:logistic", eval_metric="logloss", tree_method="hist",
        scale_pos_weight=spw, random_state=RANDOM_STATE, n_jobs=1,
    ))
    param_distributions = {
        "model__n_estimators": randint(100, 600),
        "model__max_depth": randint(2, 7),
        "model__learning_rate": loguniform(0.01, 0.2),
        "model__subsample": uniform(0.6, 0.4),
        "model__colsample_bytree": uniform(0.5, 0.5),
        "model__min_child_weight": randint(1, 10),
        "model__reg_lambda": loguniform(0.1, 10),
    }
    search = RandomizedSearchCV(
        xgb, param_distributions, n_iter=N_SEARCH_ITER, scoring="roc_auc", cv=cv,
        random_state=RANDOM_STATE, n_jobs=-1, refit=True,
    )
    t0 = time.time()
    search.fit(X_train, y_train)
    print(f"XGBoost search: {N_SEARCH_ITER} candidates x {CV_FOLDS} folds in {time.time() - t0:.0f}s "
          f"| best CV ROC-AUC {search.best_score_:.4f}")
    best_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    best_params["scale_pos_weight"] = round(spw, 4)
    results["XGBoost (tuned)"] = {"cv_roc_auc": search.best_score_, **log_run(
        "xgboost_tuned", search.best_estimator_, best_params,
        X_train, y_train, X_valid, y_valid, search.best_score_)}
    fitted["XGBoost (tuned)"] = search.best_estimator_

    # --- Compare (validation set, threshold 0.5) ---
    table = pd.DataFrame(results).T[["cv_roc_auc", "roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy"]]
    table.index.name = "model"
    table.to_csv(REPORTS_DIR / "model_comparison.csv")
    print("\nValidation results (threshold 0.5):")
    print(table.round(4).to_string())

    # Pick the winner on validation ROC-AUC (the metric we tuned for); the test set stays untouched.
    best_name = table.drop(index="Dummy (prior)")["roc_auc"].idxmax()
    best_pipeline = fitted[best_name]
    joblib.dump(best_pipeline, MODEL_FILE)
    # A small, fixed sample of TRAINING rows: the "average customer" baseline for SHAP.
    # 100 rows = SHAP's default background size: stable estimates, fast enough for the API.
    joblib.dump(X_train.sample(100, random_state=RANDOM_STATE), SHAP_BACKGROUND_FILE)
    metadata = {
        "model_name": best_name,
        "threshold": 0.5,  # replaced by the business-optimal threshold in `python -m src.evaluate`
        "valid_metrics": {k: float(v) for k, v in results[best_name].items()},
        "params": {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in best_params.items()}
        if best_name.startswith("XGBoost") else {"class_weight": "balanced"},
    }
    MODEL_METADATA_FILE.write_text(json.dumps(metadata, indent=2, default=str))
    print(f"\nBest model: {best_name} -> saved to {MODEL_FILE.relative_to(PROJECT_ROOT)}")
    print(f"Total training time: {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
