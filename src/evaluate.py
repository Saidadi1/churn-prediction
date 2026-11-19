"""Evaluation: metrics, threshold choice (on validation) and final test-set report.

Rule: the threshold is chosen on the VALIDATION set. The TEST set is scored exactly once,
at the end, to get an honest estimate of real-world performance.
"""

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.business import targeting_value
from src.config import FIGURES_DIR, MODEL_FILE, MODEL_METADATA_FILE, REPORTS_DIR
from src.data import load_split


def compute_metrics(y_true: pd.Series, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Ranking metrics (threshold-free) + classification metrics at `threshold`."""
    pred = (proba >= threshold).astype(int)
    return {
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "accuracy": accuracy_score(y_true, pred),
    }


def threshold_table(y_true: pd.Series, proba: np.ndarray, monthly_charges: pd.Series) -> pd.DataFrame:
    """Precision, recall, F1 and business net value for thresholds 0.05 ... 0.95."""
    rows = []
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        pred = proba >= t
        rows.append({
            "threshold": t,
            "precision": precision_score(y_true, pred, zero_division=0),
            "recall": recall_score(y_true, pred, zero_division=0),
            "f1": f1_score(y_true, pred, zero_division=0),
            "share_targeted": pred.mean(),
            "net_value": targeting_value(y_true, pred, monthly_charges)["expected_net_value"],
        })
    return pd.DataFrame(rows)


def plot_threshold_analysis(table: pd.DataFrame, chosen: float, path=FIGURES_DIR / "threshold_analysis.png") -> None:
    """Precision/recall/F1 vs threshold, with expected net value on a second axis."""
    fig, ax = plt.subplots(figsize=(9, 5))
    for col in ["precision", "recall", "f1"]:
        ax.plot(table["threshold"], table[col], label=col.capitalize(), lw=2)
    ax.set_xlabel("Decision threshold (predict churn if probability >= threshold)")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.02)
    ax2 = ax.twinx()
    ax2.plot(table["threshold"], table["net_value"], color="black", ls="--", label="Net value ($)")
    ax2.set_ylabel("Expected net value on validation set ($)")
    ax.axvline(chosen, color="grey", ls=":", lw=2)
    ax.text(chosen + 0.01, 0.05, f"chosen = {chosen:.2f}", color="dimgrey")
    lines = ax.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
    ax.legend(lines, labels, loc="center right")
    ax.set_title("Threshold trade-off (validation set)")
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_test_figures(y_test: pd.Series, proba: np.ndarray, threshold: float, model_name: str) -> None:
    """Confusion matrix, ROC, PR and calibration plots for the test set."""
    cm = confusion_matrix(y_test, (proba >= threshold).astype(int))
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ConfusionMatrixDisplay(cm, display_labels=["Stay", "Churn"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion matrix (test, threshold {threshold:.2f})")
    fig.savefig(FIGURES_DIR / "test_confusion_matrix.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    RocCurveDisplay.from_predictions(y_test, proba, name=model_name, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.5)")
    ax.legend(loc="lower right")
    ax.set_title("ROC curve (test)")
    fig.savefig(FIGURES_DIR / "test_roc_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5))
    PrecisionRecallDisplay.from_predictions(y_test, proba, name=model_name, ax=ax)
    ax.axhline(y_test.mean(), color="k", ls="--", label=f"Random (= churn rate {y_test.mean():.2f})")
    ax.legend(loc="upper right")
    ax.set_title("Precision-Recall curve (test)")
    fig.savefig(FIGURES_DIR / "test_pr_curve.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", label=model_name)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed churn rate")
    ax.set_title("Calibration plot (test, 10 quantile bins)")
    ax.legend()
    fig.savefig(FIGURES_DIR / "test_calibration.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.switch_backend("Agg")  # scripts only save figures; no GUI window needed
    pipeline = joblib.load(MODEL_FILE)
    metadata = json.loads(MODEL_METADATA_FILE.read_text())
    model_name = metadata["model_name"]

    # 1) Choose the threshold on VALIDATION data, maximising expected business value.
    X_valid, y_valid = load_split("valid")
    proba_valid = pipeline.predict_proba(X_valid)[:, 1]
    table = threshold_table(y_valid, proba_valid, X_valid["MonthlyCharges"])
    best_row = table.loc[table["net_value"].idxmax()]
    threshold = float(best_row["threshold"])
    table.to_csv(REPORTS_DIR / "threshold_analysis.csv", index=False)
    plot_threshold_analysis(table, threshold)
    f1_best = table.loc[table["f1"].idxmax()]
    print(f"Chosen threshold (max net value on validation): {threshold:.2f}  "
          f"-> precision {best_row.precision:.3f}, recall {best_row.recall:.3f}, "
          f"targets {best_row.share_targeted:.1%} of customers")
    print(f"(For reference, the F1-optimal threshold would be {f1_best.threshold:.2f})")

    # 2) Score the TEST set once.
    X_test, y_test = load_split("test")
    proba_test = pipeline.predict_proba(X_test)[:, 1]
    results = {
        "model_name": model_name,
        "n_test": int(len(y_test)),
        "chosen_threshold": threshold,
        "at_chosen_threshold": compute_metrics(y_test, proba_test, threshold),
        "at_default_0.5": compute_metrics(y_test, proba_test, 0.5),
        "brier_score": brier_score_loss(y_test, proba_test),
        "confusion_matrix_at_chosen": confusion_matrix(y_test, (proba_test >= threshold).astype(int)).tolist(),
    }
    (REPORTS_DIR / "test_metrics.json").write_text(json.dumps(results, indent=2))
    plot_test_figures(y_test, proba_test, threshold, model_name)

    metadata["threshold"] = threshold
    metadata["test_metrics"] = results["at_chosen_threshold"]
    MODEL_METADATA_FILE.write_text(json.dumps(metadata, indent=2))

    print(f"\nTEST set results ({model_name}, n={len(y_test)}):")
    print(pd.DataFrame({"threshold " + f"{threshold:.2f}": results["at_chosen_threshold"],
                        "threshold 0.50": results["at_default_0.5"]}).round(3).to_string())
    print(f"Brier score: {results['brier_score']:.3f}")
    tn, fp, fn, tp = np.array(results["confusion_matrix_at_chosen"]).ravel()
    print(f"Confusion matrix @ {threshold:.2f}: TN={tn} FP={fp} FN={fn} TP={tp}")


if __name__ == "__main__":
    main()
