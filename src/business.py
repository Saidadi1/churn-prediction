"""Business simulation: is it worth acting on the model's predictions?

All money values are in the dataset's currency (USD). The economics are ASSUMPTIONS from
config.py, not measured facts, so the absolute numbers should be read as "what-if" estimates.

For every customer we target with a retention offer:
    cost   = OFFER_COST (paid whether or not they would have churned)
    gain   = OFFER_SUCCESS_RATE x customer_value   only if they really were going to churn
Net value is measured against doing nothing (the "target nobody" baseline = 0).
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    ASSUMPTION_CUSTOMER_VALUE_MONTHS,
    ASSUMPTION_OFFER_COST,
    ASSUMPTION_OFFER_SUCCESS_RATE,
    FIGURES_DIR,
    MODEL_FILE,
    MODEL_METADATA_FILE,
    REPORTS_DIR,
)


def customer_value(monthly_charges: pd.Series | np.ndarray) -> np.ndarray:
    """Value of keeping a customer = monthly bill x 12 (assumption)."""
    return np.asarray(monthly_charges, dtype=float) * ASSUMPTION_CUSTOMER_VALUE_MONTHS


def targeting_value(
    y_true: pd.Series | np.ndarray,
    targeted: np.ndarray,
    monthly_charges: pd.Series | np.ndarray,
    offer_cost: float = ASSUMPTION_OFFER_COST,
    success_rate: float = ASSUMPTION_OFFER_SUCCESS_RATE,
) -> dict[str, float]:
    """Expected costs, savings and net value of sending offers to the `targeted` customers."""
    y = np.asarray(y_true).astype(bool)
    targeted = np.asarray(targeted).astype(bool)
    value = customer_value(monthly_charges)

    n_targeted = int(targeted.sum())
    churners_reached = int((targeted & y).sum())
    cost = n_targeted * offer_cost
    saved = float((value * success_rate)[targeted & y].sum())
    return {
        "customers_targeted": n_targeted,
        "churners_reached": churners_reached,
        "churners_reached_%": 100 * churners_reached / max(int(y.sum()), 1),
        "campaign_cost": cost,
        "expected_saved_value": saved,
        "expected_net_value": saved - cost,
    }


def top_k_mask(proba: np.ndarray, fraction: float) -> np.ndarray:
    """Boolean mask selecting the `fraction` of customers with the highest churn probability."""
    n = int(np.ceil(fraction * len(proba)))
    mask = np.zeros(len(proba), dtype=bool)
    mask[np.argsort(-proba)[:n]] = True
    return mask


def simulate_strategies(
    y_true: pd.Series, proba: np.ndarray, monthly_charges: pd.Series, threshold: float
) -> pd.DataFrame:
    """Compare targeting strategies on the same customers."""
    y = np.asarray(y_true)
    strategies = {
        "Target nobody": np.zeros(len(y), dtype=bool),
        "Target everyone": np.ones(len(y), dtype=bool),
        "Model: top 10% riskiest": top_k_mask(proba, 0.10),
        "Model: top 20% riskiest": top_k_mask(proba, 0.20),
        "Model: top 30% riskiest": top_k_mask(proba, 0.30),
        f"Model: threshold {threshold:.2f}": proba >= threshold,
        # Not achievable: shows the ceiling if we knew exactly who will churn.
        "Perfect knowledge (upper bound)": y.astype(bool),
    }
    rows = [{"strategy": name, **targeting_value(y, mask, monthly_charges)}
            for name, mask in strategies.items()]
    return pd.DataFrame(rows)


def plot_strategies(table: pd.DataFrame, path=FIGURES_DIR / "business_net_value.png") -> None:
    """Horizontal bar chart of expected net value per strategy."""
    data = table[table["strategy"] != "Perfect knowledge (upper bound)"].iloc[::-1]
    colors = ["#C44E52" if v < 0 else "#55A868" for v in data["expected_net_value"]]
    best = data.loc[data["expected_net_value"].idxmax(), "strategy"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(data["strategy"], data["expected_net_value"], color=colors)
    for bar, v, name in zip(bars, data["expected_net_value"], data["strategy"]):
        label = f"${v:,.0f}" + ("  <- best" if name == best else "")
        ax.text(v, bar.get_y() + bar.get_height() / 2, f"  {label}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9)
    ax.axvline(0, color="grey", lw=1)
    ax.set_xlabel("Expected net value vs doing nothing ($, test set)")
    ax.set_title(
        f"Retention campaign value (assumptions: offer ${ASSUMPTION_OFFER_COST:.0f}, "
        f"{ASSUMPTION_OFFER_SUCCESS_RATE:.0%} success, value = {ASSUMPTION_CUSTOMER_VALUE_MONTHS} x monthly bill)",
        fontsize=10,
    )
    xmin, xmax = ax.get_xlim()
    ax.set_xlim(xmin * 1.35 if xmin < 0 else xmin, xmax * 1.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.switch_backend("Agg")  # scripts only save figures; no GUI window needed
    """Run the simulation on the held-out test set with the saved model and threshold."""
    import joblib

    from src.data import load_split

    pipeline = joblib.load(MODEL_FILE)
    threshold = json.loads(MODEL_METADATA_FILE.read_text())["threshold"]
    X_test, y_test = load_split("test")
    proba = pipeline.predict_proba(X_test)[:, 1]

    table = simulate_strategies(y_test, proba, X_test["MonthlyCharges"], threshold)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORTS_DIR / "business_simulation.csv", index=False)
    plot_strategies(table)

    pd.set_option("display.width", 200)
    print(table.round(1).to_string(index=False))
    realistic = table[table["strategy"] != "Perfect knowledge (upper bound)"]
    best = realistic.loc[realistic["expected_net_value"].idxmax()]
    print(f"\nRecommended strategy: {best['strategy']} "
          f"(expected net value ${best['expected_net_value']:,.0f} on {len(y_test):,} test customers)")


if __name__ == "__main__":
    main()
