"""Explainability with SHAP: global drivers, dependence plots and one individual customer.

SHAP splits each prediction into additive contributions, one per feature:
    model output (log-odds) = baseline (average customer) + sum of feature contributions
Positive SHAP value -> pushes the customer TOWARDS churn; negative -> away from churn.
"""

import json

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.pipeline import Pipeline

from src.config import FIGURES_DIR, MODEL_FILE, REPORTS_DIR, SHAP_BACKGROUND_FILE
from src.features import add_features, get_feature_names


class ChurnExplainer:
    """Wraps a fitted churn pipeline and produces SHAP explanations on readable values."""

    def __init__(self, pipeline: Pipeline, background: pd.DataFrame) -> None:
        self.preprocess = pipeline.named_steps["preprocess"]
        self.model = pipeline.named_steps["model"]
        self.feature_names = get_feature_names(self.preprocess)
        # shap.Explainer picks the right algorithm: exact TreeExplainer for XGBoost,
        # LinearExplainer for Logistic Regression. Background = "average customer" baseline.
        self.explainer = shap.Explainer(
            self.model, self.preprocess.transform(background), feature_names=self.feature_names
        )
        self._column_map = self._build_column_map()

    def _build_column_map(self) -> list[tuple[str, str | None]]:
        """For each model column: (original column, category) — category is None for numerics."""
        ct = self.preprocess.named_steps["columns"]
        mapping: list[tuple[str, str | None]] = [(c, None) for c in ct.transformers_[0][2]]
        encoder = ct.named_transformers_["cat"]
        for col, cats in zip(ct.transformers_[1][2], encoder.categories_):
            mapping += [(col, str(cat)) for cat in cats]
        return mapping

    def _display_values(self, X_raw: pd.DataFrame) -> np.ndarray:
        """Human-readable feature values (unscaled numbers, 0/1 for categories) for plots."""
        enriched = add_features(X_raw)
        cols = []
        for col, cat in self._column_map:
            cols.append(enriched[col].to_numpy(dtype=float) if cat is None
                        else (enriched[col].astype(str) == cat).to_numpy(dtype=float))
        return np.column_stack(cols)

    def explain(self, X_raw: pd.DataFrame) -> shap.Explanation:
        """SHAP values (log-odds) for raw customer rows, with readable `data` for plotting."""
        sv = self.explainer(self.preprocess.transform(X_raw))
        return shap.Explanation(
            values=sv.values, base_values=sv.base_values,
            data=self._display_values(X_raw), feature_names=self.feature_names,
        )

    def top_reasons(self, X_raw: pd.DataFrame, k: int = 3) -> list[list[dict]]:
        """Top-k features (by absolute SHAP value) for every row, phrased for humans."""
        sv = self.explain(X_raw)
        enriched = add_features(X_raw).reset_index(drop=True)
        out = []
        for i in range(len(X_raw)):
            order = np.argsort(-np.abs(sv.values[i]))[:k]
            reasons = []
            for j in order:
                col, cat = self._column_map[j]
                raw = enriched.loc[i, col]
                if cat is None:
                    text = f"{col} = {round(float(raw), 2):g}"
                elif str(raw) == cat:
                    text = f"{col} = {cat}"
                else:
                    text = f"{col} is not {cat} (it is {raw})"
                impact = float(sv.values[i, j])
                reasons.append({
                    "feature": text,
                    "shap_value": round(impact, 4),
                    "effect": "increases churn risk" if impact > 0 else "decreases churn risk",
                })
            out.append(reasons)
        return out


def main() -> None:
    from src.data import load_split

    plt.switch_backend("Agg")
    pipeline = joblib.load(MODEL_FILE)
    explainer = ChurnExplainer(pipeline, joblib.load(SHAP_BACKGROUND_FILE))
    X_test, y_test = load_split("test")
    sv = explainer.explain(X_test)

    # 1) Global importance: mean |SHAP| over all test customers.
    importance = (pd.DataFrame({"feature": sv.feature_names, "mean_abs_shap": np.abs(sv.values).mean(0)})
                  .sort_values("mean_abs_shap", ascending=False))
    importance.to_csv(REPORTS_DIR / "shap_importance.csv", index=False)
    print("Top 10 churn drivers (mean |SHAP|, log-odds):")
    print(importance.head(10).round(4).to_string(index=False))

    plt.figure()
    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.title("SHAP summary: what drives churn predictions (test set)")
    plt.savefig(FIGURES_DIR / "shap_summary.png", dpi=120, bbox_inches="tight")
    plt.close("all")

    plt.figure()
    shap.plots.bar(sv, max_display=15, show=False)
    plt.title("Mean |SHAP value| (global importance)")
    plt.savefig(FIGURES_DIR / "shap_importance_bar.png", dpi=120, bbox_inches="tight")
    plt.close("all")

    # 2) Dependence plots for the 2 most important features.
    for rank, name in enumerate(importance["feature"].head(2), start=1):
        idx = sv.feature_names.index(name)
        shap.plots.scatter(sv[:, idx], color=sv, show=False)
        plt.title(f"SHAP dependence: {name}")
        plt.savefig(FIGURES_DIR / f"shap_dependence_{rank}_{name.replace(' ', '_')}.png",
                    dpi=120, bbox_inches="tight")
        plt.close("all")

    # 3) One individual customer: the highest-risk customer in the test set.
    proba = pipeline.predict_proba(X_test)[:, 1]
    i = int(np.argmax(proba))
    shap.plots.waterfall(sv[i], max_display=12, show=False)
    plt.title(f"Why customer #{i} is high risk (p = {proba[i]:.2f}, actually churned = {int(y_test.iloc[i])})")
    plt.savefig(FIGURES_DIR / "shap_waterfall_customer.png", dpi=120, bbox_inches="tight")
    plt.close("all")

    reasons = explainer.top_reasons(X_test.iloc[[i]])[0]
    example = {"test_row": i, "churn_probability": float(proba[i]), "actual_churn": int(y_test.iloc[i]),
               "customer": X_test.iloc[i].to_dict(), "top_reasons": reasons}
    (REPORTS_DIR / "shap_example_customer.json").write_text(json.dumps(example, indent=2, default=str))
    print(f"\nExample customer #{i}: p(churn) = {proba[i]:.3f}, actual = {int(y_test.iloc[i])}")
    for r in reasons:
        print(f"  - {r['feature']}: {r['effect']} (SHAP {r['shap_value']:+.3f})")


if __name__ == "__main__":
    main()
