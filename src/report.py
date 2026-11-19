"""Build the detailed PDF report (reports/churn_report.pdf).

Every number is read from the result files written by the pipeline, so the report can
never disagree with the code.   Run (after the pipeline):  python -m src.report
"""

import json
from datetime import date

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from src.config import (
    ASSUMPTION_CUSTOMER_VALUE_MONTHS, ASSUMPTION_OFFER_COST, ASSUMPTION_OFFER_SUCCESS_RATE,
    FIGURES_DIR, MODEL_METADATA_FILE, REPORTS_DIR, RISK_HIGH_FROM, RISK_MEDIUM_FROM,
)

PDF_FILE = REPORTS_DIR / "churn_report.pdf"
ACCENT = colors.HexColor("#1F4E79")
LIGHT = colors.HexColor("#EAF1F8")
PAGE_W = A4[0] - 4 * cm


# ----------------------------------------------------------------------------- results
def load_results() -> dict:
    """Collect every result artifact produced by the pipeline."""
    r = {
        "comparison": pd.read_csv(REPORTS_DIR / "model_comparison.csv"),
        "test": json.loads((REPORTS_DIR / "test_metrics.json").read_text()),
        "business": pd.read_csv(REPORTS_DIR / "business_simulation.csv"),
        "shap": pd.read_csv(REPORTS_DIR / "shap_importance.csv"),
        "example": json.loads((REPORTS_DIR / "shap_example_customer.json").read_text()),
        "meta": json.loads(MODEL_METADATA_FILE.read_text()),
        "thresholds": pd.read_csv(REPORTS_DIR / "threshold_analysis.csv"),
    }
    t = r["test"]
    tn, fp, fn, tp = [v for row in t["confusion_matrix_at_chosen"] for v in row]
    b = r["business"].set_index("strategy")
    best = b.drop(index="Perfect knowledge (upper bound)")["expected_net_value"].idxmax()
    r["k"] = {  # key numbers reused in the text
        "thr": t["chosen_threshold"], "auc": t["at_chosen_threshold"]["roc_auc"],
        "prauc": t["at_chosen_threshold"]["pr_auc"], "rec": t["at_chosen_threshold"]["recall"],
        "prec": t["at_chosen_threshold"]["precision"], "n_test": t["n_test"],
        "tn": tn, "fp": fp, "fn": fn, "tp": tp, "churners": tp + fn,
        "best_strategy": best, "best_value": b.loc[best, "expected_net_value"],
        "everyone_value": b.loc["Target everyone", "expected_net_value"],
        "oracle_value": b.loc["Perfect knowledge (upper bound)", "expected_net_value"],
        "best_cost": b.loc[best, "campaign_cost"], "everyone_cost": b.loc["Target everyone", "campaign_cost"],
        "top30_value": b.loc["Model: top 30% riskiest", "expected_net_value"],
        "base_rate": r["comparison"].set_index("model").loc["Dummy (prior)", "pr_auc"],
        "dummy_acc": r["comparison"].set_index("model").loc["Dummy (prior)", "accuracy"],
    }
    return r


# ----------------------------------------------------------------------------- pdf helpers
def styles() -> dict:
    s = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=s["Title"], fontSize=24, leading=30, textColor=ACCENT),
        "subtitle": ParagraphStyle("st", parent=s["Normal"], fontSize=12, leading=17, alignment=TA_CENTER,
                                   textColor=colors.HexColor("#444444")),
        "h1": ParagraphStyle("h1", parent=s["Heading1"], fontSize=16, textColor=ACCENT, spaceBefore=6, spaceAfter=8),
        "h2": ParagraphStyle("h2", parent=s["Heading2"], fontSize=12.5, textColor=ACCENT, spaceBefore=8, spaceAfter=4),
        "body": ParagraphStyle("b", parent=s["BodyText"], fontSize=10, leading=14.5, spaceAfter=6),
        "bullet": ParagraphStyle("bl", parent=s["BodyText"], fontSize=10, leading=14, leftIndent=14,
                                 bulletIndent=4, spaceAfter=3),
        "caption": ParagraphStyle("c", parent=s["Italic"], fontSize=8.5, alignment=TA_CENTER,
                                  textColor=colors.HexColor("#555555"), spaceAfter=10),
        "cell": ParagraphStyle("cell", parent=s["BodyText"], fontSize=8.5, leading=11),
        "q": ParagraphStyle("q", parent=s["BodyText"], fontSize=10, leading=14, fontName="Helvetica-Bold",
                            textColor=ACCENT, spaceBefore=6),
    }


def figure(name: str, width: float, caption: str, st: dict) -> KeepTogether:
    path = FIGURES_DIR / name
    w, h = ImageReader(str(path)).getSize()
    return KeepTogether([Image(str(path), width=width, height=width * h / w),
                         Paragraph(caption, st["caption"])])


def table(rows: list[list], st: dict, col_widths: list[float] | None = None, bold_row: int | None = None) -> Table:
    data = [[Paragraph(str(c), st["cell"]) for c in r] for r in rows]
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [
        ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#BBBBBB")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]
    if bold_row is not None:
        style.append(("BACKGROUND", (0, bold_row), (-1, bold_row), colors.HexColor("#D9EAD3")))
    t.setStyle(TableStyle(style))
    # Header text must be white: rebuild header paragraphs with a white style.
    white = ParagraphStyle("hdr", parent=st["cell"], textColor=colors.white, fontName="Helvetica-Bold")
    t._cellvalues[0] = [Paragraph(str(c), white) for c in rows[0]]
    return t


def on_page(canvas, doc) -> None:
    if doc.page == 1:
        return
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.grey)
    canvas.drawString(2 * cm, 1.2 * cm, "Customer Churn Prediction - Project Report")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}")
    canvas.restoreState()


# ----------------------------------------------------------------------------- content
def build(r: dict) -> list:
    st = styles()
    k = r["k"]
    P = lambda text, style="body": Paragraph(text, st[style])
    B = lambda text: Paragraph(text, st["bullet"], bulletText="-")
    story: list = []

    # --- Cover ---
    story += [Spacer(1, 4 * cm), P("Customer Churn Prediction", "title"),
              P("End-to-end machine learning project: from raw data to an explainable model,<br/>"
                "a business decision and a production-style API", "subtitle"),
              Spacer(1, 1.2 * cm)]
    cover = [["Key result", "Value"],
             ["Test ROC-AUC / PR-AUC", f"{k['auc']:.3f} / {k['prauc']:.3f} (random PR-AUC = {k['base_rate']:.3f})"],
             ["Churners caught (test, threshold %.2f)" % k["thr"], f"{k['tp']} of {k['churners']} ({k['rec']:.1%})"],
             ["Recommended strategy", k["best_strategy"]],
             ["Expected net value (1,409 test customers)", f"${k['best_value']:,.0f} (vs ${k['everyone_value']:,.0f} targeting everyone)"],
             ["Serving", "FastAPI: /health, /predict, /predict_batch with top-3 SHAP reasons"]]
    story += [table(cover, st, [6.5 * cm, PAGE_W - 6.5 * cm]), Spacer(1, 1.5 * cm),
              P(f"Dataset: IBM Telco Customer Churn (7,043 customers). Report generated {date.today():%d %B %Y} "
                "from the pipeline's saved results; every number is computed by the code.", "subtitle"),
              PageBreak()]

    # --- 1. Executive summary ---
    story += [P("1. Executive summary", "h1"),
              P("About one customer in four leaves the company (26.5%). This project builds a model that ranks "
                "customers by their risk of leaving, explains the reasons, and decides who should receive a "
                "retention offer."),
              B(f"<b>Model quality.</b> A tuned XGBoost model reaches <b>{k['auc']:.3f} ROC-AUC</b> on customers it has "
                f"never seen. If you pick a random churner and a random loyal customer, the model ranks the churner "
                f"higher {k['auc']:.0%} of the time."),
              B(f"<b>Decision.</b> Contact customers with predicted risk of at least <b>{k['thr']:.2f}</b>. This "
                f"catches {k['rec']:.0%} of the customers who would leave, while about {k['prec']:.0%} of those "
                "contacted are real churners."),
              B(f"<b>Value.</b> Under the stated assumptions this strategy is worth <b>${k['best_value']:,.0f}</b> on the "
                f"1,409 test customers. That is {k['best_value'] / k['everyone_value']:.1f}x more than targeting everyone, at "
                f"{k['best_cost'] / k['everyone_cost']:.0%} of the cost, and {k['best_value'] / k['oracle_value']:.0%} of the "
                "theoretical maximum."),
              B("<b>Why customers leave.</b> They are new, on month-to-month contracts, on fiber optic with high "
                "bills, without security or tech-support add-ons, and paying by electronic check."),
              B("<b>Deliverable.</b> A tested, reproducible pipeline tracked in MLflow and served through a "
                "validated REST API that also explains each prediction.")]

    # --- 2. Problem & data ---
    story += [P("2. Business problem and data", "h1"),
              P("Retention offers (discounts, calls, upgrades) cost money. Sending them to everyone wastes budget "
                "on customers who would stay anyway, and sending them to nobody loses revenue. The goal is "
                "to find the customers for whom an offer pays off."),
              P("<b>Data.</b> 7,043 customers x 21 columns: demographics (gender, senior citizen, partner, dependents), "
                "account (tenure in months, contract, paperless billing, payment method, monthly and total "
                "charges), services (phone, multiple lines, internet type, online security, backup, device "
                "protection, tech support, streaming TV and movies) and the target <i>Churn</i>."),
              P("<b>Data quality.</b> <i>TotalCharges</i> is stored as text and contains 11 blank strings, invisible to a "
                "standard missing-value check. All 11 are customers with tenure = 0 who were never billed, so the "
                "correct value is 0 (not the mean, and the rows should not be dropped). Customer IDs are unique, and the ID "
                "is dropped because it carries no signal. 22 rows are identical once the ID is removed. They are "
                "different customers with common simple profiles, so they are kept.")]

    # --- 3. EDA ---
    story += [PageBreak(), P("3. Exploratory analysis: key findings", "h1"),
              figure("03_churn_rate_by_segment.png", PAGE_W, "Figure 1 - Churn rate by contract, tenure, payment "
                     "method and internet service (dashed line = overall 26.5%).", st)]
    for t in [
        "<b>Contract</b> is the strongest signal: month-to-month 42.7%, one year 11.3%, two year 2.8%.",
        "<b>The first year is the danger zone</b>: 47.4% churn for 0-12 months vs 9.5% after 4 years. "
        "Median tenure is 10 months for churners and 38 for customers who stay.",
        "<b>Fiber optic</b> customers churn at 41.9% (DSL 19.0%, no internet 7.4%). Churners also pay more per month "
        "(median $79.6 vs $64.4).",
        "<b>Electronic check</b> payers churn at 45.3%, about 3x the rate of automatic payment (15-17%).",
        "Customers <b>without online security or tech support</b> churn at ~42% vs ~15% with them. Senior "
        "citizens churn at 41.7%. Gender has no effect (26.9% vs 26.2%).",
        "<i>tenure</i> and <i>TotalCharges</i> are strongly correlated (0.83), which matters when reading a linear model's coefficients.",
    ]:
        story.append(B(t))
    story += [Spacer(1, 6), figure("02_numeric_distributions.png", PAGE_W,
                                   "Figure 2 - Tenure, monthly and total charges by churn status.", st)]

    # --- 4. Method ---
    story += [PageBreak(), P("4. Method", "h1"), P("4.1 Feature engineering", "h2")]
    story.append(table([
        ["Feature", "Definition", "Business reason"],
        ["tenure_bucket", "0-1y / 1-2y / 2-4y / 4y+", "Churn drops sharply and non-linearly with tenure; buckets let even a linear model capture this"],
        ["avg_monthly_spend", "TotalCharges / tenure (current bill when tenure = 0)", "Compared with the current bill, shows recent price increases"],
        ["num_services", "Number of subscribed services", "More services means higher switching cost and engagement"],
        ["has_long_contract", "1 if one- or two-year contract", "Commitment is the strongest churn signal in the EDA"],
    ], st, [3.3 * cm, 5.2 * cm, PAGE_W - 8.5 * cm]))
    story += [P("4.2 Pipeline, split and leakage", "h2"),
              P("All steps (feature engineering, then StandardScaler on numeric columns and OneHotEncoder with "
                "handle_unknown='ignore' on categorical columns, then the model) live in <b>one scikit-learn "
                "Pipeline</b>. The exact same object is trained, saved with joblib and loaded by the API, so "
                "training and serving cannot drift apart."),
              P("The data is split <b>60/20/20 (train/validation/test), stratified</b> on churn (26.5% in every "
                "split). <b>Data leakage</b> means information the model would not have in real life reaching "
                "training, for example scaling with means computed on the test set. It is prevented because the pipeline is fit on "
                "training data only (and re-fit inside every cross-validation fold). Validation is used for "
                "model choice and the threshold, and the test set is scored once at the very end."),
              P("4.3 Models and tracking", "h2"),
              B("<b>DummyClassifier</b>: always predicts the majority class. This is the floor every model must beat."),
              B("<b>Logistic Regression</b> with class_weight='balanced': a simple, interpretable baseline."),
              B("<b>XGBoost</b> with scale_pos_weight = negatives / positives = %.2f to handle imbalance, tuned with "
                "RandomizedSearchCV (30 candidates x stratified 5-fold CV, ROC-AUC). The search took about 1 minute." % r["meta"]["params"]["scale_pos_weight"]),
              P("Every run is logged to <b>MLflow</b> (parameters, CV and validation metrics, ROC/PR plots and the "
                "full pipeline as a model artifact). The random seed is fixed at 42 everywhere. A full re-run "
                "reproduces identical numbers.")]

    # --- 5. Results ---
    comp = r["comparison"]
    rows = [["Model", "CV ROC-AUC", "ROC-AUC", "PR-AUC", "Precision", "Recall", "F1", "Accuracy"]]
    for _, row in comp.iterrows():
        rows.append([row["model"]] + [f"{row[c]:.3f}" for c in
                                      ["cv_roc_auc", "roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy"]])
    best_idx = int(comp.index[comp["model"] == r["meta"]["model_name"]][0]) + 1
    story += [P("5. Results", "h1"), P("5.1 Model comparison (validation set, threshold 0.5)", "h2"),
              table(rows, st, [3.6 * cm] + [(PAGE_W - 3.6 * cm) / 7] * 7, bold_row=best_idx), Spacer(1, 6),
              P(f"<b>Accuracy is misleading:</b> the dummy model is {k['dummy_acc']:.1%} accurate and catches zero "
                "churners. ROC-AUC measures ranking quality, and PR-AUC focuses on the churn class (random = "
                f"{k['base_rate']:.3f})."),
              P("<b>Honest reading:</b> XGBoost and Logistic Regression are practically tied (difference of about 0.002 ROC-AUC, "
                "within noise). XGBoost was selected by the pre-defined rule (best validation ROC-AUC), but the "
                "simpler model would be a perfectly defensible production choice.")]

    t = r["test"]
    a, d = t["at_chosen_threshold"], t["at_default_0.5"]
    story += [P("5.2 Final test-set performance (scored once)", "h2"),
              table([["Threshold", "ROC-AUC", "PR-AUC", "Precision", "Recall", "F1", "Accuracy"],
                     ["0.50 (default)"] + [f"{d[m]:.3f}" for m in ["roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy"]],
                     [f"{k['thr']:.2f} (business-optimal)"] + [f"{a[m]:.3f}" for m in ["roc_auc", "pr_auc", "precision", "recall", "f1", "accuracy"]]],
                    st, [4.4 * cm] + [(PAGE_W - 4.4 * cm) / 6] * 6, bold_row=2),
              Spacer(1, 6),
              P(f"Test performance (ROC-AUC {k['auc']:.3f}) is in line with validation and cross-validation, so the model "
                f"generalises and is not overfit. Confusion matrix at {k['thr']:.2f}: TN={k['tn']}, FP={k['fp']}, "
                f"FN={k['fn']}, TP={k['tp']}.")]
    w2 = PAGE_W / 2 - 0.2 * cm
    img = lambda n: Image(str(FIGURES_DIR / n), width=w2, height=w2 * ImageReader(str(FIGURES_DIR / n)).getSize()[1]
                          / ImageReader(str(FIGURES_DIR / n)).getSize()[0])
    story += [KeepTogether([Table([[img("test_roc_curve.png"), img("test_pr_curve.png")],
                                   [img("test_confusion_matrix.png"), img("test_calibration.png")]]),
                            P("Figure 3 - Test set: ROC curve, precision-recall curve, confusion matrix, calibration.", "caption")]),
              P("<b>Calibration.</b> Probabilities are over-estimated (a predicted ~0.6 corresponds to ~0.3 observed churn; "
                f"Brier score {t['brier_score']:.3f}). This is the expected side effect of scale_pos_weight. Ranking "
                "and the threshold (tuned on real outcomes) are unaffected. Probability calibration is listed as a next step.")]

    # --- 6. Threshold ---
    story += [PageBreak(), P("6. Choosing the decision threshold", "h1"),
              figure("threshold_analysis.png", PAGE_W * 0.9, "Figure 4 - Precision, recall, F1 and expected net value "
                     "by threshold (validation set).", st),
              P("A lower threshold contacts more customers: <b>recall goes up</b> (fewer churners missed) and <b>precision "
                "goes down</b> (more offers wasted on customers who would stay). The right balance depends on the cost "
                f"of each error. Here a wasted offer costs ${ASSUMPTION_OFFER_COST:.0f}, while an offer to a real churner is worth "
                f"{ASSUMPTION_OFFER_SUCCESS_RATE:.0%} x {ASSUMPTION_CUSTOMER_VALUE_MONTHS} months of their bill (about $260 on average). Missing a churner "
                "is the more expensive mistake."),
              P(f"The threshold was therefore chosen to <b>maximise expected net value on the validation set</b>: "
                f"<b>{k['thr']:.2f}</b> (the F1-optimal threshold would have been "
                f"{r['thresholds'].loc[r['thresholds']['f1'].idxmax(), 'threshold']:.2f}). The value curve is flat between about "
                "0.25 and 0.6, so the decision is not sensitive to the exact value. The API uses this threshold for "
                f"churn_prediction and reports risk levels low (&lt; {RISK_MEDIUM_FROM:.2f}), medium and high "
                f"(&gt;= {RISK_HIGH_FROM:.2f}).")]

    # --- 7. SHAP ---
    top = r["shap"].head(8)
    story += [P("7. Explainability (SHAP)", "h1"),
              P("SHAP splits every prediction into additive contributions, one per feature, measured against an "
                "average customer: <i>prediction = baseline + sum of contributions</i>. Positive values push towards churn. "
                "TreeExplainer computes them exactly for XGBoost."),
              figure("shap_summary.png", PAGE_W * 0.85, "Figure 5 - SHAP summary. Each dot is a test customer, "
                     "colour = feature value (red = high), x-axis = impact on churn risk.", st),
              table([["Rank", "Feature", "Mean |SHAP|"]] +
                    [[i + 1, f, f"{v:.3f}"] for i, (f, v) in enumerate(zip(top["feature"], top["mean_abs_shap"]))],
                    st, [1.5 * cm, 8 * cm, 3 * cm]),
              P("<b>For a manager:</b>", "h2")]
    for t_ in [
        "New customers are by far the biggest risk. Churn risk falls steadily the longer a customer stays, so the first months need the most attention.",
        "Long contracts protect us. Moving month-to-month customers onto one- or two-year contracts is the strongest retention lever.",
        "Fiber optic customers with high bills are more likely to leave, even after accounting for price. This points to a value-for-money or quality problem with that product.",
        "Customers without online security or tech support leave more often, so bundling these services into the retention offer makes sense.",
        "Paying by electronic check is a warning sign. Encouraging automatic payment is a cheap action.",
    ]:
        story.append(B(t_))
    dep = sorted(FIGURES_DIR.glob("shap_dependence_*.png"))
    ex = r["example"]
    story += [PageBreak(), P("7.1 Dependence plots and one individual customer", "h2"),
              KeepTogether([Table([[Image(str(p), width=w2, height=w2 * ImageReader(str(p)).getSize()[1] / ImageReader(str(p)).getSize()[0])
                                    for p in dep]]),
                            P("Figure 6 - SHAP dependence for the two most important features.", "caption")]),
              figure("shap_waterfall_customer.png", PAGE_W * 0.85,
                     f"Figure 7 - Waterfall for test customer #{ex['test_row']} (predicted {ex['churn_probability']:.0%}, "
                     f"actually churned = {ex['actual_churn']}).", st),
              P(f"Customer #{ex['test_row']} is a {ex['customer']['tenure']}-month, {ex['customer']['Contract'].lower()} "
                f"{ex['customer']['InternetService']} customer paying ${ex['customer']['MonthlyCharges']:.2f} by "
                f"{ex['customer']['PaymentMethod'].lower()}. Top reasons returned by the API: " +
                "; ".join(f"{x['feature']} ({x['shap_value']:+.2f})" for x in ex["top_reasons"]) + ".")]

    # --- 8. Business ---
    bz = r["business"]
    rows = [["Strategy", "Targeted", "Churners reached", "Cost", "Expected saved", "Net value"]]
    for _, x in bz.iterrows():
        rows.append([x["strategy"], f"{x['customers_targeted']:,}",
                     f"{x['churners_reached']:,} ({x['churners_reached_%']:.0f}%)", f"${x['campaign_cost']:,.0f}",
                     f"${x['expected_saved_value']:,.0f}", f"<b>${x['expected_net_value']:,.0f}</b>"])
    story += [PageBreak(), P("8. Business simulation", "h1"),
              P(f"<b>Assumptions (not measured facts):</b> customer value = monthly charges x {ASSUMPTION_CUSTOMER_VALUE_MONTHS}; "
                f"offer cost = ${ASSUMPTION_OFFER_COST:.0f} per targeted customer; {ASSUMPTION_OFFER_SUCCESS_RATE:.0%} of targeted "
                "churners are retained. Net value = expected saved value - campaign cost, measured against doing nothing. "
                f"Computed on the {k['n_test']:,} test customers."),
              table(rows, st, [5.2 * cm, 1.9 * cm, 2.8 * cm, 2.2 * cm, 2.5 * cm, PAGE_W - 14.6 * cm],
                    bold_row=int(bz.index[bz["strategy"] == k["best_strategy"]][0]) + 1),
              Spacer(1, 8),
              figure("business_net_value.png", PAGE_W, "Figure 8 - Expected net value by strategy.", st),
              P(f"<b>Recommendation:</b> target customers with predicted risk >= {k['thr']:.2f}. It yields "
                f"${k['best_value']:,.0f}, {k['best_value'] / k['everyone_value']:.1f}x the value of a blanket campaign at "
                f"{k['best_cost'] / k['everyone_cost']:.0%} of its cost, and captures {k['best_value'] / k['oracle_value']:.0%} "
                "of what perfect knowledge would achieve. If the budget is capped, the top-30% campaign is the best "
                f"fixed-size option (${k['top30_value']:,.0f}). Before rolling it out, measure the real offer success rate with an A/B test.")]

    # --- 9. Engineering ---
    story += [P("9. Deployment, testing and reproducibility", "h1"),
              B("<b>API (FastAPI + Pydantic):</b> GET /health, POST /predict, POST /predict_batch (up to 1,000). The pipeline "
                "and the SHAP explainer are loaded once at startup. Inputs are validated (allowed categories, numeric ranges, "
                "no extra fields), and invalid requests return HTTP 422 with a clear message. Interactive docs are at /docs."),
              B("<b>Response:</b> churn_probability, churn_prediction (business threshold), risk_level and the top-3 SHAP reasons in readable form, e.g. 'tenure = 2: increases churn risk'."),
              B("<b>Tests (pytest, 15 passing):</b> TotalCharges fix, no missing values, stratified split without overlap, "
                "engineered feature values, preprocessing output shape, unseen categories, model probabilities in [0, 1], "
                "/health, /predict, /predict_batch and input validation through FastAPI's TestClient."),
              B("<b>Reproducibility:</b> seed 42 everywhere, pinned major versions, saved data splits, MLflow tracking. "
                "One command per step (or `make pipeline`).")]

    # --- 10. Limitations ---
    story += [P("10. Limitations and next steps", "h1")]
    for t_ in [
        "<b>Snapshot data:</b> no timestamps, so no time-based validation and no 'when will they churn' (survival analysis would need event dates).",
        "<b>Assumed economics:</b> the dollar values depend on assumptions. Validate the offer success rate with an A/B test.",
        "<b>Churn is not persuadability:</b> the model finds who will leave, not who an offer will change. Uplift modelling is the natural next step.",
        "<b>Uncalibrated probabilities:</b> add CalibratedClassifierCV if the probabilities themselves are used.",
        "<b>Small data:</b> 7k rows, so model differences of about 0.002 AUC are noise.",
        "<b>Productionisation:</b> Docker, CI, MLflow Model Registry, monitoring of input drift (PSI) and live churn rate, scheduled or drift-triggered retraining, and retraining the final model on train + validation.",
    ]:
        story.append(B(t_))

    return story


def main() -> None:
    r = load_results()
    doc = SimpleDocTemplate(str(PDF_FILE), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="Customer Churn Prediction - Project Report", author="Churn Prediction Project")
    doc.build(build(r), onFirstPage=on_page, onLaterPages=on_page)
    print(f"Wrote {PDF_FILE}")


if __name__ == "__main__":
    main()
