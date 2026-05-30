# ML 01 · Customer Churn Prediction 🟢

**Problem.** A telecom company loses recurring revenue every time a subscriber cancels. If we can
flag who is *likely* to churn, the retention team can intervene with offers before they leave.
Predicting churn is one of the most common ML engagements — it maps directly to retention
budgets, so businesses pay for it.

**What you build.** A leak-free scikit-learn `Pipeline` that trains and **compares multiple
algorithms — Logistic Regression, Random Forest, Gradient Boosting (XGBoost) — and a stacking
ensemble**, keeps the best, scores a customer's churn probability, reports business-relevant metrics,
and explains the top churn drivers — all inside the shared **chat UI**.

## Dataset (real)
IBM **Telco Customer Churn** (~7,000 customers, 21 features) via **OpenML**
(mirrored on Kaggle as `blastchar/telco-customer-churn`). Downloaded on first run into `./data`
(git-ignored). If you're offline, the app falls back to a schema-compatible synthetic sample and
says so.

## Approach
1. **Preprocess** in a `ColumnTransformer`: numeric → median-impute + scale; categorical →
   most-frequent-impute + one-hot. Bundled with each model in one `Pipeline` (no train/serve skew).
2. **Train & compare algorithms** on the same split: Logistic Regression (linear baseline),
   Random Forest (bagging), Gradient Boosting / XGBoost (boosting), and a **Stacking ensemble**
   that blends all three with a logistic meta-model. The leaderboard shows which wins.
3. **Evaluate** with **ROC-AUC, precision, recall** — accuracy alone lies on imbalanced churn.
4. **Explain** with gradient-boosting gain importances (extension: add SHAP for per-customer reasons).

## Run it
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python src/download_data.py     # fetches the real dataset
pytest -q                       # all tests should pass
streamlit run app.py            # chat UI: type "train", then "drivers" / "demo"
```

## Chat commands
- `train` → trains all algorithms + ensemble, shows the leaderboard + best-model metrics + confusion matrix
- `compare` → the algorithm leaderboard (table + bar chart)
- `drivers` → top churn drivers (table + bar chart)
- `demo` → scores an example customer's churn probability with the best model

## Results (typical)
On the real Telco data, expect **ROC-AUC ≈ 0.84** with month-to-month contract, tenure, and monthly
charges among the strongest drivers. (Synthetic-sample numbers differ.)

## What you learned
- Building a leak-free preprocessing + model pipeline
- Choosing metrics that match an imbalanced business problem
- Communicating drivers to non-technical stakeholders (the retention team)

## Extensions
- Add **SHAP** for per-customer explanations · calibrate probabilities · tune a decision threshold to
  the cost of a retention offer · ship as a FastAPI endpoint.

## Tested on
CPU (macOS, Python 3.11). `pytest` runs fully offline on the synthetic sample; the app uses the real
OpenML data when online. No GPU required.

> **macOS + XGBoost note:** XGBoost's native lib needs the OpenMP runtime — install it once with
> `brew install libomp`. If it's missing, this project **automatically falls back** to scikit-learn's
> `GradientBoostingClassifier` (same role in the ensemble), so everything still runs.

> **Why it matters.** Churn/retention modelling + a simple "score a customer" UI is a frequent
> request from subscription businesses (telco, SaaS, gyms).
