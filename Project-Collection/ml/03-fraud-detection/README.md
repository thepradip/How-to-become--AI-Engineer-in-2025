# ML 03 · Credit-Card Fraud Detection 🟡

**Problem.** Catch fraudulent transactions among millions of legitimate ones. Fraud is **rare**
(~0.17%), so a model that predicts "never fraud" is 99.8% accurate and useless — the real goal is
**recall on fraud** without drowning analysts in false alarms.

**What you build.** An imbalanced-classification pipeline using **SMOTE** oversampling + multiple
algorithms + a stacking ensemble, in the shared chat UI.

## Dataset (real)
**Credit-Card Fraud** via OpenML (`creditcard`, 284,807 transactions, 492 frauds). `download_data.py`
fetches it (large). Tests/offline use a smaller synthetic *imbalanced* sample with the same challenge.

## Approach
- Leak-free preprocessing inside an **imbalanced-learn `Pipeline`** so **SMOTE runs on training folds
  only** (never on test — a classic leakage trap).
- Compare Logistic Regression, Random Forest, Gradient Boosting + a stacking ensemble.
- Judge on **ROC-AUC, precision, recall, F1** — accuracy is meaningless here.

## Run it
```bash
pip install -r requirements.txt
python src/download_data.py
pytest -q
streamlit run app.py
```

## What you learned
Why accuracy lies on imbalanced data · using SMOTE *correctly* (inside the pipeline) · trading
precision vs. recall for an alert budget.

## Tested on
CPU (Python 3.11), offline on the synthetic imbalanced sample; real data via OpenML when online.
No GPU. ✅ (SMOTE via `imbalanced-learn`.)

> **Freelance relevance.** Fraud/anomaly detection is a recurring, well-paid task in fintech,
> payments, and insurance.
