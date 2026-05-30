# ML 10 · AutoML — Hyperparameter Tuning (Optuna) 🔴 · *capstone*

**Problem.** Default hyperparameters leave performance on the table. This capstone shows the
professional tuning workflow: search the hyperparameter space with **Optuna** (Bayesian
optimization), compare the tuned model to the default, and optionally track every trial in **MLflow**.

**What you build.** An Optuna study tuning a `GradientBoostingClassifier`, a default-vs-tuned holdout
comparison, an optimization-history chart, and the best hyperparameters — in the shared chat UI.

## Dataset (real)
UCI Wisconsin Breast Cancer (bundled in scikit-learn) — real and fully offline.

## Approach
Define a search space → optimize 3-fold CV ROC-AUC with Optuna's TPE sampler → refit the best params
→ compare to the default model on an untouched test set → plot best-score-so-far.

## Run it
```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py     # set trials in the sidebar, type "tune"
# optional: pip install mlflow && mlflow ui     # then run with MLflow logging on
```

## What you learned
Bayesian hyperparameter search vs. grid/random · clean train/CV/test separation · experiment tracking
with MLflow · reading an optimization-history curve. **This is the tuning skill expected on paid ML work.**

## Tested on
CPU (Python 3.11), fully offline (5-trial fast test in CI; use 30–100 trials for real gains). No GPU. ✅
MLflow is optional — the app runs without it.

> **Freelance relevance.** "Improve our model's accuracy" engagements are largely feature work +
> disciplined hyperparameter tuning like this.
