"""AutoML capstone: hyperparameter tuning with Optuna (+ optional MLflow tracking).

Idea
----
The earlier ML projects used sensible defaults. Here we **search** for better
hyperparameters with Optuna's Bayesian optimization, compare a tuned model against the
default, and (optionally) log every trial to MLflow for experiment tracking — the
professional workflow for squeezing out performance.

Real data: the UCI Wisconsin Breast Cancer set bundled in scikit-learn (offline).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score


@dataclass
class AutoMLResult:
    best_params: dict
    best_cv_score: float
    default_test_auc: float
    tuned_test_auc: float
    history: pd.DataFrame   # trial -> value


def _data():
    b = load_breast_cancer()
    return b.data, b.target


def run_optuna(n_trials: int = 30, seed: int = 42, use_mlflow: bool = False) -> AutoMLResult:
    """Tune a GradientBoostingClassifier; compare to defaults on a holdout."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    X, y = _data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)

    mlflow = _maybe_mlflow(use_mlflow)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 400),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        }
        model = GradientBoostingClassifier(random_state=seed, **params)
        score = cross_val_score(model, X_tr, y_tr, cv=3, scoring="roc_auc").mean()
        if mlflow is not None:
            with mlflow.start_run(nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("cv_roc_auc", float(score))
        return score

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    # Compare default vs tuned on the untouched test set.
    default = GradientBoostingClassifier(random_state=seed).fit(X_tr, y_tr)
    tuned = GradientBoostingClassifier(random_state=seed, **study.best_params).fit(X_tr, y_tr)
    default_auc = round(float(roc_auc_score(y_te, default.predict_proba(X_te)[:, 1])), 4)
    tuned_auc = round(float(roc_auc_score(y_te, tuned.predict_proba(X_te)[:, 1])), 4)

    history = pd.DataFrame({"trial": [t.number for t in study.trials],
                            "cv_roc_auc": [t.value for t in study.trials]})
    return AutoMLResult(
        best_params=study.best_params,
        best_cv_score=round(float(study.best_value), 4),
        default_test_auc=default_auc,
        tuned_test_auc=tuned_auc,
        history=history,
    )


def _maybe_mlflow(use_mlflow: bool):
    if not use_mlflow:
        return None
    try:
        import mlflow

        mlflow.set_experiment("automl-optuna")
        return mlflow
    except Exception:
        return None  # MLflow optional; degrade silently (documented in README)
