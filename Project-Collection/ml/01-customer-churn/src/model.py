"""Customer-churn model: a documented, multi-algorithm pipeline with an ensemble.

Business problem
----------------
A telco loses revenue when subscribers leave ("churn"). Predicting *who* will churn
lets the retention team intervene. One of the most common real paid ML tasks.

What this teaches (concepts & algorithms)
-----------------------------------------
Rather than a single model, we train and **compare several algorithms**, then combine
them in a **stacking ensemble** — the way you'd actually approach a paid engagement:
  * Logistic Regression  – fast, interpretable linear baseline
  * Random Forest        – bagging ensemble of decision trees
  * Gradient Boosting (XGBoost) – boosting ensemble, usually the strongest single model
  * **Stacking ensemble** – meta-model that learns to blend the three above
Each runs inside the *same* leak-free preprocessing ``Pipeline`` (no train/serve skew),
and we score them on metrics that matter for an imbalanced problem (ROC-AUC, precision,
recall) and keep the best.

The real dataset (IBM Telco Customer Churn) is fetched from OpenML by
``download_data.py``. ``synthetic_churn`` provides a schema-compatible offline sample
so unit tests run fast without a network.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    StackingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Churn"


@dataclass
class TrainResult:
    """Everything the UI needs after training all algorithms."""

    pipelines: dict           # name -> fitted Pipeline
    leaderboard: pd.DataFrame  # one row per algorithm, sorted best-first
    best_name: str
    metrics: dict             # best model's metrics
    confusion: np.ndarray     # best model's confusion matrix
    importances: pd.DataFrame  # churn drivers (from the tree ensemble)
    sample_row: pd.Series

    @property
    def best(self) -> Pipeline:
        return self.pipelines[self.best_name]


def synthetic_churn(n: int = 600, seed: int = 0) -> pd.DataFrame:
    """Generate a small, schema-realistic churn dataset for offline tests/demos."""
    rng = np.random.default_rng(seed)
    tenure = rng.integers(0, 72, n)
    monthly = rng.uniform(20, 120, n).round(2)
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.2])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.35, 0.45, 0.2])
    logit = (
        -1.0 - 0.04 * tenure + 0.015 * monthly
        + (contract == "Month-to-month") * 1.3 + (internet == "Fiber optic") * 0.6
    )
    churn = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame(
        {
            "tenure": tenure, "MonthlyCharges": monthly,
            "TotalCharges": (tenure * monthly).round(2),
            "Contract": contract, "InternetService": internet,
            "PaymentMethod": rng.choice(
                ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n),
            TARGET: np.where(churn == 1, "Yes", "No"),
        }
    )


def _split_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    features = [c for c in df.columns if c != TARGET]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    return numeric, [c for c in features if c not in numeric]


def _preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), numeric),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ]
    )


def candidate_estimators() -> dict:
    """The algorithms we compare. XGBoost if available, else sklearn boosting."""
    try:
        from xgboost import XGBClassifier

        booster = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.08, subsample=0.9,
            colsample_bytree=0.9, eval_metric="logloss", n_jobs=2, random_state=42)
    except Exception:  # graceful fallback keeps the project runnable without xgboost
        booster = GradientBoostingClassifier(random_state=42)

    lr = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(n_estimators=300, n_jobs=2, random_state=42)
    estimators = {"Logistic Regression": lr, "Random Forest": rf, "Gradient Boosting": booster}
    # Stacking ensemble blends the three base learners with a logistic meta-model.
    estimators["Stacking Ensemble"] = StackingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("gb", booster)],
        final_estimator=LogisticRegression(max_iter=1000), cv=3, n_jobs=2,
    )
    return estimators


def _evaluate(pipe: Pipeline, X_te, y_te) -> tuple[dict, np.ndarray]:
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    metrics = {
        "ROC-AUC": round(float(roc_auc_score(y_te, proba)), 3),
        "Precision": round(float(precision_score(y_te, pred, zero_division=0)), 3),
        "Recall": round(float(recall_score(y_te, pred, zero_division=0)), 3),
    }
    return metrics, confusion_matrix(y_te, pred)


def train(df: pd.DataFrame, test_size: float = 0.25, seed: int = 42) -> TrainResult:
    """Train every candidate algorithm + the ensemble; return a comparison."""
    df = df.copy()
    if "TotalCharges" in df:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    y = df[TARGET].astype(str).str.strip().str.lower().isin(["yes", "1", "true"]).astype(int)
    X = df.drop(columns=[TARGET])
    numeric, categorical = _split_columns(df)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)

    pipelines, rows, confusions = {}, [], {}
    for name, est in candidate_estimators().items():
        pipe = Pipeline([("pre", _preprocessor(numeric, categorical)), ("model", est)])
        pipe.fit(X_tr, y_tr)
        metrics, cm = _evaluate(pipe, X_te, y_te)
        pipelines[name] = pipe
        confusions[name] = cm
        rows.append({"Algorithm": name, **metrics})

    leaderboard = (
        pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True))
    best_name = leaderboard.iloc[0]["Algorithm"]

    return TrainResult(
        pipelines=pipelines,
        leaderboard=leaderboard,
        best_name=best_name,
        metrics={k: leaderboard.iloc[0][k] for k in ("ROC-AUC", "Precision", "Recall")},
        confusion=confusions[best_name],
        importances=_importances(pipelines["Gradient Boosting"]),
        sample_row=X_te.iloc[0],
    )


def _importances(pipe: Pipeline, top: int = 10) -> pd.DataFrame:
    """Churn drivers from the tree ensemble (always available, interpretable)."""
    model = pipe.named_steps["model"]
    try:
        names = list(pipe.named_steps["pre"].get_feature_names_out())
    except Exception:
        names = [f"f{i}" for i in range(len(getattr(model, "feature_importances_", [])))]
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame({"feature": [], "importance": []})
    imp = pd.DataFrame({"feature": names, "importance": model.feature_importances_})
    return imp.sort_values("importance", ascending=False).head(top).reset_index(drop=True)


def score_customer(result: TrainResult, row: pd.Series) -> float:
    """Churn probability (0–1) for one customer from the best model."""
    return float(result.best.predict_proba(pd.DataFrame([row]))[:, 1][0])
