"""Shared tabular ML engine — multi-algorithm training, comparison & ensembling.

Most ``ml/`` projects share the same skeleton: take a DataFrame, preprocess
numeric + categorical columns without leakage, train **several algorithms plus a
stacking ensemble**, compare them on held-out data, and explain the drivers. This
module implements that once so each project is a thin, domain-specific wrapper —
it just supplies *which* dataset and *what* the target is.

Concepts covered (per the course goal of "use all the algorithms"):
  classification → Logistic Regression, Random Forest, Gradient Boosting + Stacking
  regression     → Ridge, Random Forest, Gradient Boosting + Stacking
Optional SMOTE oversampling for imbalanced classification (e.g. fraud).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    StackingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class TabularResult:
    pipelines: dict
    leaderboard: pd.DataFrame
    best_name: str
    metrics: dict
    importances: pd.DataFrame
    task: str                      # "classification" | "regression"
    sample_row: pd.Series
    confusion: Optional[np.ndarray] = None

    @property
    def best(self) -> Pipeline:
        return self.pipelines[self.best_name]


def _columns(df: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    feats = [c for c in df.columns if c != target]
    num = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    return num, [c for c in feats if c not in num]


def _preprocessor(num: list[str], cat: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), num),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ]
    )


def _classification_estimators() -> dict:
    lr = LogisticRegression(max_iter=1000)
    rf = RandomForestClassifier(n_estimators=300, n_jobs=2, random_state=42)
    gb = GradientBoostingClassifier(random_state=42)
    return {
        "Logistic Regression": lr, "Random Forest": rf, "Gradient Boosting": gb,
        "Stacking Ensemble": StackingClassifier(
            estimators=[("lr", lr), ("rf", rf), ("gb", gb)],
            final_estimator=LogisticRegression(max_iter=1000), cv=3, n_jobs=2),
    }


def _regression_estimators() -> dict:
    ridge = Ridge()
    rf = RandomForestRegressor(n_estimators=300, n_jobs=2, random_state=42)
    gb = GradientBoostingRegressor(random_state=42)
    return {
        "Ridge": ridge, "Random Forest": rf, "Gradient Boosting": gb,
        "Stacking Ensemble": StackingRegressor(
            estimators=[("ridge", ridge), ("rf", rf), ("gb", gb)],
            final_estimator=Ridge(), cv=3, n_jobs=2),
    }


def _make_pipeline(num, cat, estimator, use_smote: bool) -> Pipeline:
    steps = [("pre", _preprocessor(num, cat))]
    if use_smote:
        try:  # imblearn pipeline so SMOTE only runs during fit
            from imblearn.over_sampling import SMOTE
            from imblearn.pipeline import Pipeline as ImbPipeline

            return ImbPipeline(steps + [("smote", SMOTE(random_state=42)),
                                        ("model", estimator)])
        except Exception:
            pass  # imblearn not installed → plain pipeline (documented in README)
    return Pipeline(steps + [("model", estimator)])


def train_and_compare(
    df: pd.DataFrame,
    target: str,
    task: str = "classification",
    positive: Optional[str] = None,
    use_smote: bool = False,
    test_size: float = 0.25,
    seed: int = 42,
) -> TabularResult:
    """Train every candidate algorithm + ensemble and return a comparison."""
    df = df.dropna(axis=1, how="all").copy()
    num, cat = _columns(df, target)
    X = df.drop(columns=[target])

    if task == "classification":
        if positive is not None:
            y = (df[target].astype(str) == str(positive)).astype(int)
        else:
            y = pd.factorize(df[target])[0]
        strat = y
        estimators = _classification_estimators()
    else:
        y = pd.to_numeric(df[target], errors="coerce")
        strat = None
        estimators = _regression_estimators()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=strat)

    pipelines, rows, confusions = {}, [], {}
    for name, est in estimators.items():
        pipe = _make_pipeline(num, cat, est, use_smote and task == "classification")
        pipe.fit(X_tr, y_tr)
        if task == "classification":
            m, cm = _clf_metrics(pipe, X_te, y_te)
            confusions[name] = cm
        else:
            m = _reg_metrics(pipe, X_te, y_te)
        pipelines[name] = pipe
        rows.append({"Algorithm": name, **m})

    sort_key, asc = ("ROC-AUC", False) if task == "classification" else ("R²", False)
    if task == "classification" and "ROC-AUC" not in rows[0]:
        sort_key = "F1"
    lb = pd.DataFrame(rows).sort_values(sort_key, ascending=asc).reset_index(drop=True)
    best = lb.iloc[0]["Algorithm"]

    return TabularResult(
        pipelines=pipelines, leaderboard=lb, best_name=best,
        metrics={k: lb.iloc[0][k] for k in lb.columns if k != "Algorithm"},
        importances=_importances(pipelines["Gradient Boosting"]),
        task=task, sample_row=X_te.iloc[0],
        confusion=confusions.get(best) if task == "classification" else None,
    )


def _clf_metrics(pipe, X_te, y_te):
    pred = pipe.predict(X_te)
    binary = len(np.unique(y_te)) == 2
    out = {}
    if binary and hasattr(pipe, "predict_proba"):
        out["ROC-AUC"] = round(float(roc_auc_score(y_te, pipe.predict_proba(X_te)[:, 1])), 3)
    out["Precision"] = round(float(precision_score(y_te, pred, average="binary" if binary else "macro", zero_division=0)), 3)
    out["Recall"] = round(float(recall_score(y_te, pred, average="binary" if binary else "macro", zero_division=0)), 3)
    out["F1"] = round(float(f1_score(y_te, pred, average="binary" if binary else "macro", zero_division=0)), 3)
    return out, confusion_matrix(y_te, pred)


def _reg_metrics(pipe, X_te, y_te):
    pred = pipe.predict(X_te)
    rmse = float(np.sqrt(mean_squared_error(y_te, pred)))
    return {
        "R²": round(float(r2_score(y_te, pred)), 3),
        "MAE": round(float(mean_absolute_error(y_te, pred)), 2),
        "RMSE": round(rmse, 2),
    }


def _importances(pipe: Pipeline, top: int = 12) -> pd.DataFrame:
    model = pipe.named_steps["model"]
    try:
        names = list(pipe.named_steps["pre"].get_feature_names_out())
    except Exception:
        names = []
    if not hasattr(model, "feature_importances_"):
        return pd.DataFrame({"feature": [], "importance": []})
    names = names or [f"f{i}" for i in range(len(model.feature_importances_))]
    imp = pd.DataFrame({"feature": names, "importance": model.feature_importances_})
    return imp.sort_values("importance", ascending=False).head(top).reset_index(drop=True)


def make_handler(
    *,
    load_df: Callable[[], pd.DataFrame],
    target: str,
    task: str = "classification",
    positive: Optional[str] = None,
    use_smote: bool = False,
    help_text: str = "",
):
    """Build a chat ``respond`` function with train / compare / drivers / demo commands.

    Each tabular project calls this with its dataset loader + target; the multi-algorithm
    logic and chat replies are shared, so the project's handler.py stays tiny.
    """

    def _ensure(state):
        if "result" not in state:
            state["result"] = train_and_compare(
                load_df(), target, task, positive, use_smote)
        return state["result"]

    def respond(message: str, state: dict):
        from _shared.chat_ui import Reply

        msg = message.strip().lower()

        if "train" in msg or msg in {"start", "go"}:
            res = state["result"] = train_and_compare(
                load_df(), target, task, positive, use_smote)
            parts = [
                Reply(f"Trained **{len(res.pipelines)} algorithms** (incl. stacking ensemble). "
                      f"Best: **{res.best_name}**."),
                Reply(res.leaderboard, "table"),
                Reply(res.metrics, "metric"),
            ]
            if res.confusion is not None:
                parts.append(Reply(_confusion_fig(res), "figure"))
            parts.append(Reply("Try `compare`, `drivers`, or `demo`.", "text"))
            return parts

        if "compare" in msg or "leaderboard" in msg:
            res = _ensure(state)
            return [Reply("**Algorithm comparison:**"), Reply(res.leaderboard, "table"),
                    Reply(_leaderboard_fig(res), "figure")]

        if "driver" in msg or "import" in msg or "why" in msg or "feature" in msg:
            res = _ensure(state)
            if len(res.importances):
                return [Reply("**Top features** (gradient-boosting importance):"),
                        Reply(res.importances, "table"),
                        Reply(_importance_fig(res.importances), "figure")]
            return Reply("This model type doesn't expose feature importances.")

        if "demo" in msg or "predict" in msg or "score" in msg:
            res = _ensure(state)
            row = res.sample_row
            pred = res.best.predict(pd.DataFrame([row]))[0]
            metric = {"Prediction": _fmt(pred, res.task)}
            if res.task == "classification" and hasattr(res.best, "predict_proba"):
                proba = res.best.predict_proba(pd.DataFrame([row]))[0].max()
                metric["Confidence"] = f"{proba:.0%}"
            return [Reply("**Example record:**"), Reply(row.to_frame("value"), "table"),
                    Reply(metric, "metric")]

        return Reply(help_text or "Try `train`, `compare`, `drivers`, or `demo`.")

    return respond


def _fmt(pred, task):
    return f"{pred:,.2f}" if task == "regression" else str(pred)


def _confusion_fig(res: TabularResult):
    import matplotlib.pyplot as plt

    cm = res.confusion
    fig, ax = plt.subplots(figsize=(3.4, 3))
    ax.imshow(cm, cmap="Blues")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion — {res.best_name}")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout()
    return fig


def _leaderboard_fig(res: TabularResult):
    import matplotlib.pyplot as plt

    lb = res.leaderboard
    metric_col = [c for c in lb.columns if c != "Algorithm"][0]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(lb["Algorithm"][::-1], lb[metric_col][::-1], color="#a78bfa")
    ax.set_xlabel(metric_col); ax.set_title("Algorithm comparison"); fig.tight_layout()
    return fig


def _importance_fig(imp: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.barh(imp["feature"][::-1], imp["importance"][::-1], color="#38bdf8")
    ax.set_title("Top features"); fig.tight_layout()
    return fig
