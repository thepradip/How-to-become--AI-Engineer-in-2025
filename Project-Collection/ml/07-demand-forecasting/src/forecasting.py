"""Demand forecasting with feature-based ML models vs. a seasonal baseline.

Business problem
----------------
Retailers must order stock ahead of demand. Forecasting daily/weekly sales reduces
stock-outs and waste. We frame forecasting as supervised regression on **lag** and
**calendar** features and compare several models against a **seasonal-naive baseline**
(a strong, honest benchmark — beating it is the bar).

Algorithms compared: Seasonal-naive, Linear Regression, Random Forest, Gradient
Boosting. (LightGBM / Prophet are documented extensions; optional imports.)

Real data: a daily store-sales series (download_data.py). Offline → a synthetic series
with trend + weekly + yearly seasonality.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATE, TARGET = "date", "sales"


@dataclass
class ForecastResult:
    history: pd.DataFrame      # full series
    forecast: pd.DataFrame     # holdout dates: actual + best-model prediction
    leaderboard: pd.DataFrame  # model -> MAE, RMSE, MAPE
    best_model: str
    horizon: int


def synthetic_series(periods: int = 730, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(periods)
    dates = pd.date_range("2023-01-01", periods=periods, freq="D")
    trend = 50 + 0.05 * t
    weekly = 10 * np.sin(2 * np.pi * t / 7)
    yearly = 20 * np.sin(2 * np.pi * t / 365.25)
    sales = (trend + weekly + yearly + rng.normal(0, 4, periods)).round(1).clip(min=0)
    return pd.DataFrame({DATE: dates, TARGET: sales})


def _features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[DATE] = pd.to_datetime(df[DATE])
    df = df.sort_values(DATE).reset_index(drop=True)
    df["dayofweek"] = df[DATE].dt.dayofweek
    df["month"] = df[DATE].dt.month
    for lag in (1, 7, 14, 28):
        df[f"lag_{lag}"] = df[TARGET].shift(lag)
    df["roll_7"] = df[TARGET].shift(1).rolling(7).mean()
    df["roll_28"] = df[TARGET].shift(1).rolling(28).mean()
    return df.dropna().reset_index(drop=True)


def _metrics(y_true, y_pred) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100)
    return {"MAE": round(mae, 2), "RMSE": round(rmse, 2), "MAPE%": round(mape, 1)}


def forecast(df: pd.DataFrame, horizon: int = 28) -> ForecastResult:
    """Train on all but the last ``horizon`` days; compare models on that holdout."""
    feat = _features(df)
    feat_cols = [c for c in feat.columns if c not in (DATE, TARGET)]
    train, test = feat.iloc[:-horizon], feat.iloc[-horizon:]
    X_tr, y_tr = train[feat_cols], train[TARGET]
    X_te, y_te = test[feat_cols], test[TARGET]

    rows, preds = [], {}
    # Seasonal-naive: value from 7 days earlier.
    naive = test["lag_7"].values
    rows.append({"model": "Seasonal-naive", **_metrics(y_te.values, naive)})
    preds["Seasonal-naive"] = naive

    for name, est in {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, n_jobs=2, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }.items():
        est.fit(X_tr, y_tr)
        p = est.predict(X_te)
        rows.append({"model": name, **_metrics(y_te.values, p)})
        preds[name] = p

    lb = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    best = lb.iloc[0]["model"]
    fc = pd.DataFrame({DATE: test[DATE].values, "actual": y_te.values, "predicted": preds[best]})
    return ForecastResult(history=feat[[DATE, TARGET]], forecast=fc,
                          leaderboard=lb, best_model=best, horizon=horizon)
