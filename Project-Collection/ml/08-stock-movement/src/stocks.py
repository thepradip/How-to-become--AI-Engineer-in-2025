"""Predict next-day stock direction from technical features, then backtest it.

Reality check
-------------
Markets are near-efficient: predicting daily direction is *hard* and a model that
looks great in-sample often loses money live. This project teaches the **right way to
evaluate** a trading idea — strict chronological split (no shuffling → no look-ahead
leakage) and a **backtest vs. buy-and-hold** — rather than promising profits.

Algorithms compared: Logistic Regression, Random Forest, Gradient Boosting.
Real data: daily prices via **yfinance** (download_data.py). Offline → a synthetic
geometric-Brownian-motion price series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

TARGET = "up"


@dataclass
class StockResult:
    leaderboard: pd.DataFrame
    best_model: str
    backtest: pd.DataFrame   # dates, strategy & buy-hold cumulative returns
    stats: dict


def synthetic_prices(n: int = 1000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2021-01-01", periods=n, freq="B")
    rets = rng.normal(0.0004, 0.012, n)        # slight drift + daily vol
    price = 100 * np.cumprod(1 + rets)
    return pd.DataFrame({"date": dates, "close": price.round(2)})


def _rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).fillna(50)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().sort_values("date").reset_index(drop=True)
    df["ret1"] = df["close"].pct_change()
    df["ret5"] = df["close"].pct_change(5)
    df["sma_ratio"] = df["close"] / df["close"].rolling(20).mean()
    df["vol"] = df["ret1"].rolling(10).std()
    df["rsi"] = _rsi(df["close"])
    df[TARGET] = (df["close"].shift(-1) > df["close"]).astype(int)  # next-day up?
    return df.dropna().reset_index(drop=True)


FEATS = ["ret1", "ret5", "sma_ratio", "vol", "rsi"]


def train_and_backtest(df: pd.DataFrame, split: float = 0.7) -> StockResult:
    feat = make_features(df)
    n = len(feat)
    cut = int(n * split)
    tr, te = feat.iloc[:cut], feat.iloc[cut:]           # chronological — no shuffle
    X_tr, y_tr = tr[FEATS], tr[TARGET]
    X_te, y_te = te[FEATS], te[TARGET]

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200, n_jobs=2, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    rows, preds = [], {}
    for name, est in models.items():
        est.fit(X_tr, y_tr)
        p = est.predict(X_te)
        rows.append({"model": name, "accuracy": round(float(accuracy_score(y_te, p)), 3)})
        preds[name] = p
    lb = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
    best = lb.iloc[0]["model"]

    # Backtest the best: be long next day when it predicts "up", else flat.
    signal = preds[best]
    next_ret = te["close"].pct_change().shift(-1).fillna(0).values
    strat = np.cumprod(1 + signal * next_ret)
    hold = np.cumprod(1 + next_ret)
    bt = pd.DataFrame({"date": te["date"].values, "strategy": strat, "buy_hold": hold})
    stats = {
        "Best model": best,
        "Strategy return": f"{(strat[-1] - 1) * 100:.1f}%",
        "Buy & hold return": f"{(hold[-1] - 1) * 100:.1f}%",
        "Test accuracy": lb.iloc[0]["accuracy"],
    }
    return StockResult(leaderboard=lb, best_model=best, backtest=bt, stats=stats)
