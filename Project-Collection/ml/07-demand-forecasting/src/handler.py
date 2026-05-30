"""Demand-forecasting brain. Uses the shared chat UI."""

from __future__ import annotations

import pathlib

import pandas as pd

from . import forecasting as F

DATA = pathlib.Path(__file__).resolve().parent.parent / "data" / "store_sales.csv"


def _load() -> pd.DataFrame:
    if DATA.exists():
        return pd.read_csv(DATA)
    return F.synthetic_series()


def _ensure(state):
    if "result" not in state:
        state["result"] = F.forecast(_load())
    return state["result"]


def respond(message: str, state: dict):
    from _shared.chat_ui import Reply

    msg = message.strip().lower()
    if "forecast" in msg or "train" in msg or msg in {"go", "start"}:
        res = state["result"] = F.forecast(_load())
        return [
            Reply(f"Compared models on a {res.horizon}-day holdout. Best: **{res.best_model}**."),
            Reply(res.leaderboard, "table"),
            Reply(_plot(res), "figure"),
            Reply("Try `compare` or `plot`.", "text"),
        ]
    if "compare" in msg or "leaderboard" in msg:
        res = _ensure(state)
        return [Reply("**Model comparison** (lower is better):"), Reply(res.leaderboard, "table")]
    if "plot" in msg:
        return [Reply("**Actual vs. forecast** on the holdout:"), Reply(_plot(_ensure(state)), "figure")]

    return Reply("I forecast **daily sales** comparing a seasonal baseline vs ML models. "
                 "Try `forecast`, `compare`, or `plot`.")


def _plot(res: F.ForecastResult):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.2))
    hist = res.history.tail(120)
    ax.plot(pd.to_datetime(hist[F.DATE]), hist[F.TARGET], label="history", color="#5a6478")
    ax.plot(pd.to_datetime(res.forecast[F.DATE]), res.forecast["actual"], label="actual", color="#34d399")
    ax.plot(pd.to_datetime(res.forecast[F.DATE]), res.forecast["predicted"], "--",
            label=f"forecast ({res.best_model})", color="#38bdf8")
    ax.legend(fontsize=8); ax.set_title("Demand forecast"); fig.autofmt_xdate(); fig.tight_layout()
    return fig
