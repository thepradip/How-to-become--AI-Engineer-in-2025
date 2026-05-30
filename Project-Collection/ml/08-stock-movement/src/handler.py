"""Stock-movement brain. Uses the shared chat UI."""

from __future__ import annotations

import pathlib

import pandas as pd

from . import stocks as K

DATA = pathlib.Path(__file__).resolve().parent.parent / "data" / "prices.csv"


def _load() -> pd.DataFrame:
    if DATA.exists():
        return pd.read_csv(DATA)
    return K.synthetic_prices()


def _ensure(state):
    if "result" not in state:
        state["result"] = K.train_and_backtest(_load())
    return state["result"]


def respond(message: str, state: dict):
    from _shared.chat_ui import Reply

    msg = message.strip().lower()
    if "analyze" in msg or "train" in msg or "backtest" in msg or msg in {"go", "start"}:
        res = state["result"] = K.train_and_backtest(_load())
        return [
            Reply(f"Trained 3 models (chronological split). Best: **{res.best_model}**. "
                  "Remember: daily direction is hard — compare to buy-and-hold honestly."),
            Reply(res.leaderboard, "table"),
            Reply(res.stats, "metric"),
            Reply(_plot(res), "figure"),
        ]
    if "compare" in msg:
        return [Reply("**Model accuracy:**"), Reply(_ensure(state).leaderboard, "table")]

    return Reply("I predict **next-day price direction** and backtest it vs buy-and-hold. "
                 "Try `analyze` or `compare`. (Educational — not investment advice.)")


def _plot(res: K.StockResult):
    import matplotlib.pyplot as plt

    bt = res.backtest
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(pd.to_datetime(bt["date"]), bt["strategy"], label="model strategy", color="#38bdf8")
    ax.plot(pd.to_datetime(bt["date"]), bt["buy_hold"], label="buy & hold", color="#5a6478")
    ax.set_ylabel("growth of $1"); ax.legend(fontsize=8); ax.set_title("Backtest")
    fig.autofmt_xdate(); fig.tight_layout()
    return fig
