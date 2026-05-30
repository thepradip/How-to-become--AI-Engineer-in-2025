"""Recommender brain. Uses the shared chat UI."""

from __future__ import annotations

import pathlib

import pandas as pd

from . import recommender as R

DATA = pathlib.Path(__file__).resolve().parent.parent / "data" / "ratings.csv"


def _load() -> pd.DataFrame:
    if DATA.exists():
        return pd.read_csv(DATA)
    return R.synthetic_ratings()


def _ensure(state):
    if "result" not in state:
        state["result"] = R.evaluate(_load())
    return state["result"]


def respond(message: str, state: dict):
    from _shared.chat_ui import Reply

    msg = message.strip().lower()
    if "train" in msg or "evaluate" in msg or "compare" in msg or msg in {"go", "start"}:
        res = state["result"] = R.evaluate(_load())
        return [
            Reply(f"Compared methods by held-out RMSE. Best: **{res.best_method}**."),
            Reply(res.leaderboard, "table"),
            Reply("Try `recommend` for top picks.", "text"),
        ]
    if "recommend" in msg or "top" in msg or "demo" in msg:
        res = _ensure(state)
        recs = R.recommend(res, user_idx=0, k=5)
        return [Reply("**Top-5 recommendations for user 0** (SVD score):"), Reply(recs, "table")]

    return Reply("I recommend movies via collaborative filtering, comparing baselines vs **SVD** "
                 "matrix factorization. Try `train` then `recommend`.")
