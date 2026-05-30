"""Customer-segmentation brain (unsupervised). Uses the shared chat UI."""

from __future__ import annotations

import pathlib

import pandas as pd

from . import segmentation as S

DATA = pathlib.Path(__file__).resolve().parent.parent / "data" / "online_retail.csv"


def _load_tx() -> pd.DataFrame:
    if DATA.exists():
        return pd.read_csv(DATA)
    return S.synthetic_transactions()


def _ensure(state):
    if "result" not in state:
        state["result"] = S.segment(S.build_rfm(_load_tx()))
    return state["result"]


def respond(message: str, state: dict):
    from _shared.chat_ui import Reply

    msg = message.strip().lower()

    if "segment" in msg or "train" in msg or msg in {"go", "start"}:
        res = state["result"] = S.segment(S.build_rfm(_load_tx()))
        return [
            Reply(f"Compared clustering algorithms. Best: **{res.best_algo}** "
                  f"with **{res.best_k} segments**."),
            Reply(res.leaderboard.head(8), "table"),
            Reply(res.profiles, "table"),
            Reply(_scatter(res), "figure"),
            Reply("Try `profiles` or `demo`.", "text"),
        ]
    if "profile" in msg:
        res = _ensure(state)
        return [Reply("**Segment profiles** (mean RFM + size):"), Reply(res.profiles, "table")]
    if "demo" in msg or "assign" in msg:
        res = _ensure(state)
        row = res.rfm.iloc[0]
        seg = int(row["segment"])
        label = res.profiles.loc[res.profiles["segment"] == seg, "label"]
        return [Reply("**Example customer RFM:**"),
                Reply(row[S.RFM_COLS].to_frame("value"), "table"),
                Reply({"Segment": seg, "Label": label.iloc[0] if len(label) else "—"}, "metric")]

    return Reply(
        "I segment customers by **RFM** (Recency/Frequency/Monetary) comparing KMeans, "
        "Agglomerative & DBSCAN. Try `segment`, `profiles`, or `demo`.")


def _scatter(res: S.SegmentResult):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.5))
    sc = ax.scatter(res.rfm["recency"], res.rfm["monetary"],
                    c=res.rfm["segment"], cmap="viridis", s=18, alpha=0.7)
    ax.set_xlabel("Recency (days)"); ax.set_ylabel("Monetary"); ax.set_title("Customer segments")
    fig.colorbar(sc, ax=ax, label="segment"); fig.tight_layout()
    return fig
