"""LSTM forecasting brain. Uses the shared chat UI."""

from __future__ import annotations

import numpy as np

from . import lstm as L

WINDOW = 30


def respond(message: str, state: dict):
    import torch  # noqa: F401

    from _shared.chat_ui import Reply
    from _shared.torch_utils import get_device

    msg = message.strip().lower()
    cfg = state.get("config", {})

    if "train" in msg or "forecast" in msg or msg in {"go", "start"}:
        device = get_device()
        series = L.synthetic_series()
        X, y = L.make_windows(series, WINDOW)
        model = L.LSTMForecaster()
        hist = L.train(model, X, y, epochs=int(cfg.get("epochs", 20)), device=device)
        preds = L.forecast(model, series, WINDOW, steps=40, device=device)
        state["model"] = model
        return [
            Reply(f"Trained an LSTM forecaster on **{device}** (final MSE {hist[-1]:.3f})."),
            Reply(_plot(series, preds), "figure"),
        ]

    return Reply("I forecast a time series with an **LSTM** (multi-step). Try `train`.")


def _plot(series: np.ndarray, preds: np.ndarray):
    import matplotlib.pyplot as plt

    n = len(series)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(range(n), series, label="history", color="#5a6478")
    ax.plot(range(n, n + len(preds)), preds, "--", label="LSTM forecast", color="#38bdf8")
    ax.legend(fontsize=8); ax.set_title("LSTM multi-step forecast"); fig.tight_layout()
    return fig
