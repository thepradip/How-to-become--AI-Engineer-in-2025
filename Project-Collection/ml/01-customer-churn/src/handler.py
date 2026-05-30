"""Chat handler — the project "brain" the shared chat UI calls.

Contract (see _shared/chat_ui.py): ``respond(message, state) -> Reply | list[Reply]``.
``state`` caches the trained models so we only train once per session.

Commands:
  * ``train``              – train all algorithms + the ensemble, show the leaderboard
  * ``compare``            – show the algorithm leaderboard again (table + chart)
  * ``drivers`` / ``why``  – top churn drivers (from the tree ensemble)
  * ``demo`` / ``score``   – score an example customer with the best model
"""

from __future__ import annotations

import pathlib

import pandas as pd

from . import model as M

DATA_CSV = pathlib.Path(__file__).resolve().parent.parent / "data" / "telco_churn.csv"


def _load_df() -> pd.DataFrame:
    if DATA_CSV.exists():
        return pd.read_csv(DATA_CSV)
    return M.synthetic_churn(800)


def _ensure_trained(state: dict) -> M.TrainResult:
    if "result" not in state:
        state["result"] = M.train(_load_df())
    return state["result"]


def respond(message: str, state: dict):
    from _shared.chat_ui import Reply

    msg = message.strip().lower()

    if "train" in msg or msg in {"start", "go"}:
        res = state["result"] = M.train(_load_df())
        return [
            Reply(f"Trained **{len(res.pipelines)} algorithms** (incl. a stacking ensemble). "
                  f"Best model: **{res.best_name}**."),
            Reply(res.leaderboard, "table"),
            Reply(res.metrics, "metric"),
            Reply(_confusion_fig(res), "figure"),
            Reply("Try `compare`, `drivers`, or `demo`.", "text"),
        ]

    if "compare" in msg or "leaderboard" in msg:
        res = _ensure_trained(state)
        return [Reply("**Algorithm comparison** (held-out ROC-AUC):"),
                Reply(res.leaderboard, "table"), Reply(_leaderboard_fig(res), "figure")]

    if "driver" in msg or "why" in msg or "import" in msg:
        res = _ensure_trained(state)
        return [Reply("**Top churn drivers** (gradient-boosting gain importance):"),
                Reply(res.importances, "table"), Reply(_importance_fig(res.importances), "figure")]

    if "demo" in msg or "score" in msg:
        res = _ensure_trained(state)
        prob = M.score_customer(res, res.sample_row)
        verdict = "🔴 likely to churn" if prob >= 0.5 else "🟢 likely to stay"
        return [
            Reply(f"**Example customer** scored by the best model ({res.best_name}):"),
            Reply(res.sample_row.to_frame("value"), "table"),
            Reply({"Churn probability": f"{prob:.0%}", "Verdict": verdict}, "metric"),
        ]

    return Reply(
        "I predict telco **customer churn** using several algorithms + an ensemble. Try:\n"
        "- `train` — train & compare all models\n"
        "- `compare` — the algorithm leaderboard\n"
        "- `drivers` — what pushes customers to leave\n"
        "- `demo` — score an example customer"
    )


def _confusion_fig(res: M.TrainResult):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(3.2, 3))
    cm = res.confusion
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], ["Stay", "Churn"]); ax.set_yticks([0, 1], ["Stay", "Churn"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion — {res.best_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout()
    return fig


def _leaderboard_fig(res: M.TrainResult):
    import matplotlib.pyplot as plt

    lb = res.leaderboard
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.barh(lb["Algorithm"][::-1], lb["ROC-AUC"][::-1], color="#a78bfa")
    ax.set_xlim(0.5, 1.0); ax.set_xlabel("ROC-AUC"); ax.set_title("Algorithm comparison")
    fig.tight_layout()
    return fig


def _importance_fig(imp: pd.DataFrame):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.barh(imp["feature"][::-1], imp["importance"][::-1], color="#38bdf8")
    ax.set_title("Top churn drivers"); fig.tight_layout()
    return fig
