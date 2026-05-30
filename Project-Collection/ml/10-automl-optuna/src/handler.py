"""AutoML brain. Uses the shared chat UI."""

from __future__ import annotations

import pandas as pd

from . import automl as A


def _ensure(state, n_trials=30):
    if "result" not in state:
        state["result"] = A.run_optuna(n_trials=n_trials)
    return state["result"]


def respond(message: str, state: dict):
    from _shared.chat_ui import Reply

    msg = message.strip().lower()
    n_trials = state.get("config", {}).get("n_trials", 30)

    if "tune" in msg or "train" in msg or "optimize" in msg or msg in {"go", "start"}:
        res = state["result"] = A.run_optuna(n_trials=n_trials)
        gain = res.tuned_test_auc - res.default_test_auc
        return [
            Reply(f"Ran **Optuna** for {n_trials} trials. Best CV ROC-AUC: **{res.best_cv_score}**."),
            Reply({"Default test AUC": res.default_test_auc,
                   "Tuned test AUC": res.tuned_test_auc,
                   "Gain": f"{gain:+.4f}"}, "metric"),
            Reply("**Best hyperparameters:**"),
            Reply(pd.DataFrame([res.best_params]).T.rename(columns={0: "value"}), "table"),
            Reply(_history_fig(res), "figure"),
        ]
    if "params" in msg or "best" in msg:
        res = _ensure(state, n_trials)
        return [Reply("**Best hyperparameters:**"),
                Reply(pd.DataFrame([res.best_params]).T.rename(columns={0: "value"}), "table")]

    return Reply("I tune a model's hyperparameters with **Optuna** and compare tuned vs default "
                 "(optional MLflow logging). Try `tune`.")


def _history_fig(res: A.AutoMLResult):
    import matplotlib.pyplot as plt

    h = res.history
    best_so_far = h["cv_roc_auc"].cummax()
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    ax.plot(h["trial"], h["cv_roc_auc"], "o", ms=3, alpha=0.5, label="trial")
    ax.plot(h["trial"], best_so_far, "-", color="#34d399", label="best so far")
    ax.set_xlabel("trial"); ax.set_ylabel("CV ROC-AUC"); ax.legend(fontsize=8)
    ax.set_title("Optuna optimization history"); fig.tight_layout()
    return fig
