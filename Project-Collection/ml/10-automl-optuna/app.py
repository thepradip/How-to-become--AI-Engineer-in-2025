"""AutoML + Optuna — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402


def sidebar():
    import streamlit as st

    return {"n_trials": st.slider("Optuna trials", 10, 100, 30, 10)}


run_chat_app(
    title="AutoML — Hyperparameter Tuning (Optuna)",
    page_icon="⚙️",
    intro="Tunes a model with **Optuna** and compares tuned vs default. Set trials in the sidebar, type **tune**.",
    respond=respond,
    examples=["tune", "best params"],
    sidebar=sidebar,
)
