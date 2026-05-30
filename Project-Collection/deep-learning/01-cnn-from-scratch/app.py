"""CNN from Scratch (Fashion-MNIST) — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402


def sidebar():
    import streamlit as st

    return {"epochs": st.slider("Epochs", 1, 5, 1),
            "max_batches": st.slider("Train batches (demo speed)", 20, 200, 60, 20)}


run_chat_app(
    title="CNN from Scratch — Fashion-MNIST",
    page_icon="👕",
    intro="Trains a convolutional net from scratch in PyTorch. Type **train**, then **demo**.",
    respond=respond,
    examples=["train", "demo"],
    sidebar=sidebar,
)
