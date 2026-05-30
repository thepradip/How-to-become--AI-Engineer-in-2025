"""Text Summarizer — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402


def sidebar():
    import streamlit as st

    return {"n_sentences": st.slider("Summary sentences", 1, 5, 2)}


run_chat_app(
    title="Text Summarizer",
    page_icon="📝",
    intro="Paste a passage and I'll extract a short summary. Type **demo** for an example.",
    respond=respond,
    examples=["demo"],
    sidebar=sidebar,
)
