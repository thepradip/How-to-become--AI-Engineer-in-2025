"""Machine Translation — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402


def sidebar():
    import streamlit as st

    lang = st.selectbox("Target language", ["fr", "es", "de"],
                        format_func=lambda c: {"fr": "French", "es": "Spanish", "de": "German"}[c])
    return {"target": lang}


run_chat_app(
    title="Machine Translation",
    page_icon="🌍",
    intro="Type English text to translate (offline demo; real NLLB path in README). Pick a language in the sidebar.",
    respond=respond,
    examples=["demo", "hello friend"],
    sidebar=sidebar,
)
