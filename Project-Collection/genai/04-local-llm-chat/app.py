"""Local LLM Chat — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402


def sidebar():
    import streamlit as st

    return {"model": st.text_input("Ollama/vLLM model", "qwen3:4b")}


run_chat_app(
    title="Local LLM Chat (Ollama / vLLM)",
    page_icon="💬",
    intro="Chat with a locally-served model. Run `ollama run qwen3:4b` first, or it uses an offline mock. `reset` to clear.",
    respond=respond,
    examples=["hello!", "providers"],
    sidebar=sidebar,
)
