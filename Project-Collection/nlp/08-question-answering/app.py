"""Extractive Question Answering — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="Extractive Question Answering",
    page_icon="❓",
    intro="Ask a question about the built-in passage, or set your own with `context: ...`. Type **demo** to see it.",
    respond=respond,
    examples=["demo", "what does RAG reduce?", "what is the dominant pattern in 2026?"],
)
