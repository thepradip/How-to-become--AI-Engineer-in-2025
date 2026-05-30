"""ReAct Agent From Scratch — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="ReAct Agent (from scratch)",
    page_icon="🧠",
    intro="Watch the Thought → Action → Observation loop. Try a math expression or 'what is RAG?'.",
    respond=respond,
    examples=["what is RAG?", "12 * (3 + 4)"],
)
