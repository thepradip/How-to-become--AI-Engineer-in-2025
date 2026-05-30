"""A2A Multi-Framework Agents — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="A2A — Agents Across Frameworks",
    page_icon="🔌",
    intro="A LangGraph math-bot and a CrewAI writer-bot collaborate via the A2A protocol. Give a math expression.",
    respond=respond,
    examples=["6 * 7", "(10 + 2) * 5"],
)
