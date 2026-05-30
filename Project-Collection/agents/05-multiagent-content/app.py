"""Multi-Agent Content Team — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="Multi-Agent Content Team",
    page_icon="👥",
    intro="Give a topic; a researcher → writer → editor crew produces an article. (CrewAI in README.)",
    respond=respond,
    examples=["RAG", "AI agents", "fine-tuning"],
)
