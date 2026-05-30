"""Conversational Multi-Agent (AutoGen) — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="Conversational Multi-Agent (AutoGen)",
    page_icon="🗣️",
    intro="A Solver and a Critic refine an answer through dialogue. Give a task. (AutoGen/AG2 in README.)",
    respond=respond,
    examples=["write a tagline for a coffee shop", "outline a blog post on RAG"],
)
