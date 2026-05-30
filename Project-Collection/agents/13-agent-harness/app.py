"""Agent Harness / Orchestration — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="Agent Harness / Orchestration",
    page_icon="🧭",
    intro="A production-style harness: routing → execution → trace + metrics + retries. Give a task. (Claude Agent SDK in README.)",
    respond=respond,
    examples=["12 * (3 + 4)", "upper: hello world", "just echo this"],
)
