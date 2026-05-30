"""Web / Browser Agent — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="Web / Browser Agent",
    page_icon="🌐",
    intro="Asks the agent to navigate a (mock) site and extract info. Real browsing = browser-use/Playwright (README).",
    respond=respond,
    examples=["what is the product price?", "when was the company founded?"],
)
