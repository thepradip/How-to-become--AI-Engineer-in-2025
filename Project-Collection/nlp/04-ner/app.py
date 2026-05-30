"""Named Entity Recognition — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="Named Entity Recognition",
    page_icon="🔖",
    intro="Paste text and I'll extract entities (people, orgs, money, dates, emails…). Type **demo** for an example.",
    respond=respond,
    examples=["demo"],
)
