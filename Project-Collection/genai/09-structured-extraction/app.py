"""Structured Extraction + Guardrails — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="Structured Extraction + Guardrails",
    page_icon="🧱",
    intro="Paste a message; I extract a validated Lead (name/email/company/budget). Type **demo**.",
    respond=respond,
    examples=["demo", "I'm Tom from Initech, tom@initech.com, budget 12000"],
)
