"""Evals + Guardrails Harness — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="LLM Evals + Guardrails",
    page_icon="🛡️",
    intro="Type **demo** to run the eval harness, or paste text to run PII/safety guardrails on it.",
    respond=respond,
    examples=["demo", "my ssn is 123-45-6789"],
)
