"""Prompt Playground — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="Prompt Engineering Playground",
    page_icon="🎛️",
    intro="Type a task; I'll answer using the best available LLM (or an offline mock). Prefix `cot:` for chain-of-thought; `providers` to see backends.",
    respond=respond,
    examples=["explain RAG simply", "cot: 17 * 24 = ?", "providers"],
)
