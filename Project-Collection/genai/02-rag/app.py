"""RAG — Chat With Your Documents — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="RAG — Chat With Your Docs",
    page_icon="📚",
    intro="Ask questions about the sample docs (with citations), or load your own via `docs: <text>`. Type **demo**.",
    respond=respond,
    examples=["demo", "how long do refunds take?", "what does the premium plan include?"],
)
