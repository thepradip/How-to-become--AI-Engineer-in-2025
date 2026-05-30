"""Document Intelligence / OCR — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="Document Intelligence / OCR",
    page_icon="📄",
    intro="Paste OCR'd document text to extract fields (invoice #, date, total…). Type **demo** for a sample invoice.",
    respond=respond,
    examples=["demo"],
)
