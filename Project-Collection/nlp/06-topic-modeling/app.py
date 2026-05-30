"""Topic Modeling — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="Topic Modeling (NMF & LDA)",
    page_icon="🗂️",
    intro="Type **discover** to find topics (NMF + LDA), or paste a document to match it to a topic.",
    respond=respond,
    examples=["discover", "nmf", "lda"],
)
