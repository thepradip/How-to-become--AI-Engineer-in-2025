"""Face Verification — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="Face Verification (embeddings)",
    page_icon="🧑",
    intro="Learns face embeddings and verifies same-vs-different person by cosine similarity. Type **train**, then **verify**.",
    respond=respond,
    examples=["train", "verify"],
)
