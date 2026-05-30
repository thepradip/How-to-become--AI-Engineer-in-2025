"""Sentiment Analysis — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="Sentiment Analysis",
    page_icon="🎭",
    intro="Type **train**, then paste a review and I'll judge positive/negative.",
    respond=respond,
    examples=["train", "an absolute masterpiece, loved it", "boring waste of time"],
)
