"""Traffic Sign Recognition — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="Traffic Sign Recognition (GTSRB)",
    page_icon="🚦",
    intro="A CNN classifying 43 traffic-sign types. Type **train** (synthetic demo); README shows real GTSRB.",
    respond=respond,
    examples=["train"],
)
