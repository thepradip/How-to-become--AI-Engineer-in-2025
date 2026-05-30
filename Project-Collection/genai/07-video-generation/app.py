"""Video Generation — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="Video Generation",
    page_icon="🎬",
    intro="Type a prompt for a short clip (offline procedural demo; real Wan2.2/LTX in README).",
    respond=respond,
    examples=["flowing colors", "ocean waves"],
)
