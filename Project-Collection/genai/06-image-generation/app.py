"""Image Generation — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="Image Generation",
    page_icon="🎨",
    intro="Type a prompt for an image (offline procedural demo; real SD3.5/FLUX in README).",
    respond=respond,
    examples=["a calm sunset over mountains", "neon city at night"],
)
