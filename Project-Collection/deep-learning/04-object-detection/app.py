"""YOLO Object Detection — UI (shared chat + upload). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="YOLO11 Object Detection",
    page_icon="🎯",
    intro="Detects objects in images with YOLO11. **Upload an image** in the sidebar, or type **demo**.",
    respond=respond,
    examples=["demo"],
    accepts_files=True,
)
