"""U-Net Image Segmentation — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="U-Net Image Segmentation",
    page_icon="🧬",
    intro="Trains a U-Net to label every pixel (synthetic shape task; swap in real masks). Type **train**, then **demo**.",
    respond=respond,
    examples=["train", "demo"],
)
