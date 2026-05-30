"""Transfer Learning Image Classifier — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="Transfer Learning (ResNet-18)",
    page_icon="🖼️",
    intro="Fine-tunes a pretrained ResNet-18 on your image classes. Type **train** (synthetic demo); see README for real data.",
    respond=respond,
    examples=["train"],
)
