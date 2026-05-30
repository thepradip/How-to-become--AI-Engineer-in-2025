"""Pneumonia X-ray Classifier + Grad-CAM — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="Pneumonia X-ray + Grad-CAM",
    page_icon="🫁",
    intro="Detects pneumonia in chest X-rays and explains *where* it looked via Grad-CAM. Type **train**, then **explain**.",
    respond=respond,
    examples=["train", "explain"],
)
