"""Speech Emotion Recognition — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="Speech Emotion Recognition",
    page_icon="🎙️",
    intro="Classifies emotion from audio features (MLP). Type **train**, then **demo**. Real audio via librosa + RAVDESS (README).",
    respond=respond,
    examples=["train", "demo"],
)
