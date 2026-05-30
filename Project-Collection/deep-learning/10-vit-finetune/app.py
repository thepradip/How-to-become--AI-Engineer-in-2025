"""ViT Fine-Tune (+ W&B) — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="Vision Transformer Fine-Tune",
    page_icon="🧠",
    intro="Fine-tunes ViT-B/16 with optional W&B logging/sweeps. Type **train** (synthetic demo); README has real data + sweeps.",
    respond=respond,
    examples=["train"],
)
