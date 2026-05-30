"""LoRA/QLoRA LLM Fine-Tune (Unsloth) — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="LLM Fine-Tune — LoRA/QLoRA (Unsloth)",
    page_icon="🦥",
    intro="Prepares instruction data and fine-tunes an LLM with Unsloth on a 16 GB GPU. Type **data** to preview the format.",
    respond=respond,
    examples=["data"],
)
