"""Voice Agent — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="Voice Agent (STT → LLM → TTS)",
    page_icon="🎤",
    intro="Type what you'd 'say'; the agent replies (the text middle of a voice pipeline). Full STT/TTS in README.",
    respond=respond,
    examples=["what's the weather like?", "set a reminder for 9am"],
)
