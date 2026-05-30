"""Type-safe Tool Agent (PydanticAI) — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="Type-safe Tool Agent (PydanticAI)",
    page_icon="🧰",
    intro="Tools with validated arguments. Ask for weather or a currency conversion. (PydanticAI in README.)",
    respond=respond,
    examples=["what's the weather in Paris?", "convert 100 USD to EUR"],
)
