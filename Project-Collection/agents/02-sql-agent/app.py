"""SQL Agent — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="SQL Agent (NL → SQL)",
    page_icon="🗃️",
    intro="Ask questions about an employees table; I write SQL and run it. Type **schema** to see columns.",
    respond=respond,
    examples=["how many employees in Engineering?", "average salary in Sales", "who is the top earner?"],
)
