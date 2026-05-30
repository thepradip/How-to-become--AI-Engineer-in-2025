"""Research Agent — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="Research Agent",
    page_icon="🔬",
    intro="Ask a research question; I search sources and synthesize a cited note. (CrewAI in README.)",
    respond=respond,
    examples=["how is RAG evaluated?", "what frameworks build agents?"],
)
