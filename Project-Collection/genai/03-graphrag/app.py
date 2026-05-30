"""GraphRAG — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="GraphRAG — Knowledge-Graph Retrieval",
    page_icon="🕸️",
    intro="Builds a knowledge graph from text and answers relationship questions. Type **graph**, then ask.",
    respond=respond,
    examples=["graph", "what is Carol connected to?", "who founded Acme?"],
)
