"""Healthcare Agent — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="Healthcare Agent (guardrailed)",
    page_icon="⚕️",
    intro="General wellness Q&A with safety guardrails — it won't diagnose or prescribe. Educational only.",
    respond=respond,
    examples=["how can I stay healthy?", "what helps with a mild cold?"],
)
