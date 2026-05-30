"""Stock Movement Predictor — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="Stock Movement Predictor",
    page_icon="📊",
    intro="Predicts next-day direction and **backtests vs buy-and-hold** (educational, not advice). Type **analyze**.",
    respond=respond,
    examples=["analyze", "compare"],
)
