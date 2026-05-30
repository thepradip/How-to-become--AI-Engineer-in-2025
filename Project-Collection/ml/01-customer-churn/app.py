"""Customer Churn Predictor — UI.

This file is intentionally tiny: all UI is the shared chat component, all logic is
in src/. Run with:  streamlit run app.py
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))  # Project-Collection root → import _shared
sys.path.insert(0, str(HERE.parent))      # project root → import src package

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="Customer Churn Predictor",
    page_icon="📉",
    intro="Predicts which telco customers will leave — trains several algorithms + an ensemble. Type **train** to start, then **compare**, **drivers** or **demo**.",
    respond=respond,
    examples=["train", "compare", "drivers", "demo"],
)
