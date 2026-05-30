"""House Price Estimator — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="House Price Estimator",
    page_icon="🏠",
    intro="Predicts house sale prices (Ames dataset), comparing several regressors + an ensemble. Type **train**.",
    respond=respond,
    examples=["train", "compare", "drivers", "demo"],
)
