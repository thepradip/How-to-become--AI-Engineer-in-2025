"""Breast Cancer Diagnosis — UI (shared chat). Run: streamlit run app.py"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[2]))  # Project-Collection root → _shared
sys.path.insert(0, str(HERE.parent))      # project root → src

from _shared.chat_ui import run_chat_app  # noqa: E402
from src.handler import respond  # noqa: E402

run_chat_app(
    title="Breast Cancer Diagnosis",
    page_icon="🩺",
    intro="Classifies tumours as **malignant**/**benign** from biopsy data (real UCI dataset). Type **train** to begin.",
    respond=respond,
    examples=["train", "compare", "drivers", "demo"],
)
