"""Shared utilities reused across every Project-Collection project.

Importing this package from a project requires the Project-Collection root to be
on ``sys.path``. Each ``app.py`` / test does this with a 3-line bootstrap::

    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

Modules:
  * ``chat_ui``  – the one shared Streamlit chat interface (see chat_ui.py).
  * ``data``     – real-dataset download helpers (OpenML / sklearn / URL / yfinance).
"""

from .chat_ui import Reply, run_chat_app  # noqa: F401

__all__ = ["Reply", "run_chat_app"]
