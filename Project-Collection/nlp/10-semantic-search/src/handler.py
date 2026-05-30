"""Semantic-search brain. Uses the shared chat UI — paste a résumé, get matching jobs."""

from __future__ import annotations

from . import search as S

_SAMPLE_RESUME = ("Experienced Python developer skilled in PyTorch, building RAG pipelines and "
                  "fine-tuning LLMs, with AWS deployment experience.")


def respond(message: str, state: dict):
    from _shared.chat_ui import Reply

    query = _SAMPLE_RESUME if message.strip().lower() in {"demo", "", "help"} else message
    ranked = S.rank(query, S.SAMPLE_JOBS)
    return [
        Reply("**Best-matching jobs** for that profile (TF-IDF similarity):"),
        Reply(ranked, "table"),
        Reply("_Paste your own skills/résumé to re-rank. Real semantic match uses sentence-transformers (README)._", "text"),
    ]
