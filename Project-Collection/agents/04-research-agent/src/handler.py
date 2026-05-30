"""Research-agent brain. Uses the shared chat UI."""

from __future__ import annotations

from . import research as R


def respond(message: str, state: dict):
    import pandas as pd

    from _shared.chat_ui import Reply

    q = message.strip() or "how is RAG evaluated?"
    res = R.run(q)
    parts = [Reply(f"**Research note** for: _{q}_"), Reply(res["summary"])]
    if res["sources"]:
        parts.append(Reply(pd.DataFrame(res["sources"])[["title", "score"]], "table"))
    parts.append(Reply("_(Local corpus demo. Real multi-tool research uses CrewAI + web search — README.)_", "text"))
    return parts
