"""SQL agent brain. Uses the shared chat UI — ask a question, see SQL + results."""

from __future__ import annotations

from . import sqlagent as S


def respond(message: str, state: dict):
    import pandas as pd

    from _shared.chat_ui import Reply

    q = message.strip() or "how many employees are in Engineering?"
    if q.lower() in {"demo", "help", "schema"}:
        return Reply(f"Table: `{S.SCHEMA}`. Ask e.g. *average salary in Sales* or *who is the top earner?*")
    res = S.answer(q)
    parts = [Reply(f"**Generated SQL:**"), Reply(res["sql"], "code", {"language": "sql"})]
    if res["rows"]:
        parts.append(Reply(pd.DataFrame(res["rows"]), "table"))
    else:
        parts.append(Reply("No rows.", "text"))
    return parts
