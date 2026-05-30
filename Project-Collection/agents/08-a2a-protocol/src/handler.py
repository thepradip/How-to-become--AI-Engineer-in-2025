"""A2A brain. Uses the shared chat UI — shows the agent-to-agent message exchange."""

from __future__ import annotations

import re

from . import a2a as A


def respond(message: str, state: dict):
    import pandas as pd

    from _shared.chat_ui import Reply

    m = re.search(r"[-+*/%().\d\s]{3,}", message)
    expr = m.group().strip() if m else "6 * 7"
    res = A.run(expr)
    rows = [{"from": t["sender"], "to": t["recipient"], "parts": ", ".join(t["parts"])}
            for t in res["transcript"]]
    return [
        Reply("**Agent cards (A2A discovery):**"),
        Reply(pd.DataFrame(res["cards"]), "table"),
        Reply("**A2A message exchange** (LangGraph math-bot ↔ CrewAI writer-bot):"),
        Reply(pd.DataFrame(rows), "table"),
        Reply(f"✅ **Final:** {res['final']}"),
    ]
