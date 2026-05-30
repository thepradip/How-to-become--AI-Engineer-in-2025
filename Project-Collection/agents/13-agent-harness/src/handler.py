"""Agent-harness brain. Uses the shared chat UI — shows routing, trace, and metrics."""

from __future__ import annotations

from . import harness as Ha


def respond(message: str, state: dict):
    import pandas as pd

    from _shared.chat_ui import Reply

    task = message.strip() or "12 * (3 + 4)"
    res = Ha.run(task)
    return [
        Reply(f"🧭 **Routed to skill:** `{res.skill}`"),
        Reply("**Execution trace:**"),
        Reply(pd.DataFrame(res.trace), "table"),
        Reply(res.metrics, "metric"),
        Reply(f"✅ **Result:** {res.result}"),
    ]
