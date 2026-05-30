"""Finance-agent brain. Uses the shared chat UI."""

from __future__ import annotations

from . import finance as F


def respond(message: str, state: dict):
    from _shared.chat_ui import Reply

    q = message.strip() or "what was the revenue growth?"
    res = F.answer(q)
    tag = "🛡️ no-advice guardrail" if res["blocked"] else "📊 analysis"
    return [Reply(f"`{tag}`", "text"), Reply(res["answer"])]
