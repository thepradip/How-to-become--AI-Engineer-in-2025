"""Healthcare-agent brain. Uses the shared chat UI."""

from __future__ import annotations

from . import health as H


def respond(message: str, state: dict):
    from _shared.chat_ui import Reply

    q = message.strip() or "how can I stay healthy?"
    res = H.answer(q)
    tag = "🛡️ guardrail triggered" if res["blocked"] else "ℹ️ general info"
    return [Reply(f"`{tag}`", "text"), Reply(res["answer"])]
