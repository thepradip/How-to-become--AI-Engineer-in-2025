"""Conversational multi-agent brain. Uses the shared chat UI — shows the dialogue."""

from __future__ import annotations

from . import conversation as Co


def respond(message: str, state: dict):
    from _shared.chat_ui import Reply

    task = message.strip() or "write a tagline for a coffee shop"
    d = Co.run(task)
    lines = [f"**{t['agent']}:** {t['message']}" for t in d.transcript]
    return [Reply("\n\n".join(lines)), Reply(f"✅ **Final:** {d.final}")]
