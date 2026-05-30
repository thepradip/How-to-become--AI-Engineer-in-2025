"""Web-agent brain. Uses the shared chat UI — shows navigation trace + answer."""

from __future__ import annotations

from . import webagent as W


def respond(message: str, state: dict):
    from _shared.chat_ui import Reply

    task = message.strip() or "what is the product price?"
    res = W.run(task)
    return [
        Reply("🌐 **Navigation:** " + " → ".join(res["trace"])),
        Reply(f"✅ **Answer:** {res['answer']}"),
        Reply("_(Mock site for an offline demo. Real browsing uses browser-use / Playwright — README.)_", "text"),
    ]
