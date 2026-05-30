"""Multi-agent content-team brain. Uses the shared chat UI — shows each role's output."""

from __future__ import annotations

from . import crew as C


def respond(message: str, state: dict):
    from _shared.chat_ui import Reply

    topic = message.strip() or "RAG"
    out = C.run(topic)
    return [
        Reply("🔎 **Researcher** gathered:\n" + "\n".join(f"- {p}" for p in out.research)),
        Reply(f"✍️ **Writer** drafted:\n\n{out.draft}"),
        Reply("🧹 **Editor** finalised:"),
        Reply(out.final),
    ]
