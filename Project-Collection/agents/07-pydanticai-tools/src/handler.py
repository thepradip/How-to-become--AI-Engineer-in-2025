"""Type-safe tool agent brain. Uses the shared chat UI."""

from __future__ import annotations

from . import typedagent as T


def respond(message: str, state: dict):
    from _shared.chat_ui import Reply

    req = message.strip() or "what's the weather in Paris?"
    try:
        res = T.run(req)
        return [Reply(f"🔧 tool: **{res['tool']}** · args validated ✅"),
                Reply({"args": str(res["args"])}, "metric") if res["args"] else Reply(""),
                Reply(res["result"])]
    except Exception as exc:
        return [Reply(f"> {req}", "text"), Reply(f"❌ Validation error: {exc}", "error")]
