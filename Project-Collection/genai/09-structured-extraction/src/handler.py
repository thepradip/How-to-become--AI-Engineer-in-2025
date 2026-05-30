"""Structured-extraction brain. Uses the shared chat UI — paste text, get validated JSON."""

from __future__ import annotations

from . import extract as E

_SAMPLE = "Hi, I'm Jane Lee from Globex. Email jane.lee@globex.com — our budget is 50,000 for this."


def respond(message: str, state: dict):
    from _shared.chat_ui import Reply

    text = message.strip()
    if text.lower() in {"", "demo", "help"}:
        text = _SAMPLE
    try:
        lead = E.extract(text)
        return [Reply("**Validated structured output:**"),
                Reply(lead.model_dump_json(indent=2), "code", {"language": "json"}),
                Reply("✅ Passed schema validation (email format, non-negative budget).", "text")]
    except Exception as exc:
        return [Reply(f"> {text}", "text"),
                Reply(f"❌ Guardrail caught invalid data: {exc}", "error")]
