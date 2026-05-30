"""Translation brain. Uses the shared chat UI."""

from __future__ import annotations

from . import translate as Tr


def respond(message: str, state: dict):
    from _shared.chat_ui import Reply

    target = state.get("config", {}).get("target", "fr")
    text = message.strip()
    if text.lower() in {"demo", "", "help"}:
        text = "hello friend the book is good today"
    out = Tr.translate(text, target)
    return [
        Reply(f"**English → {Tr.LANGUAGES.get(target, target)}** (offline word-lookup demo):"),
        Reply({"Source": text, "Translation": out}, "metric"),
        Reply("_For real neural translation install transformers and use NLLB (README)._", "text"),
    ]
