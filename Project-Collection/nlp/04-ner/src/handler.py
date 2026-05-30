"""NER brain. Uses the shared chat UI — type text, get entities."""

from __future__ import annotations

from . import ner as N

_SAMPLE = ("John Smith joined Acme Corp on March 3, 2026 as CTO, earning $250,000 a year. "
           "Contact him at john.smith@acme.com about the 15% equity offer in New York.")


def respond(message: str, state: dict):
    import pandas as pd

    from _shared.chat_ui import Reply

    text = _SAMPLE if message.strip().lower() in {"demo", "example", ""} else message
    ents = N.extract_entities(text)
    parts = [Reply(f"> {text}", "text")]
    if ents:
        parts.append(Reply(pd.DataFrame(ents)[["text", "label"]], "table"))
    else:
        parts.append(Reply("No entities found. Try `demo`, or paste text with names/dates/money.", "text"))
    return parts
