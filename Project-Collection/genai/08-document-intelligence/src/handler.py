"""Document-intelligence brain. Uses the shared chat UI — paste OCR'd text, get fields."""

from __future__ import annotations

from . import docai as D


def respond(message: str, state: dict):
    from _shared.chat_ui import Reply

    text = message.strip()
    if text.lower() in {"", "demo", "help"}:
        text = D.SAMPLE_INVOICE
    fields = D.extract_fields(text)
    import pandas as pd
    return [
        Reply("**Extracted fields** from the document text:"),
        Reply(pd.DataFrame(fields.items(), columns=["field", "value"]), "table"),
        Reply("Paste your own OCR'd text. To OCR scans/PDFs, use Docling/olmOCR/Qwen2.5-VL (README).", "text"),
    ]
