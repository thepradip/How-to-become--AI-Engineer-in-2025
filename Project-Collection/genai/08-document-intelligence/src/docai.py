"""Document Intelligence — OCR a document, then extract structured fields.

A real pipeline: scan/PDF → **OCR** (Mistral OCR / olmOCR / Docling / Qwen2.5-VL) → text
→ **field extraction**. The OCR step needs a model/engine (documented); the extraction
step (regex/LLM over the OCR'd text) is the part we test offline, since it's where most
of the business logic lives.
"""

from __future__ import annotations

import re

SAMPLE_INVOICE = """ACME SUPPLIES LTD
Invoice #: INV-2026-00471
Date: 2026-03-14
Bill To: Globex Corp

Description           Qty   Amount
Widgets               10    $1,200.00
Installation           1      $450.00

Subtotal: $1,650.00
Tax (10%): $165.00
Total Due: $1,815.00
"""


def extract_fields(text: str) -> dict:
    """Pull key invoice fields from OCR'd text with regex (offline)."""
    def find(pat, default="—"):
        m = re.search(pat, text, re.I)
        return m.group(1).strip() if m else default

    return {
        "invoice_number": find(r"invoice\s*#?:?\s*([A-Z0-9\-]+)"),
        "date": find(r"date:?\s*(\d{4}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"),
        "vendor": (text.strip().splitlines() or ["—"])[0].strip(),
        "total": find(r"\btotal\s*(?:due)?:?\s*([$£€]?\s?\d[\d,]*\.?\d*)"),
        "tax": find(r"tax[^:]*:?\s*([$£€]?\s?\d[\d,]*\.?\d*)"),
    }


def ocr_image(path: str) -> str:
    """OCR an image/PDF to text. Documented production path (needs an OCR engine).

    Options: ``docling`` (IBM), ``olmocr`` (AllenAI), Mistral OCR API, or a VLM like
    Qwen2.5-VL. Example with Docling:
        from docling.document_converter import DocumentConverter
        return DocumentConverter().convert(path).document.export_to_markdown()
    """
    from docling.document_converter import DocumentConverter

    return DocumentConverter().convert(path).document.export_to_markdown()
