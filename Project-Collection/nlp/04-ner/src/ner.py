"""Named Entity Recognition.

NER pulls structured entities (people, orgs, money, dates…) out of free text — the
backbone of resume parsers, contract analysis and clinical-note mining. This module
ships a dependency-free **rule-based** extractor (regex + capitalisation heuristics) so
it runs offline, and documents the **fine-tuned BERT / spaCy** path for production
accuracy.
"""

from __future__ import annotations

import re

# Order matters: more specific patterns first so they win overlapping spans.
_PATTERNS = [
    ("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")),
    ("MONEY", re.compile(r"(?:[$£€]\s?\d[\d,]*(?:\.\d+)?|\b\d[\d,]*(?:\.\d+)?\s?(?:USD|EUR|GBP|dollars|euros)\b)")),
    ("PERCENT", re.compile(r"\b\d+(?:\.\d+)?\s?%")),
    ("DATE", re.compile(r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
                        r"(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}?,?\s*\d{4}|"
                        r"(?:19|20)\d{2})\b")),
    ("ORG", re.compile(r"\b(?:[A-Z][\w&.-]+\s)*[A-Z][\w&.-]+\s(?:Inc|Corp|Corporation|Ltd|LLC|LLP|Group|Technologies|Labs|University)\b")),
]
# Generic proper-noun run (people/places) — applied last on remaining spans.
_PROPER = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")


def extract_entities(text: str) -> list[dict]:
    """Return a list of {text, label, start} entities, non-overlapping, in order."""
    spans: list[tuple[int, int, str, str]] = []
    taken = [False] * (len(text) + 1)

    def claim(s, e):
        if any(taken[s:e]):
            return False
        for i in range(s, e):
            taken[i] = True
        return True

    for label, pat in _PATTERNS:
        for m in pat.finditer(text):
            if claim(m.start(), m.end()):
                spans.append((m.start(), m.end(), label, m.group().strip()))
    for m in _PROPER.finditer(text):
        if claim(m.start(), m.end()):
            spans.append((m.start(), m.end(), "PERSON/PLACE", m.group().strip()))

    spans.sort()
    return [{"text": t, "label": lab, "start": s} for s, e, lab, t in spans]


def extract_with_transformer(text: str):
    """Optional: HF token-classification pipeline (downloads a model). Documented path."""
    from transformers import pipeline

    ner = pipeline("ner", grouped_entities=True)
    return [{"text": e["word"], "label": e["entity_group"], "start": int(e["start"])} for e in ner(text)]
