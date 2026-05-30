"""Guardrails / safety rails — input and output checks around an agent.

Production agents wrap the model in **rails**: an *input rail* blocks jailbreaks / disallowed
requests before they reach the model, and an *output rail* scrubs PII and unsafe content
before the response reaches the user. This is the offline, rule-based version of what
**NeMo Guardrails** / **Guardrails AI** do declaratively (documented).
"""

from __future__ import annotations

import re

_JAILBREAK = ["ignore previous instructions", "ignore all instructions", "pretend you are",
              "developer mode", "disregard your rules", "you have no restrictions"]
_DISALLOWED = ["build a bomb", "make a weapon", "hack into", "steal credentials"]
_PII = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "phone": re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
}


def input_rail(user_input: str) -> dict:
    """Return {'allowed': bool, 'reason': str} for an incoming request."""
    low = user_input.lower()
    if any(j in low for j in _JAILBREAK):
        return {"allowed": False, "reason": "jailbreak attempt blocked"}
    if any(d in low for d in _DISALLOWED):
        return {"allowed": False, "reason": "disallowed request blocked"}
    return {"allowed": True, "reason": "ok"}


def output_rail(text: str) -> dict:
    """Mask PII in an outgoing response; return {'text', 'masked': [types]}."""
    masked = []
    out = text
    for kind, pat in _PII.items():
        if pat.search(out):
            masked.append(kind)
            out = pat.sub(f"[REDACTED {kind.upper()}]", out)
    return {"text": out, "masked": masked}


def run(user_input: str, model_output: str) -> dict:
    """Apply input rail, then (if allowed) the output rail to the model's response."""
    inp = input_rail(user_input)
    if not inp["allowed"]:
        return {"allowed": False, "reason": inp["reason"],
                "response": "Sorry, I can't help with that request."}
    out = output_rail(model_output)
    return {"allowed": True, "reason": "ok", "response": out["text"], "masked": out["masked"]}
