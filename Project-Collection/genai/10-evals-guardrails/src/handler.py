"""Evals + guardrails brain. Uses the shared chat UI."""

from __future__ import annotations

from . import evals as E

_SAMPLES = [
    {"pred": "RAG reduces hallucination by grounding answers in documents.",
     "ref": "RAG reduces hallucination.", "keywords": ["hallucination", "documents"]},
    {"pred": "The capital is Paris.", "ref": "Paris", "keywords": ["Paris"]},
    {"pred": "Contact me at john@acme.com or 555-123-4567.", "ref": "", "keywords": []},
]


def respond(message: str, state: dict):
    from _shared.chat_ui import Reply

    msg = message.strip()
    if msg and msg.lower() not in {"demo", "run", "help", "evaluate"}:
        # Treat free text as a single output to guardrail-check.
        pii = E.pii_guardrail(msg)
        unsafe = E.safety_guardrail(msg)
        return [Reply("**Guardrail check on your text:**"),
                Reply({"PII found": ", ".join(pii) or "none", "Unsafe terms": ", ".join(unsafe) or "none"}, "metric")]

    res = E.evaluate(_SAMPLES)
    return [
        Reply("Ran the eval + guardrail harness on sample outputs:"),
        Reply(res["per_sample"], "table"),
        Reply(res["aggregate"], "metric"),
        Reply("Paste any text to run the PII/safety guardrails on it. Production: Ragas/DeepEval + Guardrails AI/NeMo (README).", "text"),
    ]
