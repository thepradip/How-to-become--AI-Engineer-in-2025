"""Evals + Guardrails harness — measure LLM output quality and enforce safety.

You can't improve what you don't measure. This harness scores model outputs (exact
match, keyword/faithfulness recall) and applies **guardrails** (PII detection, banned-
content checks) — the LLMOps layer that keeps an app trustworthy. The README maps these
to production tools: **Ragas, DeepEval, Guardrails AI, NeMo Guardrails**.
"""

from __future__ import annotations

import re

import pandas as pd

_PII = {
    "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "phone": re.compile(r"\b(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)\d{3}[\s-]?\d{4}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
}
_BANNED = {"hate", "kill", "bomb", "suicide"}


def exact_match(pred: str, ref: str) -> float:
    return 1.0 if pred.strip().lower() == ref.strip().lower() else 0.0


def keyword_recall(pred: str, keywords: list[str]) -> float:
    """Faithfulness proxy: fraction of expected keywords present in the answer."""
    if not keywords:
        return 1.0
    pl = pred.lower()
    return round(sum(k.lower() in pl for k in keywords) / len(keywords), 3)


def pii_guardrail(text: str) -> dict:
    """Return any PII detected (a blocking guardrail in production)."""
    return {kind: pat.findall(text) for kind, pat in _PII.items() if pat.search(text)}


def safety_guardrail(text: str, banned: set[str] | None = None) -> list[str]:
    banned = banned or _BANNED
    return sorted({w for w in banned if re.search(rf"\b{re.escape(w)}\b", text.lower())})


def evaluate(samples: list[dict]) -> dict:
    """Score a list of {pred, ref, keywords} and return per-sample + aggregate metrics."""
    rows = []
    for s in samples:
        rows.append({
            "exact_match": exact_match(s["pred"], s.get("ref", "")),
            "keyword_recall": keyword_recall(s["pred"], s.get("keywords", [])),
            "pii": bool(pii_guardrail(s["pred"])),
            "unsafe": bool(safety_guardrail(s["pred"])),
        })
    df = pd.DataFrame(rows)
    agg = {
        "exact_match": round(float(df["exact_match"].mean()), 3),
        "keyword_recall": round(float(df["keyword_recall"].mean()), 3),
        "pii_flag_rate": round(float(df["pii"].mean()), 3),
        "unsafe_rate": round(float(df["unsafe"].mean()), 3),
    }
    return {"per_sample": df, "aggregate": agg}
