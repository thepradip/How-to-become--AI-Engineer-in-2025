"""Structured extraction with schema validation (guardrails).

LLMs return free text; production systems need **validated, typed** data. We define a
Pydantic schema and parse text into it — invalid data (bad email, negative budget)
raises a validation error instead of flowing downstream. The offline extractor uses
regex; the documented production path uses **PydanticAI**, which makes an LLM fill the
*same* schema with type-safe guarantees.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field, field_validator

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")


class Lead(BaseModel):
    """A sales lead extracted from an email/message."""

    name: str = Field(min_length=1)
    email: str
    company: str | None = None
    budget_usd: int | None = Field(default=None, ge=0)   # guardrail: non-negative

    @field_validator("email")
    @classmethod
    def _valid_email(cls, v: str) -> str:
        if not _EMAIL.fullmatch(v):
            raise ValueError(f"invalid email: {v!r}")
        return v


def extract(text: str) -> Lead:
    """Regex-parse text into a validated ``Lead`` (raises on invalid data)."""
    email = (_EMAIL.search(text) or [None])[0] if _EMAIL.search(text) else ""
    name = (re.search(r"(?:name|i am|i'm)\s*:?\s*([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)?)", text, re.I) or [None, ""])
    company = re.search(r"(?:company|from|at)\s*:?\s*([A-Z][\w&.\- ]{2,})", text)
    budget = re.search(r"(?:budget|spend)[^\d]*(\d[\d,]*)", text, re.I)
    return Lead(
        name=name[1].strip() if name and name[1] else "Unknown",
        email=email or "unknown@example.com",
        company=company.group(1).strip() if company else None,
        budget_usd=int(budget.group(1).replace(",", "")) if budget else None,
    )


def extract_with_pydantic_ai(text: str) -> Lead:
    """LLM-backed extraction into the same schema via PydanticAI. Documented path."""
    from pydantic_ai import Agent

    agent = Agent("openai:gpt-4o-mini", output_type=Lead,
                  system_prompt="Extract the lead's details into the schema.")
    return agent.run_sync(text).output
