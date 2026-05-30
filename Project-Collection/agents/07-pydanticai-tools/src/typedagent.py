"""Type-safe tool agent (PydanticAI style) — validated tool contracts.

PydanticAI gives agents **type-safe** tools: each tool's arguments are a Pydantic model,
so malformed calls are rejected before execution. Here we route a request to a tool and
validate its args with Pydantic (offline). The README shows the real PydanticAI ``Agent``
where an LLM fills the same typed arguments.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, field_validator

_RATES = {"USD": 1.0, "EUR": 0.92, "GBP": 0.79, "INR": 83.0}


class WeatherArgs(BaseModel):
    city: str

    @field_validator("city")
    @classmethod
    def _nonempty(cls, v):
        if not v.strip():
            raise ValueError("city required")
        return v


class CurrencyArgs(BaseModel):
    amount: float
    src: str
    dst: str

    @field_validator("src", "dst")
    @classmethod
    def _known(cls, v):
        if v.upper() not in _RATES:
            raise ValueError(f"unknown currency {v!r}")
        return v.upper()


def get_weather(a: WeatherArgs) -> str:
    return f"{a.city}: 22°C, sunny (mock)"


def convert_currency(a: CurrencyArgs) -> str:
    out = a.amount / _RATES[a.src] * _RATES[a.dst]
    return f"{a.amount} {a.src} = {out:.2f} {a.dst}"


def run(request: str) -> dict:
    """Route a request to a typed tool, validating arguments (raises on bad args)."""
    r = request.lower()
    if "weather" in r:
        city = re.search(r"in ([A-Za-z ]+)", request)
        args = WeatherArgs(city=(city.group(1).strip() if city else ""))
        return {"tool": "get_weather", "args": args.model_dump(), "result": get_weather(args)}
    if "convert" in r or re.search(r"\d+\s*[a-z]{3}\s+to\s+[a-z]{3}", r):
        m = re.search(r"(\d+(?:\.\d+)?)\s*([a-zA-Z]{3})\s*(?:to|in)\s*([a-zA-Z]{3})", request)
        if not m:
            raise ValueError("could not parse currency request")
        args = CurrencyArgs(amount=float(m.group(1)), src=m.group(2), dst=m.group(3))
        return {"tool": "convert_currency", "args": args.model_dump(), "result": convert_currency(args)}
    return {"tool": None, "args": {}, "result": "No matching tool. Try weather or currency conversion."}
