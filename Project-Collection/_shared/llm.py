"""Shared LLM client with a graceful provider chain — used by GenAI & agent projects.

``complete(prompt, ...)`` tries, in order:
  1. OpenAI  (if ``OPENAI_API_KEY`` set and ``openai`` installed)
  2. Anthropic (if ``ANTHROPIC_API_KEY`` set and ``anthropic`` installed)
  3. Ollama  (if a local server is reachable at 127.0.0.1:11434)
  4. a deterministic **mock** — so every project runs and is testable offline.

The mock is intentionally simple but useful: given a ``Context:`` block it answers
extractively (great for RAG demos); otherwise it returns a short, structured reply.
This lets the course run with zero credentials, then "light up" when a key or local
model is available.
"""

from __future__ import annotations

import os
import re


def available_providers() -> list[str]:
    out = []
    if os.getenv("OPENAI_API_KEY"):
        out.append("openai")
    if os.getenv("ANTHROPIC_API_KEY"):
        out.append("anthropic")
    if _ollama_up():
        out.append("ollama")
    out.append("mock")
    return out


def complete(prompt: str, system: str | None = None, model: str | None = None,
             temperature: float = 0.0) -> dict:
    """Return {'text': str, 'provider': str} using the first available backend."""
    for provider in available_providers():
        try:
            if provider == "openai":
                return {"text": _openai(prompt, system, model or "gpt-4o-mini", temperature), "provider": "openai"}
            if provider == "anthropic":
                return {"text": _anthropic(prompt, system, model or "claude-3-5-haiku-latest", temperature), "provider": "anthropic"}
            if provider == "ollama":
                return {"text": _ollama(prompt, system, model or "qwen3:4b"), "provider": "ollama"}
        except Exception:
            continue  # fall through to the next provider
    return {"text": mock_complete(prompt, system), "provider": "mock"}


def mock_complete(prompt: str, system: str | None = None) -> str:
    """Deterministic offline stand-in. Extractive when a Context block is present."""
    m = re.search(r"context:\s*(.+?)(?:\n\n|\nquestion:|\nq:|$)", prompt, re.I | re.S)
    q = re.search(r"(?:question|q):\s*(.+)$", prompt, re.I | re.S)
    if m:
        context, question = m.group(1), (q.group(1) if q else prompt)
        sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", context) if s.strip()]
        if sents:
            qwords = set(re.findall(r"[a-z]+", question.lower()))
            best = max(sents, key=lambda s: len(qwords & set(re.findall(r"[a-z]+", s.lower()))))
            return f"(mock) Based on the context: {best}"
    head = (prompt.strip().splitlines() or ["…"])[0][:200]
    return f"(mock LLM) I would respond to: \"{head}\". Set an API key or run Ollama for a real model."


# ── backends ────────────────────────────────────────────────────────────────
def _openai(prompt, system, model, temperature):
    from openai import OpenAI

    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    r = OpenAI().chat.completions.create(model=model, messages=msgs, temperature=temperature)
    return r.choices[0].message.content


def _anthropic(prompt, system, model, temperature):
    import anthropic

    r = anthropic.Anthropic().messages.create(
        model=model, max_tokens=1024, temperature=temperature,
        system=system or "", messages=[{"role": "user", "content": prompt}])
    return r.content[0].text


def _ollama_up() -> bool:
    try:
        import urllib.request

        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=0.3)
        return True
    except Exception:
        return False


def _ollama(prompt, system, model):
    import json
    import urllib.request

    body = json.dumps({"model": model, "prompt": prompt, "system": system or "", "stream": False}).encode()
    req = urllib.request.Request("http://127.0.0.1:11434/api/generate", data=body,
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=60).read())["response"]
