"""Prompt-engineering playground. Uses the shared chat UI + shared LLM client.

Type a task; see the model's answer and the provider used. The `cot` command shows how
a chain-of-thought instruction changes the prompt. Works offline (mock) or with a real
provider when an API key / Ollama is available.
"""

from __future__ import annotations


def respond(message: str, state: dict):
    from _shared.chat_ui import Reply
    from _shared.llm import available_providers, complete

    msg = message.strip()
    if msg.lower() in {"providers", "status"}:
        return Reply("Available providers (first is used): " + ", ".join(available_providers()))

    if msg.lower().startswith("cot:"):
        task = msg[4:].strip()
        prompt = f"{task}\n\nLet's think step by step, then give the final answer."
    else:
        prompt = msg or "Explain what prompt engineering is in one sentence."

    res = complete(prompt, system=state.get("config", {}).get("system"))
    return [
        Reply(f"`provider: {res['provider']}`", "text"),
        Reply(res["text"], "text"),
        Reply("Tip: prefix with `cot:` for chain-of-thought, or type `providers`.", "text"),
    ]
