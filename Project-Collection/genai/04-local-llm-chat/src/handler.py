"""Local LLM chat brain — talks to Ollama / vLLM (or mock) via the shared client.

Keeps a short conversation history and sends it to a locally-served model. With a
running Ollama (`ollama run qwen3:4b`) or vLLM OpenAI-compatible server it's a real
local chatbot; offline it falls back to the deterministic mock.
"""

from __future__ import annotations

SYSTEM = "You are a concise, helpful assistant running locally."


def respond(message: str, state: dict):
    from _shared.chat_ui import Reply
    from _shared.llm import available_providers, complete

    msg = message.strip()
    if msg.lower() == "reset":
        state["hist"] = []
        return Reply("Conversation cleared.")
    if msg.lower() in {"providers", "status"}:
        return Reply("Backends (first available is used): " + ", ".join(available_providers()))

    hist = state.setdefault("hist", [])
    hist.append(("User", msg))
    transcript = "\n".join(f"{role}: {text}" for role, text in hist[-8:])
    res = complete(f"{transcript}\nAssistant:", system=SYSTEM,
                   model=state.get("config", {}).get("model"))
    hist.append(("Assistant", res["text"]))
    return [Reply(f"`provider: {res['provider']}`", "text"), Reply(res["text"], "text")]
