"""RAG brain. Uses the shared chat UI. `docs: <text>` loads docs; else ask a question."""

from __future__ import annotations

from . import rag as R


def _index(state):
    if "index" not in state:
        state["index"] = R.build_index(state.get("docs", R.SAMPLE_DOCS))
    return state["index"]


def respond(message: str, state: dict):
    from _shared.chat_ui import Reply

    msg = message.strip()
    if msg.lower().startswith("docs:"):
        state["docs"] = [msg[5:].strip()]
        state.pop("index", None)
        return Reply("Documents loaded & indexed. Ask a question about them.")

    if msg.lower() in {"", "help", "demo"}:
        return [Reply("Indexed sample policy docs (refunds, shipping, premium plan). "
                      "Ask e.g. *how long do refunds take?* — or load your own with `docs: ...`.")]

    res = R.answer(_index(state), msg)
    import pandas as pd
    parts = [Reply(f"`provider: {res['provider']}`", "text"), Reply(res["answer"], "text")]
    if res["sources"]:
        parts.append(Reply("**Sources:**", "text"))
        parts.append(Reply(pd.DataFrame(res["sources"])[["chunk", "score", "text"]], "table"))
    return parts
