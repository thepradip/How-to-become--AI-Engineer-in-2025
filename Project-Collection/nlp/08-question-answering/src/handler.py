"""QA brain. Uses the shared chat UI. `context: <text>` sets the passage; else ask a question."""

from __future__ import annotations

from . import qa as Q


def respond(message: str, state: dict):
    from _shared.chat_ui import Reply

    msg = message.strip()
    context = state.get("context", Q.SAMPLE_CONTEXT)

    if msg.lower().startswith("context:"):
        state["context"] = msg[len("context:"):].strip()
        return Reply("Context set. Now ask a question about it.")

    if msg.lower() in {"", "help", "demo"}:
        return [Reply("Current context:", "text"), Reply(context, "text"),
                Reply("Ask a question, e.g. *what does RAG reduce?* "
                      "(or set your own with `context: ...`)", "text")]

    res = Q.answer(msg, context)
    return [Reply(f"**Q:** {msg}"), Reply({"Answer": res["answer"], "Relevance": res["score"]}, "metric")]
