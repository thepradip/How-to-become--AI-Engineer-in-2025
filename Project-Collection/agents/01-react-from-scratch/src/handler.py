"""ReAct agent brain. Uses the shared chat UI — shows the reasoning trace."""

from __future__ import annotations

from . import react as Rx


def respond(message: str, state: dict):
    from _shared.chat_ui import Reply

    task = message.strip() or "what is RAG?"
    trace = Rx.run(task)
    lines = []
    for step in trace.steps:
        if "thought" in step:
            lines.append(f"🧠 **Thought:** {step['thought']}")
        if "action" in step:
            lines.append(f"🔧 **Action:** `{step['action']}`")
        if "observation" in step:
            lines.append(f"👁️ **Observation:** {step['observation']}")
    return [Reply("\n\n".join(lines)), Reply(f"✅ **Answer:** {trace.answer}")]
