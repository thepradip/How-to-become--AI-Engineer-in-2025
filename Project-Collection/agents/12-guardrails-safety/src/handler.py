"""Guardrails brain. Uses the shared chat UI. `out: <text>` tests the output rail."""

from __future__ import annotations

from . import rails as R


def respond(message: str, state: dict):
    from _shared.chat_ui import Reply

    msg = message.strip()
    if msg.lower().startswith("out:"):
        res = R.output_rail(msg[4:].strip())
        return [Reply("**Output rail (PII scrub):**"),
                Reply(res["text"]),
                Reply({"masked": ", ".join(res["masked"]) or "none"}, "metric")]

    # Treat input as a user request; run the input rail with a benign mock model output.
    mock_output = "Here is some general information for you."
    res = R.run(msg or "hello", mock_output)
    tag = "✅ allowed" if res["allowed"] else f"🛡️ blocked ({res['reason']})"
    return [Reply(f"`input rail: {tag}`", "text"), Reply(res["response"])]
