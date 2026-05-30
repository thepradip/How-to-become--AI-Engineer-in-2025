"""Conversational multi-agent (AutoGen/AG2 style) — agents refine via dialogue.

AutoGen popularised agents that *talk to each other* to improve an answer. Here a
**Solver** drafts and a **Critic** gives feedback over several rounds (deterministic,
offline). The README shows the AutoGen/AG2 version where both are LLM agents in a chat.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Dialogue:
    transcript: list[dict] = field(default_factory=list)
    final: str = ""


def solver(task: str, feedback: str | None = None) -> str:
    if feedback is None:
        return f"Draft: {task.strip().capitalize()} — a solid first attempt."
    # Incorporate the critic's feedback into a revision.
    return f"Revised: {task.strip().capitalize()} — now more specific and concise ({feedback})."


def critic(draft: str) -> str:
    if "concise" not in draft.lower():
        return "make it more concise"
    if "specific" not in draft.lower():
        return "add a concrete detail"
    return "looks good"


def run(task: str, rounds: int = 2) -> Dialogue:
    """Run a Solver↔Critic conversation for ``rounds`` and return the transcript."""
    d = Dialogue()
    draft = solver(task)
    d.transcript.append({"agent": "Solver", "message": draft})
    for _ in range(rounds):
        fb = critic(draft)
        d.transcript.append({"agent": "Critic", "message": fb})
        if fb == "looks good":
            break
        draft = solver(task, fb)
        d.transcript.append({"agent": "Solver", "message": draft})
    d.final = draft
    return d
