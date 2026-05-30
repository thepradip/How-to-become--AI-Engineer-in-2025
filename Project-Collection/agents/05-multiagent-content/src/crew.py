"""Multi-agent content team — researcher → writer → editor (sequential crew).

Decomposing work into role-specialised agents that hand off to each other is CrewAI's
sweet spot. Here three roles run as a deterministic pipeline (offline); the README shows
the CrewAI version where each role is an LLM agent with its own goal and tools.
"""

from __future__ import annotations

from dataclasses import dataclass

_KB = {
    "rag": ["RAG grounds answers in retrieved documents", "it reduces hallucination",
            "a vector DB stores embeddings for retrieval"],
    "agents": ["agents combine an LLM with tools", "they loop: reason then act",
               "frameworks include LangGraph and CrewAI"],
    "fine-tuning": ["LoRA trains small adapters", "QLoRA adds 4-bit quantization",
                    "Unsloth speeds it up on one GPU"],
}


@dataclass
class CrewOutput:
    research: list[str]
    draft: str
    final: str


def researcher(topic: str) -> list[str]:
    key = next((k for k in _KB if k in topic.lower()), None)
    return _KB.get(key, [f"{topic} is an important topic", "it has several key aspects",
                         "practitioners apply it widely"])


def writer(topic: str, points: list[str]) -> str:
    body = ". ".join(p.capitalize() for p in points)
    return f"{topic.capitalize()} matters today. {body}."


def editor(draft: str, topic: str) -> str:
    title = f"# {topic.title()}\n\n"
    polished = draft.replace("  ", " ").strip()
    if not polished.endswith("."):
        polished += "."
    return title + polished


def run(topic: str) -> CrewOutput:
    """Run the researcher → writer → editor pipeline."""
    points = researcher(topic)
    draft = writer(topic, points)
    final = editor(draft, topic)
    return CrewOutput(research=points, draft=draft, final=final)
