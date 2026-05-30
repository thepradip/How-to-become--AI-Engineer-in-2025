"""A2A — agents from different frameworks talking via a shared protocol.

The **A2A (Agent-to-Agent) protocol** lets agents built with *different* frameworks
(LangGraph, CrewAI, ADK…) discover each other's skills and exchange tasks over a common
message format. Here we model that protocol with a tiny message schema and two agents
(a "LangGraph-style" math agent + a "CrewAI-style" writer agent) that collaborate. The
README points to the real A2A spec.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from _shared.agents import safe_calc


@dataclass
class A2AMessage:
    task_id: str
    sender: str
    recipient: str
    parts: list[str] = field(default_factory=list)


class Agent:
    def __init__(self, name: str, framework: str, skills: list[str]):
        self.name = name
        self.framework = framework
        self.skills = skills

    def card(self) -> dict:
        """An A2A 'agent card' advertising skills (what discovery reads)."""
        return {"name": self.name, "framework": self.framework, "skills": self.skills}


class MathAgent(Agent):
    def __init__(self):
        super().__init__("calc-bot", "LangGraph", ["compute"])

    def handle(self, msg: A2AMessage) -> A2AMessage:
        result = str(safe_calc(msg.parts[0]))
        return A2AMessage(msg.task_id, self.name, msg.sender, [result])


class WriterAgent(Agent):
    def __init__(self):
        super().__init__("writer-bot", "CrewAI", ["summarize"])

    def handle(self, msg: A2AMessage) -> A2AMessage:
        return A2AMessage(msg.task_id, self.name, msg.sender, [f"The result is {msg.parts[0]}."])


REGISTRY = [MathAgent(), WriterAgent()]


def discover(skill: str) -> Agent | None:
    """A2A discovery: find an agent advertising a skill."""
    return next((a for a in REGISTRY if skill in a.skills), None)


def run(expression: str) -> dict:
    """Coordinator: ask the math agent to compute, then the writer agent to announce — over A2A."""
    transcript = []
    math = discover("compute")
    m1 = A2AMessage("t1", "coordinator", math.name, [expression])
    transcript.append(vars(m1))
    r1 = math.handle(m1)
    transcript.append(vars(r1))

    writer = discover("summarize")
    m2 = A2AMessage("t1", "coordinator", writer.name, r1.parts)
    transcript.append(vars(m2))
    r2 = writer.handle(m2)
    transcript.append(vars(r2))
    return {"transcript": transcript, "final": r2.parts[0],
            "cards": [a.card() for a in REGISTRY]}
