"""A ReAct agent from scratch — understand the loop before using a framework.

ReAct = **Reason + Act**: the agent alternates Thought → Action (call a tool) →
Observation, until it can give a final Answer. Here the policy is rule-based (so it runs
offline and deterministically); swapping in an LLM to choose actions is the only change
needed for the real thing (documented in the README).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from _shared.agents import safe_calc

_FACTS = {
    "python": "Python is a high-level programming language created by Guido van Rossum.",
    "rag": "RAG retrieves documents and feeds them to an LLM to ground its answers.",
    "transformer": "The Transformer is a neural network architecture based on attention.",
}


@dataclass
class Trace:
    steps: list[dict] = field(default_factory=list)
    answer: str = ""


def _calculator(arg: str) -> str:
    try:
        return str(safe_calc(arg))
    except Exception as exc:
        return f"error: {exc}"


def _lookup(arg: str) -> str:
    key = next((k for k in _FACTS if k in arg.lower()), None)
    return _FACTS[key] if key else "no entry found"


TOOLS = {"calculator": _calculator, "lookup": _lookup}


def run(task: str, max_steps: int = 5) -> Trace:
    """Run the ReAct loop with a rule-based policy. Returns the reasoning trace."""
    trace = Trace()
    # Policy: arithmetic → calculator; "what is X" → lookup; else answer directly.
    math = re.search(r"[-+*/%().\d\s]{3,}", task)
    if any(c.isdigit() for c in task) and math and re.search(r"[-+*/]", task):
        expr = math.group().strip()
        trace.steps.append({"thought": "This needs arithmetic.", "action": f"calculator({expr})"})
        obs = TOOLS["calculator"](expr)
        trace.steps.append({"observation": obs})
        trace.answer = f"The result is {obs}."
    elif re.search(r"what is|who is|define|explain", task, re.I):
        trace.steps.append({"thought": "I should look this up.", "action": f"lookup({task})"})
        obs = TOOLS["lookup"](task)
        trace.steps.append({"observation": obs})
        trace.answer = obs
    else:
        trace.steps.append({"thought": "I can answer directly."})
        trace.answer = "I can use a calculator or a knowledge lookup. Try a math expression or 'what is RAG?'."
    return trace
