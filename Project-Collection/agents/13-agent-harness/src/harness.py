"""Agent harness / orchestration — the production wrapper around agents.

Shipping agents needs more than a loop: a **registry** of skills, a **router/orchestrator**,
**observability** (step traces + metrics), and **retries**. This module implements a small
harness with those pieces. The README maps it to the **Claude Agent SDK** and LangGraph
for production orchestration + tracing.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from _shared.agents import safe_calc


def _skill_math(arg: str) -> str:
    return str(safe_calc(arg))


def _skill_echo(arg: str) -> str:
    return arg.strip()


def _skill_upper(arg: str) -> str:
    return arg.strip().upper()


REGISTRY = {
    "math": {"run": _skill_math, "desc": "evaluate arithmetic"},
    "echo": {"run": _skill_echo, "desc": "repeat input"},
    "upper": {"run": _skill_upper, "desc": "uppercase text"},
}


@dataclass
class RunResult:
    result: str
    skill: str
    trace: list[dict] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def route(task: str) -> str:
    """Pick a skill for the task (the orchestrator's decision)."""
    t = task.lower()
    if any(c.isdigit() for c in t) and any(op in t for op in "+-*/"):
        return "math"
    if t.startswith("upper") or "uppercase" in t:
        return "upper"
    return "echo"


def run(task: str, max_retries: int = 1) -> RunResult:
    """Route → execute with retries → return result + observability trace + metrics."""
    skill = route(task)
    arg = task
    if skill == "math":
        import re
        m = re.search(r"[-+*/%().\d\s]{3,}", task)
        arg = m.group().strip() if m else task
    elif skill == "upper":
        arg = task.split(":", 1)[-1].strip() if ":" in task else task

    trace, attempts = [], 0
    result = ""
    while attempts <= max_retries:
        attempts += 1
        trace.append({"step": attempts, "skill": skill, "action": f"{skill}({arg})"})
        try:
            result = REGISTRY[skill]["run"](arg)
            trace.append({"step": attempts, "observation": result, "status": "ok"})
            break
        except Exception as exc:  # retry on failure
            trace.append({"step": attempts, "observation": str(exc), "status": "error"})
            result = f"failed: {exc}"
    metrics = {"skill": skill, "attempts": attempts,
               "steps": len(trace), "approx_tokens": sum(len(str(s)) for s in trace) // 4}
    return RunResult(result=result, skill=skill, trace=trace, metrics=metrics)
