"""Shared agent utilities — safe tools and small primitives reused by agent projects.

The agent projects teach concepts (ReAct loop, tool use, multi-agent roles) with
offline, deterministic logic so they're testable without API keys, while each README
shows the production framework (LangGraph, CrewAI, AutoGen, PydanticAI, A2A).
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from typing import Callable

# Safe arithmetic evaluator (no eval()) — a common, safe agent "calculator" tool.
_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.USub: operator.neg, ast.FloorDiv: operator.floordiv,
}


def safe_calc(expr: str) -> float:
    """Evaluate an arithmetic expression safely (digits and + - * / ** % only)."""
    def ev(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError("non-numeric constant")
        if isinstance(node, ast.BinOp):
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp):
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("unsupported expression")
    return ev(ast.parse(expr, mode="eval").body)


@dataclass
class Tool:
    name: str
    description: str
    run: Callable[[str], str]
