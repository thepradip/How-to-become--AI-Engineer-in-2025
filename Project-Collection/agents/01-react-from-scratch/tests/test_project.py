import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from _shared.agents import safe_calc  # noqa: E402
from src import react as Rx  # noqa: E402


def test_safe_calc():
    assert safe_calc("12 * (3 + 4)") == 84
    import pytest
    with pytest.raises(Exception):
        safe_calc("__import__('os')")


def test_react_math():
    trace = Rx.run("compute 12 * (3 + 4)")
    assert "84" in trace.answer
    assert any("action" in s for s in trace.steps)


def test_react_lookup():
    trace = Rx.run("what is RAG?")
    assert "retriev" in trace.answer.lower()


def test_handler():
    from src import handler as H

    assert isinstance(H.respond("what is python?", {"config": {}}), list)
