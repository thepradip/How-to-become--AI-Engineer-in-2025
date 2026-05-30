import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import conversation as Co  # noqa: E402


def test_dialogue_alternates_and_refines():
    d = Co.run("write a tagline", rounds=2)
    agents = [t["agent"] for t in d.transcript]
    assert agents[0] == "Solver" and "Critic" in agents
    assert d.final and "Revised" in d.final


def test_handler():
    from src import handler as H

    parts = H.respond("outline a post", {"config": {}})
    assert isinstance(parts, list) and len(parts) == 2
