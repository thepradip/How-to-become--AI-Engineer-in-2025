import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import a2a as A  # noqa: E402


def test_discovery_by_skill():
    assert A.discover("compute").framework == "LangGraph"
    assert A.discover("summarize").framework == "CrewAI"
    assert A.discover("nope") is None


def test_two_agents_collaborate():
    res = A.run("6 * 7")
    assert "42" in res["final"]
    # 4 messages: coord->math, math->coord, coord->writer, writer->coord
    assert len(res["transcript"]) == 4


def test_handler():
    from src import handler as H

    assert isinstance(H.respond("(10 + 2) * 5", {"config": {}}), list)
