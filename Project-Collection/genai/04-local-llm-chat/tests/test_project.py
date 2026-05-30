import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))


def test_chat_keeps_history_and_responds():
    from src import handler as H

    state: dict = {"config": {}}
    out1 = H.respond("hello", state)
    assert isinstance(out1, list) and len(state["hist"]) == 2  # user + assistant
    H.respond("and again", state)
    assert len(state["hist"]) == 4


def test_reset():
    from src import handler as H

    state: dict = {"config": {}}
    H.respond("hi", state)
    H.respond("reset", state)
    assert state["hist"] == []
