import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import voice as V  # noqa: E402


def test_pipeline_text_orchestration():
    res = V.pipeline("hello there")
    assert res["transcript"] == "hello there"
    assert isinstance(res["reply"], str) and res["reply"]


def test_handler():
    from src import handler as H

    parts = H.respond("what time is it?", {"config": {}})
    assert isinstance(parts, list) and len(parts) == 3
