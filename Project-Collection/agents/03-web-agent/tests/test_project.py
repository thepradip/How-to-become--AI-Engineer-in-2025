import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import webagent as W  # noqa: E402


def test_extracts_price():
    res = W.run("what is the product price?")
    assert res["answer"] == "$49.99"
    assert any("product" in step for step in res["trace"])


def test_about_navigation():
    res = W.run("when was the company founded?")
    assert "2020" in res["answer"]


def test_handler():
    from src import handler as H

    assert isinstance(H.respond("price?", {"config": {}}), list)
