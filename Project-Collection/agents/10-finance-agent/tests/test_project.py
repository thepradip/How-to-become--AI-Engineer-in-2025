import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import finance as F  # noqa: E402


def test_growth():
    res = F.answer("what was the revenue growth?")
    assert not res["blocked"] and "%" in res["answer"]


def test_advice_blocked():
    res = F.answer("should I invest in this company?")
    assert res["blocked"] and "advisor" in res["answer"].lower()


def test_total_revenue():
    res = F.answer("total revenue this year?")
    assert "5750" in res["answer"]   # 1200+1350+1500+1700


def test_handler():
    from src import handler as H

    assert isinstance(H.respond("net margin?", {"config": {}}), list)
