import pathlib
import sys

import pytest

pytest.importorskip("pydantic")

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import typedagent as T  # noqa: E402


def test_weather_tool():
    res = T.run("what's the weather in Paris?")
    assert res["tool"] == "get_weather" and "Paris" in res["result"]


def test_currency_tool():
    res = T.run("convert 100 USD to EUR")
    assert res["tool"] == "convert_currency" and "EUR" in res["result"]


def test_invalid_currency_rejected():
    with pytest.raises(Exception):
        T.run("convert 100 USD to XYZ")


def test_handler():
    from src import handler as H

    assert isinstance(H.respond("weather in Tokyo", {"config": {}}), list)
