import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import harness as Ha  # noqa: E402


def test_routes_math():
    res = Ha.run("12 * (3 + 4)")
    assert res.skill == "math" and res.result == "84"
    assert res.trace and "approx_tokens" in res.metrics


def test_routes_upper():
    res = Ha.run("upper: hello")
    assert res.skill == "upper" and res.result == "HELLO"


def test_routes_echo():
    assert Ha.run("just say this back").skill == "echo"


def test_handler():
    from src import handler as H

    parts = H.respond("5 + 5", {"config": {}})
    assert isinstance(parts, list) and len(parts) == 5
