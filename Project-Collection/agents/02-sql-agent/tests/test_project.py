import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import sqlagent as S  # noqa: E402


def test_count_query():
    res = S.answer("how many employees are in Engineering?")
    assert "COUNT(*)" in res["sql"] and "Engineering" in res["sql"]
    assert res["rows"][0]["n"] == 3


def test_top_earner():
    res = S.answer("who is the top earner?")
    assert "ORDER BY salary DESC" in res["sql"]
    assert res["rows"][0]["name"] == "Frank"


def test_group_by():
    res = S.answer("total salary by department")
    assert "GROUP BY department" in res["sql"]
    assert len(res["rows"]) == 3


def test_handler():
    from src import handler as H

    assert isinstance(H.respond("average salary in Sales", {"config": {}}), list)
