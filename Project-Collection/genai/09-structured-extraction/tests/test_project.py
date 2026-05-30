import pathlib
import sys

import pytest

pytest.importorskip("pydantic")

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import extract as E  # noqa: E402


def test_extracts_valid_lead():
    lead = E.extract("I'm Jane Lee from Globex. Email jane@globex.com budget 50,000")
    assert lead.email == "jane@globex.com"
    assert lead.budget_usd == 50000


def test_guardrail_rejects_bad_email():
    with pytest.raises(Exception):
        E.Lead(name="X", email="not-an-email")


def test_guardrail_rejects_negative_budget():
    with pytest.raises(Exception):
        E.Lead(name="X", email="x@y.com", budget_usd=-5)


def test_handler_demo():
    from src import handler as H

    assert isinstance(H.respond("demo", {"config": {}}), list)
