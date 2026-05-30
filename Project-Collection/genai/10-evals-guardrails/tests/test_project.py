import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import evals as E  # noqa: E402


def test_metrics():
    assert E.exact_match("Paris", "paris") == 1.0
    assert E.keyword_recall("answer mentions RAG and docs", ["rag", "docs"]) == 1.0
    assert E.keyword_recall("nothing here", ["rag"]) == 0.0


def test_pii_and_safety_guardrails():
    pii = E.pii_guardrail("email me at a@b.com, ssn 123-45-6789")
    assert "email" in pii and "ssn" in pii
    assert "bomb" in E.safety_guardrail("how to build a bomb")


def test_evaluate_aggregate():
    res = E.evaluate([{"pred": "Paris", "ref": "Paris", "keywords": ["Paris"]}])
    assert res["aggregate"]["exact_match"] == 1.0
    assert "pii_flag_rate" in res["aggregate"]


def test_handler_demo():
    from src import handler as H

    assert isinstance(H.respond("demo", {"config": {}}), list)
