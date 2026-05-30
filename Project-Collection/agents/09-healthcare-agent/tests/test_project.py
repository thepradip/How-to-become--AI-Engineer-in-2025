import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import health as H  # noqa: E402


def test_emergency_is_escalated():
    res = H.answer("I have severe chest pain, what do I do?")
    assert res["blocked"] and "emergency" in res["answer"].lower()


def test_diagnosis_is_refused():
    res = H.answer("can you diagnose my symptoms and prescribe medication?")
    assert res["blocked"] and "diagnos" in res["answer"].lower()


def test_general_info_has_disclaimer():
    res = H.answer("how does exercise help health?")
    assert not res["blocked"] and "not medical advice" in res["answer"].lower()


def test_handler():
    from src import handler as H2

    assert isinstance(H2.respond("how can I stay healthy?", {"config": {}}), list)
