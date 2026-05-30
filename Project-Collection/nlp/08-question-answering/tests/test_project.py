import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import qa as Q  # noqa: E402


def test_answer_is_relevant_sentence():
    res = Q.answer("What does RAG reduce?")
    assert "hallucination" in res["answer"].lower()
    assert 0.0 <= res["score"] <= 1.0


def test_answer_from_custom_context():
    ctx = "The Eiffel Tower is in Paris. The Colosseum is in Rome."
    res = Q.answer("Where is the Colosseum?", ctx)
    assert "rome" in res["answer"].lower()


def test_handler_question():
    from src import handler as H

    out = H.respond("what is the dominant pattern in 2026?", {"config": {}})
    assert isinstance(out, list)
