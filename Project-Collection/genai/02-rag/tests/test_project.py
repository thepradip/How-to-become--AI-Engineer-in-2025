import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import rag as R  # noqa: E402


def test_chunk_and_index():
    idx = R.build_index(R.SAMPLE_DOCS)
    assert len(idx.chunks) >= len(R.SAMPLE_DOCS)


def test_retrieve_relevant_chunk():
    idx = R.build_index(R.SAMPLE_DOCS)
    hits = R.retrieve(idx, "how long do refunds take?")
    assert hits and "refund" in hits[0][1].lower()


def test_answer_has_sources():
    idx = R.build_index(R.SAMPLE_DOCS)
    res = R.answer(idx, "what does the premium plan include?")
    assert "answer" in res and isinstance(res["sources"], list) and res["sources"]


def test_handler_question():
    from src import handler as H

    out = H.respond("how long do refunds take?", {"config": {}})
    assert isinstance(out, list)
