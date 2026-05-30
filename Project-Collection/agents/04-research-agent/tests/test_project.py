import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import research as R  # noqa: E402


def test_search_relevant():
    hits = R.search("how is RAG evaluated?")
    assert hits and any("faithfulness" in h["text"].lower() or "evaluat" in h["text"].lower() for h in hits)


def test_run_has_summary_and_sources():
    res = R.run("vector database for RAG")
    assert res["summary"] and res["sources"]
    assert all("title" in s for s in res["sources"])


def test_handler():
    from src import handler as H

    assert isinstance(H.respond("agents frameworks", {"config": {}}), list)
