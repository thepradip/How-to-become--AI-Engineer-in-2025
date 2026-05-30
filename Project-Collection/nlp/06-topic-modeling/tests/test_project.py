import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import topics as T  # noqa: E402


def test_nmf_finds_topics():
    res = T.fit_topics(T.synthetic_docs(200), n_topics=4, method="NMF")
    assert len(res.topics) == 4
    assert "top_words" in res.topics.columns
    assert len(res.doc_topics) == 200


def test_lda_runs():
    res = T.fit_topics(T.synthetic_docs(150), n_topics=4, method="LDA")
    assert len(res.topics) == 4


def test_handler_discover():
    from src import handler as H

    out = H.respond("discover", {"config": {}})
    assert isinstance(out, list)
