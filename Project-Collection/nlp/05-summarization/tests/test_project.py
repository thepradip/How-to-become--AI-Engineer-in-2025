import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import summarize as S  # noqa: E402

TXT = ("AI is changing software. Models can write code and tests. They also make mistakes. "
       "Good engineers review the output. Tooling keeps improving every month. The future is hybrid.")


def test_extractive_returns_subset_of_sentences():
    summary = S.extractive_summary(TXT, 2)
    sents = S.split_sentences(summary)
    assert 1 <= len(sents) <= 2
    originals = set(S.split_sentences(TXT))
    assert all(s in originals for s in sents)        # extractive → exact sentences


def test_short_text_returned_asis():
    assert S.extractive_summary("Only one sentence here.", 3).startswith("Only one")


def test_handler_demo():
    from src import handler as H

    out = H.respond("demo", {"config": {"n_sentences": 2}})
    assert isinstance(out, list)
