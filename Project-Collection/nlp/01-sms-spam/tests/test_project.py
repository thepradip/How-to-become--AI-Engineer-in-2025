import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from _shared.text import train_text_classifiers  # noqa: E402
from src.data_loader import LABEL, TEXT, synthetic  # noqa: E402


def test_compares_classifiers():
    df = synthetic(600)
    res = train_text_classifiers(df[TEXT], df[LABEL])
    assert len(res.leaderboard) == 3
    assert res.leaderboard.iloc[0]["F1 (macro)"] > 0.7   # separable vocab
    assert set(res.classes) == {"spam", "ham"}


def test_handler_classifies_typed_text():
    from src import handler as H

    state: dict = {"config": {}}
    H.respond("train", state)
    out = H.respond("WIN a FREE prize, call now to claim!", state)
    assert isinstance(out, list)
