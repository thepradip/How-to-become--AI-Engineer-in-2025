import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from _shared.tabular import train_and_compare  # noqa: E402
from src.data_loader import POSITIVE, TARGET, synthetic  # noqa: E402


def test_classification_multi_algorithm():
    res = train_and_compare(synthetic(1000), TARGET, "classification", positive=POSITIVE)
    assert len(res.pipelines) == 4 and "Stacking Ensemble" in res.pipelines
    assert 0.0 <= res.metrics["ROC-AUC"] <= 1.0
    assert res.confusion.shape == (2, 2)


def test_handler_commands():
    from src import handler as H

    state: dict = {"config": {}}
    assert isinstance(H.respond("train", state), list)
    assert isinstance(H.respond("compare", state), list)
