import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))  # Project-Collection root
sys.path.insert(0, str(HERE.parents[1]))  # project root

from _shared.tabular import train_and_compare  # noqa: E402
from src.data_loader import POSITIVE, TARGET, load  # noqa: E402


def test_loads_real_data():
    df = load()
    assert len(df) > 500 and TARGET in df.columns
    assert set(df[TARGET].unique()) == {"malignant", "benign"}


def test_multi_algorithm_and_ensemble():
    res = train_and_compare(load(), TARGET, "classification", positive=POSITIVE)
    assert len(res.pipelines) == 4 and "Stacking Ensemble" in res.pipelines
    # On this well-separated medical dataset, ROC-AUC should be strong.
    assert res.metrics["ROC-AUC"] > 0.95
    assert res.confusion.shape == (2, 2)
    assert len(res.importances) > 0


def test_handler_commands():
    from src import handler as H

    state: dict = {"config": {}}
    assert isinstance(H.respond("train", state), list)
    assert isinstance(H.respond("demo", state), list)
