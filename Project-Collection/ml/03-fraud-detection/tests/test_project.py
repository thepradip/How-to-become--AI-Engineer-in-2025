import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from _shared.tabular import train_and_compare  # noqa: E402
from src.data_loader import POSITIVE, TARGET, synthetic  # noqa: E402


def test_imbalanced_dataset():
    df = synthetic(4000)
    rate = (df[TARGET] == 1).mean()
    assert 0.0 < rate < 0.1  # genuinely imbalanced


def test_multi_algorithm_with_smote():
    res = train_and_compare(synthetic(4000), TARGET, "classification",
                            positive=POSITIVE, use_smote=True)
    assert len(res.pipelines) == 4 and "Stacking Ensemble" in res.pipelines
    assert res.confusion.shape == (2, 2)
    # The fraud signal is learnable → better than chance.
    assert res.metrics.get("ROC-AUC", 0) > 0.7


def test_handler_commands():
    from src import handler as H

    state: dict = {"config": {}}
    assert isinstance(H.respond("train", state), list)
    assert isinstance(H.respond("demo", state), list)
