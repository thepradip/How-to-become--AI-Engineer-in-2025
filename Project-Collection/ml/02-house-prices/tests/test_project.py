import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from _shared.tabular import train_and_compare  # noqa: E402
from src.data_loader import TARGET, synthetic  # noqa: E402


def test_regression_multi_algorithm():
    res = train_and_compare(synthetic(700), TARGET, task="regression")
    assert len(res.pipelines) == 4 and "Stacking Ensemble" in res.pipelines
    assert {"R²", "MAE", "RMSE"}.issubset(res.leaderboard.columns)
    assert res.metrics["R²"] > 0.7          # learns the synthetic price signal
    assert len(res.importances) > 0


def test_handler_commands():
    from src import handler as H

    state: dict = {"config": {}}
    assert isinstance(H.respond("train", state), list)
    assert isinstance(H.respond("demo", state), list)
