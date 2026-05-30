"""Tests for the churn project (multi-algorithm + ensemble).

Uses the offline synthetic sample so tests need no network. The same code path
runs on the real OpenML data when downloaded.
"""

import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))  # Project-Collection root (for _shared)
sys.path.insert(0, str(HERE.parents[1]))  # project root (for `src` package)

from src import model as M  # noqa: E402


def _result():
    return M.train(M.synthetic_churn(500))


def test_synthetic_schema():
    df = M.synthetic_churn(50)
    assert len(df) == 50 and M.TARGET in df.columns
    assert df[M.TARGET].isin(["Yes", "No"]).all()


def test_trains_multiple_algorithms_and_ensemble():
    res = _result()
    # All four candidates (3 base + stacking ensemble) should be trained & compared.
    assert len(res.pipelines) == 4
    assert "Stacking Ensemble" in res.pipelines
    assert len(res.leaderboard) == 4
    assert res.best_name in res.pipelines


def test_leaderboard_metrics_in_range():
    res = _result()
    assert {"Algorithm", "ROC-AUC", "Precision", "Recall"}.issubset(res.leaderboard.columns)
    assert 0.0 <= res.metrics["ROC-AUC"] <= 1.0
    assert res.metrics["ROC-AUC"] > 0.6  # beats random on the separable signal
    assert res.confusion.shape == (2, 2)
    # Leaderboard is sorted best-first.
    assert res.leaderboard.iloc[0]["ROC-AUC"] == res.leaderboard["ROC-AUC"].max()


def test_importances_nonempty():
    imp = _result().importances
    assert len(imp) > 0 and {"feature", "importance"}.issubset(imp.columns)


def test_score_customer_is_probability():
    res = _result()
    assert 0.0 <= M.score_customer(res, res.sample_row) <= 1.0


def test_handler_commands():
    from src import handler as H

    state: dict = {"config": {}}
    assert isinstance(H.respond("train", state), list)
    assert "result" in state
    assert isinstance(H.respond("compare", state), list)
    assert isinstance(H.respond("demo", state), list)
