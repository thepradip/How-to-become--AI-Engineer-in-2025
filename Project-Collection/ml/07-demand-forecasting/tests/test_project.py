import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import forecasting as F  # noqa: E402


def test_features_have_lags():
    feat = F._features(F.synthetic_series(200))
    assert {"lag_1", "lag_7", "roll_7", "dayofweek"}.issubset(feat.columns)


def test_compares_models_and_metrics():
    res = F.forecast(F.synthetic_series(400), horizon=28)
    assert {"model", "MAE", "RMSE", "MAPE%"}.issubset(res.leaderboard.columns)
    assert len(res.leaderboard) == 4               # naive + 3 ML models
    assert res.best_model in set(res.leaderboard["model"])
    assert len(res.forecast) == 28
    # Best model's errors are finite and non-negative.
    assert res.leaderboard.iloc[0]["RMSE"] >= 0


def test_handler_commands():
    from src import handler as H

    state: dict = {"config": {}}
    assert isinstance(H.respond("forecast", state), list)
    assert isinstance(H.respond("compare", state), list)
