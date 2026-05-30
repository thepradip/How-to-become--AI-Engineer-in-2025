import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import stocks as K  # noqa: E402


def test_features_and_target():
    feat = K.make_features(K.synthetic_prices(400))
    assert set(K.FEATS + [K.TARGET]).issubset(feat.columns)
    assert feat[K.TARGET].isin([0, 1]).all()


def test_train_and_backtest():
    res = K.train_and_backtest(K.synthetic_prices(800))
    assert len(res.leaderboard) == 3
    assert res.best_model in set(res.leaderboard["model"])
    assert {"strategy", "buy_hold"}.issubset(res.backtest.columns)
    assert len(res.backtest) > 0
    assert "Strategy return" in res.stats


def test_handler_commands():
    from src import handler as H

    state: dict = {"config": {}}
    assert isinstance(H.respond("analyze", state), list)
