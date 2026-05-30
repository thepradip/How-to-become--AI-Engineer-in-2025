import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import lstm as L  # noqa: E402


def test_windows_and_forward():
    series = L.synthetic_series(200)
    X, y = L.make_windows(series, 30)
    assert X.shape[1:] == (30, 1) and y.shape[1] == 1
    out = L.LSTMForecaster()(X[:4])
    assert out.shape == (4, 1)


def test_train_reduces_loss():
    series = L.synthetic_series(300)
    X, y = L.make_windows(series, 30)
    hist = L.train(L.LSTMForecaster(), X, y, epochs=15, device=torch.device("cpu"))
    assert hist[-1] <= hist[0]                # loss should not increase overall
    preds = L.forecast(L.LSTMForecaster(), series, 30, steps=20)
    assert preds.shape == (20,)


def test_handler_smoke():
    from src import handler as H

    state = {"config": {"epochs": 3}}
    assert isinstance(H.respond("train", state), list)
