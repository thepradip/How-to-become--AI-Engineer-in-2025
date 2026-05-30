import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import model as M  # noqa: E402


def test_forward_shape():
    out = M.TrafficCNN(n_classes=43)(torch.rand(2, 3, 32, 32))
    assert out.shape == (2, 43)


def test_train_step():
    from _shared.torch_utils import train_classifier

    hist = train_classifier(M.TrafficCNN(), M.synthetic_loader(64, batch=16),
                            epochs=1, device=torch.device("cpu"))
    assert len(hist) > 0


def test_handler_smoke():
    from src import handler as H

    state = {"config": {"max_batches": 2, "epochs": 1}}
    assert isinstance(H.respond("train", state), list) and "model" in state
