import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")  # skip cleanly where torch isn't installed

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import cnn as C  # noqa: E402


def test_model_forward_shape():
    model = C.SimpleCNN()
    out = model(torch.rand(4, 1, 28, 28))
    assert out.shape == (4, 10)


def test_trains_one_step_on_synthetic():
    from _shared.torch_utils import accuracy, train_classifier

    loader = C.synthetic_loader(128, batch=32)
    hist = train_classifier(C.SimpleCNN(), loader, epochs=1, lr=1e-3, device=torch.device("cpu"))
    assert len(hist) > 0 and all(h >= 0 for h in hist)
    acc = accuracy(C.SimpleCNN(), loader, device=torch.device("cpu"))
    assert 0.0 <= acc <= 1.0


def test_handler_train_smoke():
    from src import handler as H

    state = {"config": {"use_synthetic": True, "epochs": 1, "max_batches": 2}}
    out = H.respond("train", state)
    assert isinstance(out, list) and "model" in state
    assert isinstance(H.respond("demo", state), list)
