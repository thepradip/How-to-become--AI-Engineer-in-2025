import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import transfer as T  # noqa: E402


def test_model_head_matches_classes():
    model = T.build_model(n_classes=7, pretrained=False)
    out = model(torch.rand(2, 3, 64, 64))
    assert out.shape == (2, 7)


def test_trains_one_step():
    from _shared.torch_utils import train_classifier

    loader = T.synthetic_loader(64, n_classes=5, batch=16)
    hist = train_classifier(T.build_model(5), loader, epochs=1, device=torch.device("cpu"))
    assert len(hist) > 0


def test_handler_smoke():
    from src import handler as H

    state = {"config": {"max_batches": 2, "epochs": 1, "n_classes": 5}}
    assert isinstance(H.respond("train", state), list) and "model" in state
