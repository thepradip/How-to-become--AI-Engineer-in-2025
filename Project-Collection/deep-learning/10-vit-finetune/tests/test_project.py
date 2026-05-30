import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import vit as V  # noqa: E402


def test_vit_head_matches_classes():
    model = V.build_vit(n_classes=7, pretrained=False)
    out = model(torch.rand(2, 3, 224, 224))
    assert out.shape == (2, 7)


def test_train_one_step():
    from _shared.torch_utils import train_classifier

    hist = train_classifier(V.build_vit(5), V.synthetic_loader(n=8, batch=4),
                            epochs=1, device=torch.device("cpu"))
    assert len(hist) > 0


def test_handler_smoke():
    from src import handler as H

    state = {"config": {"epochs": 1, "n_classes": 5}}
    assert isinstance(H.respond("train", state), list) and "model" in state
