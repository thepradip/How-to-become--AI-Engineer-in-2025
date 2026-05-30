import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import unet as U  # noqa: E402


def test_unet_output_shape():
    out = U.UNet(n_classes=2)(torch.rand(2, 3, 64, 64))
    assert out.shape == (2, 2, 64, 64)


def test_train_step_and_predict():
    ld = U.loader(32, batch=8)
    hist = U.train_seg(U.UNet(), ld, epochs=1, device=torch.device("cpu"))
    assert len(hist) > 0
    xb, _ = next(iter(ld))
    mask = U.predict_mask(U.UNet(), xb[0:1])
    assert mask.shape == (64, 64)


def test_handler_smoke():
    from src import handler as H

    state = {"config": {"max_batches": 2, "epochs": 1}}
    assert isinstance(H.respond("train", state), list)
    assert isinstance(H.respond("demo", state), list)
