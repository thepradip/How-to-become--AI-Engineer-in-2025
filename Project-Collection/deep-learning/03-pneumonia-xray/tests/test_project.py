import pathlib
import sys

import pytest

torch = pytest.importorskip("torch")

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import xray as X  # noqa: E402


def test_model_binary_output():
    out = X.build_model(pretrained=False)(torch.rand(2, 3, 64, 64))
    assert out.shape == (2, 2)


def test_grad_cam_shape_and_range():
    model = X.build_model(pretrained=False)
    cam, cls = X.grad_cam(model, torch.rand(1, 3, 64, 64))
    assert cam.ndim == 2
    assert 0.0 <= float(cam.min()) and float(cam.max()) <= 1.0 + 1e-6
    assert cls in (0, 1)


def test_overlay_is_rgb_image():
    img = torch.rand(1, 3, 64, 64)
    cam, _ = X.grad_cam(X.build_model(), img)
    rgb = X.overlay(img, cam)
    assert rgb.shape[-1] == 3 and rgb.dtype.name == "uint8"


def test_handler_smoke():
    from src import handler as H

    state = {"config": {"max_batches": 2, "epochs": 1}}
    assert isinstance(H.respond("train", state), list)
    assert isinstance(H.respond("explain", state), list)
