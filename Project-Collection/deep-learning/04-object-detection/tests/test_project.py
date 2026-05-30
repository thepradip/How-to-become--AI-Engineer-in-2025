import pathlib
import sys

import pytest

pytest.importorskip("torch")
pytest.importorskip("ultralytics")

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import detect as D  # noqa: E402


def test_builds_architecture_offline():
    # Build from the bundled YAML config — no network / no pretrained weights.
    model = D.build_arch()
    assert model is not None and hasattr(model, "predict")


def test_detection_on_random_image():
    # Architecture (random weights) still runs the forward path on an array.
    import numpy as np

    model = D.build_arch()
    annotated, dets = D.run_detection(model, np.zeros((64, 64, 3), dtype="uint8"))
    assert annotated.ndim == 3 and annotated.shape[-1] == 3
    assert isinstance(dets, list)
