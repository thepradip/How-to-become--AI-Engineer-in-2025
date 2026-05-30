import pathlib
import sys

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import imagegen as I  # noqa: E402


def test_generates_rgb_image():
    img = I.generate("a cat", size=128)
    assert img.shape == (128, 128, 3) and img.dtype.name == "uint8"


def test_deterministic_for_same_prompt():
    import numpy as np

    assert np.array_equal(I.generate("same", 64), I.generate("same", 64))
    assert not np.array_equal(I.generate("a", 64), I.generate("b", 64))


def test_handler_demo():
    from src import handler as H

    assert isinstance(H.respond("demo", {"config": {"size": 64}}), list)
