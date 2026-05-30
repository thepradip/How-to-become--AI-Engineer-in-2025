import pathlib
import sys

import pytest

pytest.importorskip("PIL")

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import videogen as V  # noqa: E402


def test_frames_shape():
    frames = V.generate_frames("waves", n_frames=8, size=64)
    assert frames.shape == (8, 64, 64, 3)


def test_to_gif_returns_bytes():
    frames = V.generate_frames("waves", n_frames=6, size=48)
    gif = V.to_gif(frames)
    assert isinstance(gif, bytes) and gif[:3] == b"GIF"


def test_handler_demo():
    from src import handler as H

    assert isinstance(H.respond("demo", {"config": {"frames": 6}}), list)
