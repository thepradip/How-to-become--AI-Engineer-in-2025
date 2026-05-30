import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE.parents[1]))

from src import quant as Q  # noqa: E402


def test_ternary_values():
    W = np.random.default_rng(0).normal(0, 1, (64, 64)).astype("float32")
    q, scale = Q.ternary_quantize(W)
    assert set(np.unique(q)).issubset({-1, 0, 1})
    assert scale > 0


def test_report_shows_compression():
    W = np.random.default_rng(0).normal(0, 1, (128, 128)).astype("float32")
    rep = Q.report(W)
    assert rep["params"] == 128 * 128
    assert "×" in rep["compression"]
    assert rep["ternary_size_kb"] < rep["fp32_size_kb"]   # smaller after quantization


def test_handler_demo():
    from src import handler as H

    assert isinstance(H.respond("demo", {"config": {"size": 128}}), list)
