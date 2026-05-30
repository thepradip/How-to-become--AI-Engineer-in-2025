"""1-bit / ternary weight quantization — the idea behind BitNet b1.58 & Bonsai.

1-bit LLMs store each weight as {-1, 0, 1} (1.58-bit) or true 1-bit, shrinking models
10–16× so they run on a phone/CPU. This module demonstrates the *core operation*
(absmean ternary quantization) on a weight matrix and measures the size reduction and
reconstruction error — then the README shows how to run a real BitNet/Bonsai model.
"""

from __future__ import annotations

import numpy as np


def ternary_quantize(weights: np.ndarray) -> tuple[np.ndarray, float]:
    """BitNet-style absmean quantization → values in {-1,0,1} plus a scale.

    Returns (ternary_matrix, scale) such that ``scale * ternary ≈ weights``.
    """
    scale = float(np.mean(np.abs(weights))) + 1e-8
    q = np.round(weights / scale)
    q = np.clip(q, -1, 1)
    return q.astype(np.int8), scale


def dequantize(q: np.ndarray, scale: float) -> np.ndarray:
    return q.astype(np.float32) * scale


def report(weights: np.ndarray) -> dict:
    """Quantize and summarise: size reduction + reconstruction error."""
    q, scale = ternary_quantize(weights)
    recon = dequantize(q, scale)
    fp32_bits = weights.size * 32
    ternary_bits = weights.size * 1.58            # log2(3) ≈ 1.58 bits per weight
    err = float(np.sqrt(np.mean((weights - recon) ** 2)))
    nonzero = float(np.mean(q != 0))
    return {
        "params": int(weights.size),
        "fp32_size_kb": round(fp32_bits / 8 / 1024, 1),
        "ternary_size_kb": round(ternary_bits / 8 / 1024, 1),
        "compression": f"{fp32_bits / ternary_bits:.1f}×",
        "rmse": round(err, 4),
        "nonzero_fraction": round(nonzero, 3),
    }
