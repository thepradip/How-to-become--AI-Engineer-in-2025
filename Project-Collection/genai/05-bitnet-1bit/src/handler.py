"""1-bit quantization brain. Uses the shared chat UI."""

from __future__ import annotations

import numpy as np

from . import quant as Q


def respond(message: str, state: dict):
    from _shared.chat_ui import Reply

    rng = np.random.default_rng(0)
    size = int(state.get("config", {}).get("size", 1024))
    W = rng.normal(0, 1, (size, size)).astype("float32")
    rep = Q.report(W)
    return [
        Reply(f"Ternary-quantized a {size}×{size} weight matrix (BitNet-style absmean):"),
        Reply(rep, "metric"),
        Reply("Each weight becomes {-1, 0, 1} (1.58-bit) → big memory savings with small error. "
              "Run a real **BitNet b1.58** or **Bonsai** model per the README.", "text"),
    ]
