"""U-Net segmentation brain. Uses the shared chat UI."""

from __future__ import annotations

import itertools

import numpy as np

from . import unet as U


def respond(message: str, state: dict):
    import torch

    from _shared.chat_ui import Reply
    from _shared.torch_utils import get_device

    msg = message.strip().lower()
    cfg = state.get("config", {})

    if "train" in msg or msg in {"go", "start"}:
        device = get_device()
        ld = U.loader(96)
        model = U.UNet()
        hist = U.train_seg(model, list(itertools.islice(ld, int(cfg.get("max_batches", 6)))),
                           epochs=int(cfg.get("epochs", 2)), device=device)
        state["model"] = model
        state["loader"] = ld
        return [
            Reply(f"Trained a U-Net for pixel segmentation on **{device}**."),
            Reply({"Final loss": f"{hist[-1]:.3f}"}, "metric"),
            Reply("Type `demo` to see image vs. predicted mask.", "text"),
        ]

    if "demo" in msg or "predict" in msg or "mask" in msg:
        if "model" not in state:
            return Reply("Train first — type `train`.")
        import torch

        model, ld = state["model"], state["loader"]
        xb, yb = next(iter(ld))
        image = xb[0:1]
        mask = U.predict_mask(model, image, device=get_device())
        img = (image.squeeze()[0].cpu().numpy() * 255).astype("uint8")
        side = np.concatenate([np.stack([img] * 3, -1),
                               (np.stack([mask] * 3, -1) * 255).astype("uint8")], axis=1)
        return [Reply("**Left:** input · **Right:** predicted mask", "text"),
                Reply(side, "image", {"full_width": True})]

    return Reply("I segment images pixel-by-pixel with a **U-Net**. Try `train`, then `demo`.")
