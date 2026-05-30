"""Pneumonia X-ray brain with Grad-CAM. Uses the shared chat UI."""

from __future__ import annotations

import itertools

from . import xray as X


def respond(message: str, state: dict):
    import torch

    from _shared.chat_ui import Reply
    from _shared.torch_utils import accuracy, get_device, train_classifier

    msg = message.strip().lower()
    cfg = state.get("config", {})

    if "train" in msg or msg in {"go", "start"}:
        device = get_device()
        loader = X.synthetic_loader(128)
        model = X.build_model(pretrained=bool(cfg.get("pretrained", False)))
        hist = train_classifier(model, list(itertools.islice(loader, int(cfg.get("max_batches", 4)))),
                                epochs=int(cfg.get("epochs", 1)), device=device)
        acc = accuracy(model, loader, device=device)
        state["model"] = model
        state["loader"] = loader
        return [
            Reply(f"Fine-tuned a ResNet-18 X-ray classifier on **{device}**."),
            Reply({"Train accuracy": f"{acc:.1%}", "Final loss": f"{hist[-1]:.3f}"}, "metric"),
            Reply("Type `explain` to see a **Grad-CAM** heatmap on an example X-ray.", "text"),
        ]

    if "explain" in msg or "gradcam" in msg or "grad-cam" in msg or "demo" in msg:
        if "model" not in state:
            return Reply("Train first — type `train`.")
        import torch

        model, loader = state["model"], state["loader"]
        xb, yb = next(iter(loader))
        image = xb[0:1]
        cam, cls = X.grad_cam(model, image)
        return [
            Reply("**Grad-CAM** — brighter = more influential for the prediction:"),
            Reply(X.overlay(image, cam), "image"),
            Reply({"Predicted": X.CLASSES[cls], "Actual": X.CLASSES[int(yb[0])]}, "metric"),
        ]

    return Reply("I detect **pneumonia** in chest X-rays and explain decisions with **Grad-CAM**. "
                 "Try `train`, then `explain`.")
