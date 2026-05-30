"""Traffic-sign CNN brain. Uses the shared chat UI."""

from __future__ import annotations

import itertools

from . import model as M


def respond(message: str, state: dict):
    import torch  # noqa: F401

    from _shared.chat_ui import Reply
    from _shared.torch_utils import accuracy, get_device, train_classifier

    msg = message.strip().lower()
    cfg = state.get("config", {})

    if "train" in msg or msg in {"go", "start"}:
        device = get_device()
        loader = M.synthetic_loader(256)
        model = M.TrafficCNN()
        hist = train_classifier(model, list(itertools.islice(loader, int(cfg.get("max_batches", 4)))),
                                epochs=int(cfg.get("epochs", 1)), device=device)
        acc = accuracy(model, loader, device=device)
        state["model"] = model
        return [
            Reply(f"Trained a traffic-sign CNN (43 classes) on **{device}**."),
            Reply({"Train accuracy": f"{acc:.1%}", "Final loss": f"{hist[-1]:.3f}"}, "metric"),
            Reply("This is a synthetic-data demo; see the README to train on real GTSRB images.", "text"),
        ]

    return Reply("I classify **traffic signs** (GTSRB, 43 classes) with a CNN. Try `train`.")
