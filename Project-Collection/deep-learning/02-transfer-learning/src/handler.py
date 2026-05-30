"""Transfer-learning brain. Uses the shared chat UI."""

from __future__ import annotations

import itertools

from . import transfer as T


def respond(message: str, state: dict):
    import torch  # noqa: F401

    from _shared.chat_ui import Reply
    from _shared.torch_utils import accuracy, get_device, train_classifier

    msg = message.strip().lower()
    cfg = state.get("config", {})
    n_classes = int(cfg.get("n_classes", 5))

    if "train" in msg or msg in {"go", "start"}:
        device = get_device()
        loader = T.synthetic_loader(128, n_classes)        # offline demo / smoke
        model = T.build_model(n_classes, pretrained=bool(cfg.get("pretrained", False)))
        hist = train_classifier(model, list(itertools.islice(loader, int(cfg.get("max_batches", 4)))),
                                epochs=int(cfg.get("epochs", 1)), lr=1e-3, device=device)
        acc = accuracy(model, loader, device=device)
        state["model"] = model
        return [
            Reply(f"Fine-tuned a ResNet-18 head ({n_classes} classes) on **{device}**."),
            Reply({"Train accuracy": f"{acc:.1%}", "Final loss": f"{hist[-1]:.3f}"}, "metric"),
            Reply("Point `real_loader('path/to/images')` at your own ImageFolder for real training "
                  "(see README).", "text"),
        ]

    return Reply("I **fine-tune a pretrained ResNet-18** on your image classes. Try `train`. "
                 "For real data, use `src.transfer.real_loader('your_image_folder')`.")
