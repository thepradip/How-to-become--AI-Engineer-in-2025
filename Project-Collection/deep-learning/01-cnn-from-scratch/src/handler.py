"""Fashion-MNIST CNN brain. Uses the shared chat UI.

For a snappy demo the chat trains briefly (a few hundred images, 1 epoch). The README
shows how to run full training for higher accuracy. Smoke tests use synthetic data.
"""

from __future__ import annotations

import itertools

from . import cnn as C


def _loaders(state: dict):
    """Real Fashion-MNIST unless config asks for synthetic (tests) or download fails."""
    cfg = state.get("config", {})
    if cfg.get("use_synthetic"):
        return C.synthetic_loader(256), C.synthetic_loader(128, seed=1)
    try:
        return C.real_loaders(batch=128)
    except Exception:
        return C.synthetic_loader(256), C.synthetic_loader(128, seed=1)


def respond(message: str, state: dict):
    import torch  # noqa: F401

    from _shared.chat_ui import Reply
    from _shared.torch_utils import accuracy, get_device, train_classifier

    msg = message.strip().lower()
    cfg = state.get("config", {})
    epochs = int(cfg.get("epochs", 1))
    max_batches = int(cfg.get("max_batches", 60))

    if "train" in msg or msg in {"go", "start"}:
        train_loader, test_loader = _loaders(state)
        # Cap batches so the in-chat demo stays fast on CPU.
        capped = list(itertools.islice(train_loader, max_batches))
        device = get_device()
        model = C.SimpleCNN()
        hist = train_classifier(model, capped, epochs=epochs, lr=1e-3, device=device)
        acc = accuracy(model, list(itertools.islice(test_loader, 20)), device=device)
        state["model"] = model
        state["test_loader"] = test_loader
        return [
            Reply(f"Trained a from-scratch CNN ({epochs} epoch, {len(capped)} batches) on **{device}**."),
            Reply({"Test accuracy (subset)": f"{acc:.1%}", "Final loss": f"{hist[-1]:.3f}"}, "metric"),
            Reply(_loss_fig(hist), "figure"),
            Reply("Type `demo` to classify an example image. Full training instructions are in the README.", "text"),
        ]

    if "demo" in msg or "predict" in msg:
        if "model" not in state:
            return Reply("Train first — type `train`.")
        return _demo(state)

    return Reply("I train a **CNN from scratch** on Fashion-MNIST. Try `train`, then `demo`.")


def _demo(state):
    import torch

    from _shared.chat_ui import Reply
    from _shared.torch_utils import get_device

    model, loader = state["model"], state["test_loader"]
    xb, yb = next(iter(loader))
    img, true = xb[0:1], int(yb[0])
    with torch.no_grad():
        pred = int(model.to(get_device())(img.to(get_device())).argmax(1))
    arr = (img.squeeze().cpu().numpy() * 255).astype("uint8")
    return [
        Reply(arr, "image", {"full_width": False}),
        Reply({"Predicted": C.LABELS[pred], "Actual": C.LABELS[true]}, "metric"),
    ]


def _loss_fig(hist):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(hist, color="#38bdf8")
    ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_title("Training loss"); fig.tight_layout()
    return fig
