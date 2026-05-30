"""ViT fine-tune brain. Uses the shared chat UI."""

from __future__ import annotations

from . import vit as V


def respond(message: str, state: dict):
    import torch  # noqa: F401

    from _shared.chat_ui import Reply
    from _shared.torch_utils import accuracy, get_device, train_classifier

    msg = message.strip().lower()
    cfg = state.get("config", {})

    if "train" in msg or "finetune" in msg or msg in {"go", "start"}:
        device = get_device()
        n_classes = int(cfg.get("n_classes", 5))
        loader = V.synthetic_loader(n=8, n_classes=n_classes, batch=4)
        run = V.maybe_wandb({"lr": 1e-3, "n_classes": n_classes})
        model = V.build_vit(n_classes, pretrained=bool(cfg.get("pretrained", False)))
        hist = train_classifier(model, loader, epochs=int(cfg.get("epochs", 1)), lr=1e-3, device=device)
        acc = accuracy(model, loader, device=device)
        if run is not None:
            run.log({"loss": hist[-1], "acc": acc}); run.finish()
        state["model"] = model
        return [
            Reply(f"Fine-tuned **ViT-B/16** head ({n_classes} classes) on **{device}**. "
                  "(Synthetic demo — see README for real data + W&B sweeps.)"),
            Reply({"Train accuracy": f"{acc:.1%}", "Final loss": f"{hist[-1]:.3f}",
                   "W&B": "logged (offline)" if run is not None else "not installed"}, "metric"),
        ]

    return Reply("I fine-tune a **Vision Transformer** and can log/sweep with **W&B**. Try `train`. "
                 "See the README for `wandb sweep` instructions.")
