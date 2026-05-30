"""Face-verification brain. Uses the shared chat UI."""

from __future__ import annotations

from . import face as Fc


def respond(message: str, state: dict):
    import torch

    from _shared.chat_ui import Reply
    from _shared.torch_utils import get_device, train_classifier

    msg = message.strip().lower()
    cfg = state.get("config", {})

    if "train" in msg or msg in {"go", "start"}:
        device = get_device()
        ld = Fc.loader(n_ids=10, per_id=30)
        model = Fc.FaceNet(n_ids=10)
        hist = train_classifier(model, ld, epochs=int(cfg.get("epochs", 10)), device=device)
        state["model"] = model
        state["data"] = Fc.synthetic_identities(10, 30)
        return [
            Reply(f"Trained a face-embedding network on **{device}** (final loss {hist[-1]:.3f})."),
            Reply("Type `verify` to compare same-person vs different-person pairs.", "text"),
        ]

    if "verify" in msg or "demo" in msg or "compare" in msg:
        if "model" not in state:
            return Reply("Train first — type `train`.")
        import torch

        model = state["model"]
        X, y = state["data"]
        # same-identity pair (label 0) and different-identity pair (labels 0 vs 1)
        idx0 = (y == 0).nonzero().flatten()
        idx1 = (y == 1).nonzero().flatten()
        same_ok, same_sim = Fc.verify(model, X[idx0[0]:idx0[0] + 1], X[idx0[1]:idx0[1] + 1])
        diff_ok, diff_sim = Fc.verify(model, X[idx0[0]:idx0[0] + 1], X[idx1[0]:idx1[0] + 1])
        return [
            Reply("**Cosine similarity** of face embeddings (higher = more likely same person):"),
            Reply({"Same person": f"{same_sim:.2f} → {'MATCH' if same_ok else 'no'}",
                   "Different people": f"{diff_sim:.2f} → {'MATCH' if diff_ok else 'no'}"}, "metric"),
        ]

    return Reply("I verify whether two faces are the **same person** via embedding similarity. "
                 "Try `train`, then `verify`.")
