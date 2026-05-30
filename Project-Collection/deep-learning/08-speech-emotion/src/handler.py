"""Speech-emotion brain. Uses the shared chat UI."""

from __future__ import annotations

import torch

from . import audio as A


def respond(message: str, state: dict):
    from _shared.chat_ui import Reply
    from _shared.torch_utils import accuracy, get_device, train_classifier

    msg = message.strip().lower()
    cfg = state.get("config", {})

    if "train" in msg or msg in {"go", "start"}:
        device = get_device()
        ld = A.loader(per_class=40)
        model = A.EmotionMLP()
        hist = train_classifier(model, ld, epochs=int(cfg.get("epochs", 8)), device=device)
        acc = accuracy(model, ld, device=device)
        state["model"] = model
        return [
            Reply(f"Trained an emotion classifier ({len(A.EMOTIONS)} emotions) on **{device}**."),
            Reply({"Train accuracy": f"{acc:.1%}", "Final loss": f"{hist[-1]:.3f}"}, "metric"),
            Reply("Type `demo` to classify an example clip. Install librosa + RAVDESS for real audio (README).", "text"),
        ]

    if "demo" in msg or "predict" in msg:
        if "model" not in state:
            return Reply("Train first — type `train`.")
        import numpy as np

        true = 2  # "sad" example
        feat = torch.tensor(A.extract_features(A.synthetic_waveform(true)), dtype=torch.float32)
        with torch.no_grad():
            pred = int(state["model"].cpu()(feat.unsqueeze(0)).argmax(1))
        return [Reply({"Predicted emotion": A.EMOTIONS[pred], "Actual": A.EMOTIONS[true]}, "metric")]

    return Reply("I recognise **speech emotion** from audio features. Try `train`, then `demo`.")
