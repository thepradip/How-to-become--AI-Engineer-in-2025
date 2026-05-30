"""YOLO object-detection brain. Uses the shared chat UI (with file upload)."""

from __future__ import annotations

from . import detect as D


def _model(state):
    if "model" not in state:
        state["model"] = D.load_pretrained()  # downloads COCO weights on first use
    return state["model"]


def respond(message: str, state: dict):
    import numpy as np
    from PIL import Image

    from _shared.chat_ui import Reply

    up = state.get("uploaded_file")
    msg = message.strip().lower()

    # If the user uploaded an image, detect on it.
    if up is not None:
        try:
            img = np.array(Image.open(up).convert("RGB"))
            annotated, dets = D.run_detection(_model(state), img)
            parts = [Reply("**Detections:**", "text"), Reply(annotated, "image", {"full_width": True})]
            if dets:
                import pandas as pd
                parts.append(Reply(pd.DataFrame(dets), "table"))
            else:
                parts.append(Reply("No objects above the confidence threshold.", "text"))
            return parts
        except Exception as exc:
            return Reply(f"Couldn't process the image: {exc}", "error")

    if "demo" in msg:
        try:
            annotated, dets = D.run_detection(_model(state), "https://ultralytics.com/images/bus.jpg")
            import pandas as pd
            return [Reply("Detections on the sample image:", "text"),
                    Reply(annotated, "image", {"full_width": True}),
                    Reply(pd.DataFrame(dets), "table")]
        except Exception:
            return Reply("Demo needs internet for the sample image. **Upload an image** in the sidebar instead.")

    return Reply("I detect objects with **YOLO11**. **Upload an image** in the sidebar, or type `demo` "
                 "to run on a sample image (needs internet for first-time weight download).")
