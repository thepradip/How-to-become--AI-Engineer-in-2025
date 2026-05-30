"""Object detection with YOLO11 (Ultralytics).

Detection finds *what* and *where*: bounding boxes + labels. We use the modern YOLO11
family. ``build_arch`` constructs the network from a bundled YAML config offline (for
tests); ``load_pretrained`` downloads COCO-pretrained weights for real detection.
"""

from __future__ import annotations

import numpy as np

ARCH = "yolo11n.yaml"    # architecture only — offline, no weights
WEIGHTS = "yolo11n.pt"   # COCO-pretrained — downloads on first use


def build_arch():
    """Build the YOLO11-n architecture from config (offline, random weights)."""
    from ultralytics import YOLO

    return YOLO(ARCH)


def load_pretrained():
    """Load COCO-pretrained YOLO11-n (downloads weights on first call)."""
    from ultralytics import YOLO

    return YOLO(WEIGHTS)


def run_detection(model, image, conf: float = 0.25):
    """Run detection on a path / array / PIL image.

    Returns (annotated_rgb_uint8, detections_list[{label, confidence}]).
    """
    results = model.predict(image, conf=conf, verbose=False)
    r = results[0]
    annotated = r.plot()                # BGR uint8
    annotated = annotated[:, :, ::-1]   # → RGB
    names = r.names
    dets = [{"label": names[int(c)], "confidence": round(float(p), 3)}
            for c, p in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist())]
    return np.ascontiguousarray(annotated), dets
