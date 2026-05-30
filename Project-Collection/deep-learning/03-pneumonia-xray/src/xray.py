"""Pneumonia detection from chest X-rays, with Grad-CAM explanations.

A fine-tuned ResNet-18 classifies an X-ray as NORMAL vs PNEUMONIA. Crucially for
medical AI, we add **Grad-CAM** — a heatmap of where the model "looked" — so a
clinician can sanity-check that it focuses on the lungs, not artefacts.

Real data: the Kaggle Chest X-Ray Pneumonia dataset (folders NORMAL/PNEUMONIA).
Smoke tests use synthetic 3×64×64 tensors and ``weights=None`` (offline).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

CLASSES = ["NORMAL", "PNEUMONIA"]


def build_model(pretrained: bool = False) -> nn.Module:
    from torchvision import models

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


def synthetic_loader(n: int = 128, batch: int = 32, seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, 3, 64, 64, generator=g)
    y = torch.randint(0, 2, (n,), generator=g)
    return DataLoader(TensorDataset(X, y), batch_size=batch, shuffle=True)


def grad_cam(model: nn.Module, image: torch.Tensor, target_class: int | None = None):
    """Return (heatmap[H,W] in 0..1, predicted_class) for a single image (1,3,H,W)."""
    model.eval()
    image = image.to(next(model.parameters()).device)  # match model device (CPU/MPS/CUDA)
    layer = model.layer4[-1]
    store: dict = {}
    h1 = layer.register_forward_hook(lambda m, i, o: store.__setitem__("act", o))
    h2 = layer.register_full_backward_hook(lambda m, gi, go: store.__setitem__("grad", go[0].detach()))
    try:
        out = model(image)
        cls = int(out.argmax(1)) if target_class is None else target_class
        model.zero_grad()
        out[0, cls].backward()
        weights = store["grad"].mean(dim=(2, 3), keepdim=True)        # GAP over spatial dims
        cam = (weights * store["act"].detach()).sum(1).relu().squeeze()
        cam = cam / (cam.max() + 1e-8)
        return cam.cpu().numpy(), cls
    finally:
        h1.remove()
        h2.remove()


def overlay(image: torch.Tensor, cam: np.ndarray) -> np.ndarray:
    """Blend a Grad-CAM heatmap onto the grayscale X-ray → RGB uint8 image."""
    import matplotlib.cm as cm
    from PIL import Image

    base = (image.squeeze()[0].cpu().numpy() * 255).astype("uint8")
    h, w = base.shape
    heat = np.array(Image.fromarray((cam * 255).astype("uint8")).resize((w, h)))
    heat_rgb = (cm.jet(heat / 255.0)[..., :3] * 255).astype("uint8")
    base_rgb = np.stack([base] * 3, axis=-1)
    return (0.55 * base_rgb + 0.45 * heat_rgb).astype("uint8")
