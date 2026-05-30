"""Small PyTorch helpers shared across the deep-learning projects.

Keeps device selection and a minimal train/eval loop in one place so each DL
project focuses on its *model*, not boilerplate. ``torch`` is imported lazily
inside functions so this module can be imported even where torch isn't installed
(tests use ``pytest.importorskip('torch')``).
"""

from __future__ import annotations

from typing import Iterable


def get_device():
    """Return the best available device: CUDA → Apple MPS → CPU."""
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_classifier(model, loader: Iterable, *, epochs: int = 1, lr: float = 1e-3, device=None):
    """Train a classification model with Adam + cross-entropy. Returns loss history."""
    import torch
    from torch import nn

    device = device or get_device()
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    history = []
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            history.append(float(loss.item()))
    return history


def accuracy(model, loader: Iterable, device=None) -> float:
    """Top-1 accuracy over a data loader."""
    import torch

    device = device or get_device()
    model = model.to(device).eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            correct += int((pred == yb).sum())
            total += int(yb.numel())
    return correct / max(total, 1)
