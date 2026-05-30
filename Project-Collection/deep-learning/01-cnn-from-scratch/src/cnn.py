"""A convolutional neural network built from scratch in PyTorch (Fashion-MNIST).

Teaches the fundamentals: conv → ReLU → pool blocks, a classifier head, a training
loop, and evaluation. Real data is Fashion-MNIST (28×28 grayscale, 10 classes) via
torchvision; a synthetic loader keeps the smoke test fast and offline.
"""

from __future__ import annotations

import pathlib

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
LABELS = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
          "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


class SimpleCNN(nn.Module):
    """Two conv blocks + a linear head — small enough to train on CPU."""

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 28→14
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 14→7
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(32 * 7 * 7, 128),
                                  nn.ReLU(), nn.Dropout(0.25), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def synthetic_loader(n: int = 256, batch: int = 64, seed: int = 0) -> DataLoader:
    """Random 1×28×28 images for a fast, offline smoke test."""
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, 1, 28, 28, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    return DataLoader(TensorDataset(X, y), batch_size=batch, shuffle=True)


def real_loaders(batch: int = 128):
    """Fashion-MNIST train/test loaders (downloads into ./data on first run)."""
    from torchvision import datasets, transforms

    tf = transforms.ToTensor()
    DATA.mkdir(parents=True, exist_ok=True)
    train = datasets.FashionMNIST(DATA, train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(DATA, train=False, download=True, transform=tf)
    return (DataLoader(train, batch_size=batch, shuffle=True),
            DataLoader(test, batch_size=batch))
