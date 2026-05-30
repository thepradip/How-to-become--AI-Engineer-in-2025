"""Traffic-sign recognition CNN (GTSRB, 43 classes).

A small color-image CNN for the German Traffic Sign Recognition Benchmark — the kind
of classifier inside driver-assist systems. Real data via torchvision GTSRB; synthetic
3×32×32 tensors keep the smoke test offline.
"""

from __future__ import annotations

import pathlib

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
N_CLASSES = 43


class TrafficCNN(nn.Module):
    def __init__(self, n_classes: int = N_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32→16
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16→8
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 8 * 8, 128),
                                  nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def synthetic_loader(n: int = 256, n_classes: int = N_CLASSES, batch: int = 64, seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, 3, 32, 32, generator=g)
    y = torch.randint(0, n_classes, (n,), generator=g)
    return DataLoader(TensorDataset(X, y), batch_size=batch, shuffle=True)


def real_loaders(batch: int = 128):
    from torchvision import datasets, transforms

    tf = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    DATA.mkdir(parents=True, exist_ok=True)
    train = datasets.GTSRB(DATA, split="train", download=True, transform=tf)
    test = datasets.GTSRB(DATA, split="test", download=True, transform=tf)
    return DataLoader(train, batch_size=batch, shuffle=True), DataLoader(test, batch_size=batch)
