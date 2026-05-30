"""Transfer learning: fine-tune a pretrained ResNet-18 on a small custom dataset.

Transfer learning is *the* practical way to get a strong image classifier with little
data: take a network pretrained on ImageNet, freeze the backbone, and train a new head
on your classes. Real use → a folder of labelled images (torchvision ``ImageFolder``).
Smoke tests use synthetic 3×64×64 tensors and ``weights=None`` so they run offline.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def build_model(n_classes: int = 5, pretrained: bool = False) -> nn.Module:
    """ResNet-18 with a fresh classifier head for ``n_classes``.

    ``pretrained=True`` downloads ImageNet weights (do this for real training);
    ``False`` keeps it offline for tests.
    """
    from torchvision import models

    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    if pretrained:  # freeze backbone, train only the head
        for p in model.parameters():
            p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    return model


def synthetic_loader(n: int = 128, n_classes: int = 5, batch: int = 32, seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, 3, 64, 64, generator=g)
    y = torch.randint(0, n_classes, (n,), generator=g)
    return DataLoader(TensorDataset(X, y), batch_size=batch, shuffle=True)


def real_loader(folder: str, batch: int = 32):
    """torchvision ImageFolder loader for a real labelled-image directory."""
    from torchvision import datasets, transforms

    tf = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    ds = datasets.ImageFolder(folder, transform=tf)
    return DataLoader(ds, batch_size=batch, shuffle=True), ds.classes
