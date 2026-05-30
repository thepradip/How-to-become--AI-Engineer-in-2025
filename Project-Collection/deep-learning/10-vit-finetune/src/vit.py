"""Fine-tune a Vision Transformer (ViT) + hyperparameter sweeps with Weights & Biases.

Modern image models are Transformers. Here we fine-tune torchvision's ``vit_b_16`` on a
custom set of classes and (optionally) track/sweep hyperparameters with **W&B**. ViT
expects 224×224 inputs. ``weights=None`` keeps the smoke test offline; use
``pretrained=True`` for real fine-tuning (downloads ImageNet weights — do this on a GPU).
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def build_vit(n_classes: int = 5, pretrained: bool = False) -> nn.Module:
    from torchvision import models

    weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
    model = models.vit_b_16(weights=weights)
    if pretrained:                              # freeze backbone, train the head
        for p in model.parameters():
            p.requires_grad = False
    in_features = model.heads.head.in_features
    model.heads.head = nn.Linear(in_features, n_classes)
    return model


def synthetic_loader(n: int = 8, n_classes: int = 5, batch: int = 4, seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    X = torch.rand(n, 3, 224, 224, generator=g)     # ViT input size
    y = torch.randint(0, n_classes, (n,), generator=g)
    return DataLoader(TensorDataset(X, y), batch_size=batch, shuffle=True)


def maybe_wandb(config: dict | None = None):
    """Return a W&B run if wandb is installed (offline mode), else None."""
    try:
        import wandb

        return wandb.init(project="vit-finetune", config=config or {}, mode="offline", reinit=True)
    except Exception:
        return None
