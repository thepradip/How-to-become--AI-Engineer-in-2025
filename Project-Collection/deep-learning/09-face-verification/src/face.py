"""Face verification via learned embeddings + cosine similarity (PyTorch).

Verification answers "are these two faces the same person?" — the core of face login
and de-duplication. We learn an embedding network (trained as an identity classifier),
then compare two images by the cosine similarity of their L2-normalised embeddings.
Synthetic "identities" keep it offline; for real faces use facenet-pytorch + LFW.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class FaceNet(nn.Module):
    def __init__(self, emb: int = 64, n_ids: int = 10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 32→16
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 16→8
            nn.Flatten(), nn.Linear(32 * 8 * 8, emb))
        self.head = nn.Linear(emb, n_ids)

    def embed(self, x):
        return F.normalize(self.features(x), dim=1)   # unit-length embedding

    def forward(self, x):
        return self.head(self.features(x))            # identity logits (for training)


def synthetic_identities(n_ids: int = 10, per_id: int = 30, seed: int = 0):
    """Each identity is a distinct base image; samples add small noise."""
    rng = np.random.default_rng(seed)
    bases = rng.normal(0.5, 0.25, (n_ids, 3, 32, 32)).astype("float32")
    X, y = [], []
    for i in range(n_ids):
        for _ in range(per_id):
            X.append(np.clip(bases[i] + rng.normal(0, 0.05, (3, 32, 32)), 0, 1).astype("float32"))
            y.append(i)
    return (torch.tensor(np.array(X)), torch.tensor(np.array(y), dtype=torch.long))


def loader(n_ids: int = 10, per_id: int = 30, batch: int = 64, seed: int = 0) -> DataLoader:
    X, y = synthetic_identities(n_ids, per_id, seed)
    return DataLoader(TensorDataset(X, y), batch_size=batch, shuffle=True)


def verify(model: nn.Module, img1: torch.Tensor, img2: torch.Tensor, threshold: float = 0.7):
    """Return (is_same, cosine_similarity) for two images (each (1,3,32,32))."""
    model.eval()
    dev = next(model.parameters()).device
    with torch.no_grad():
        e1, e2 = model.embed(img1.to(dev)), model.embed(img2.to(dev))
        sim = float((e1 * e2).sum().item())
    return sim >= threshold, sim
