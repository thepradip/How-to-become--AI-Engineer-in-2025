"""Image segmentation with a compact U-Net (PyTorch).

Segmentation labels *every pixel* (e.g., tumour vs. background, road vs. not-road).
U-Net's encoder-decoder with skip connections is the classic architecture. We train on
a synthetic "find the shape" task so it runs anywhere; swap in a real mask dataset
(medical, satellite) for production.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class _DoubleConv(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Two-level U-Net for small images (e.g. 64×64)."""

    def __init__(self, in_ch: int = 3, n_classes: int = 2, c: int = 16):
        super().__init__()
        self.e1 = _DoubleConv(in_ch, c)
        self.e2 = _DoubleConv(c, c * 2)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _DoubleConv(c * 2, c * 4)
        self.up2 = nn.ConvTranspose2d(c * 4, c * 2, 2, stride=2)
        self.d2 = _DoubleConv(c * 4, c * 2)
        self.up1 = nn.ConvTranspose2d(c * 2, c, 2, stride=2)
        self.d1 = _DoubleConv(c * 2, c)
        self.out = nn.Conv2d(c, n_classes, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.d2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


def synthetic_dataset(n: int = 96, size: int = 64, seed: int = 0):
    """Images with a random bright disk on noisy background; mask = the disk."""
    rng = np.random.default_rng(seed)
    imgs, masks = [], []
    yy, xx = np.mgrid[0:size, 0:size]
    for _ in range(n):
        cx, cy, r = rng.integers(15, size - 15, 2).tolist() + [rng.integers(6, 14)]
        disk = ((xx - cx) ** 2 + (yy - cy) ** 2) < r ** 2
        img = rng.normal(0.3, 0.1, (size, size))
        img[disk] += 0.6
        imgs.append(np.clip(np.stack([img] * 3), 0, 1))
        masks.append(disk.astype("int64"))
    X = torch.tensor(np.array(imgs), dtype=torch.float32)
    Y = torch.tensor(np.array(masks), dtype=torch.long)
    return TensorDataset(X, Y)


def loader(n: int = 96, batch: int = 16, seed: int = 0) -> DataLoader:
    return DataLoader(synthetic_dataset(n, seed=seed), batch_size=batch, shuffle=True)


def train_seg(model: nn.Module, loader: DataLoader, *, epochs: int = 1, lr: float = 1e-3, device=None):
    """Pixel-wise cross-entropy training. Returns loss history."""
    device = device or torch.device("cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    hist = []
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            hist.append(float(loss.item()))
    return hist


def predict_mask(model: nn.Module, image: torch.Tensor, device=None) -> np.ndarray:
    """Return a predicted class mask (H,W) for one image (1,3,H,W)."""
    device = device or torch.device("cpu")
    model = model.to(device).eval()
    with torch.no_grad():
        return model(image.to(device)).argmax(1).squeeze().cpu().numpy()
