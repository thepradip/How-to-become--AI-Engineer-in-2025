"""Time-series forecasting with an LSTM (PyTorch).

Where ML 07 used feature-based models, here we use a recurrent network that reads a
window of recent values and predicts the next — then forecasts multiple steps by
feeding predictions back in. Synthetic seasonal series keeps it offline; swap in real
energy/sales data the same way.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class LSTMForecaster(nn.Module):
    def __init__(self, hidden: int = 32, layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden, num_layers=layers, batch_first=True)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x):                  # x: (B, T, 1)
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :])    # predict from last timestep → (B, 1)


def synthetic_series(n: int = 600, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    return (10 + 0.01 * t + 3 * np.sin(2 * np.pi * t / 50) + rng.normal(0, 0.3, n)).astype("float32")


def make_windows(series: np.ndarray, window: int = 30):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i:i + window])
        y.append(series[i + window])
    X = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)  # (N, window, 1)
    y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)  # (N, 1)
    return X, y


def train(model: nn.Module, X, y, *, epochs: int = 20, lr: float = 1e-2, device=None):
    device = device or torch.device("cpu")
    model = model.to(device)
    X, y = X.to(device), y.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    hist = []
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
        hist.append(float(loss.item()))
    return hist


def forecast(model: nn.Module, series: np.ndarray, window: int = 30, steps: int = 40, device=None) -> np.ndarray:
    """Recursive multi-step forecast from the tail of ``series``."""
    device = device or torch.device("cpu")
    model = model.to(device).eval()
    hist = list(series[-window:])
    preds = []
    with torch.no_grad():
        for _ in range(steps):
            x = torch.tensor(np.array(hist[-window:]), dtype=torch.float32).view(1, window, 1).to(device)
            nxt = float(model(x).item())
            preds.append(nxt)
            hist.append(nxt)
    return np.array(preds, dtype="float32")
