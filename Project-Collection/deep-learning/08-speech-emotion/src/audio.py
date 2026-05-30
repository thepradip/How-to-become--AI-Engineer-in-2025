"""Speech emotion recognition from audio features (PyTorch MLP).

Pipeline: waveform → fixed-length feature vector → small MLP classifier. Real data is
the **RAVDESS** emotional-speech set; features are MFCCs via **librosa** when available.
To stay offline and dependency-light for tests, a numpy spectral-band extractor and
synthetic class-dependent waveforms are provided as a fallback.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SR = 16000
FEAT_DIM = 20
EMOTIONS = ["neutral", "happy", "sad", "angry", "fearful", "disgust"]


def synthetic_waveform(emotion: int, n_classes: int = len(EMOTIONS), dur: float = 1.0, seed: int = 0) -> np.ndarray:
    """A 1-second waveform whose spectral content depends on the emotion class."""
    rng = np.random.default_rng(seed + emotion)
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    base = 110 + emotion * 70                                   # class-dependent pitch
    wave = np.sin(2 * np.pi * base * t) + 0.4 * np.sin(2 * np.pi * 2 * base * t)
    wave *= (1.0 + 0.3 * np.sin(2 * np.pi * (1 + emotion) * t))  # class-dependent envelope
    return (wave + rng.normal(0, 0.05, t.shape)).astype("float32")


def extract_features(wave: np.ndarray, sr: int = SR, n: int = FEAT_DIM) -> np.ndarray:
    """MFCC mean via librosa if installed; else log spectral-band energies (numpy)."""
    try:
        import librosa

        mfcc = librosa.feature.mfcc(y=wave.astype("float32"), sr=sr, n_mfcc=n)
        return mfcc.mean(axis=1).astype("float32")
    except Exception:
        spec = np.abs(np.fft.rfft(wave))
        bands = np.array_split(spec, n)
        return np.log1p(np.array([b.mean() for b in bands], dtype="float32"))


def synthetic_dataset(per_class: int = 40, n_classes: int = len(EMOTIONS), seed: int = 0):
    X, y = [], []
    for c in range(n_classes):
        for k in range(per_class):
            X.append(extract_features(synthetic_waveform(c, n_classes, seed=seed + k)))
            y.append(c)
    return (torch.tensor(np.array(X), dtype=torch.float32),
            torch.tensor(np.array(y), dtype=torch.long))


def loader(per_class: int = 40, batch: int = 32, seed: int = 0) -> DataLoader:
    X, y = synthetic_dataset(per_class, seed=seed)
    return DataLoader(TensorDataset(X, y), batch_size=batch, shuffle=True)


class EmotionMLP(nn.Module):
    def __init__(self, in_dim: int = FEAT_DIM, n_classes: int = len(EMOTIONS)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, n_classes))

    def forward(self, x):
        return self.net(x)
