"""Real data: the Credit-Card Fraud dataset (highly imbalanced classification).

Fetched from OpenML (``creditcard``, 284k transactions, ~0.17% fraud). Because the
real file is large, tests use a smaller, imbalanced synthetic sample with the same
challenge: rare positives where naive accuracy is useless and recall is everything.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
TARGET = "Class"
POSITIVE = 1  # fraud


def synthetic(n: int = 6000, seed: int = 2, fraud_rate: float = 0.03) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud
    # Legit and fraud differ in a few latent components + amount distribution.
    legit = rng.normal(0, 1, size=(n_legit, 6))
    fraud = rng.normal(1.6, 1.2, size=(n_fraud, 6))
    X = np.vstack([legit, fraud])
    amount = np.concatenate([
        rng.gamma(2.0, 30, n_legit),          # everyday spend
        rng.gamma(2.0, 200, n_fraud),         # fraud skews larger
    ])
    y = np.concatenate([np.zeros(n_legit, int), np.ones(n_fraud, int)])
    df = pd.DataFrame(X, columns=[f"V{i}" for i in range(1, 7)])
    df["Amount"] = amount.round(2)
    df[TARGET] = y
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load() -> pd.DataFrame:
    csv = DATA / "creditcard.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        return df if TARGET in df.columns else synthetic()
    return synthetic()
