"""Real data: the UCI/Statlog German Credit dataset (credit-risk classification).

Fetched from OpenML (``credit-g``, 1,000 applicants). Predicts whether a loan
applicant is a **good** or **bad** credit risk. Offline → synthetic fallback.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
TARGET = "class"
POSITIVE = "bad"  # the risk we want to catch


def synthetic(n: int = 1000, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(19, 75, n)
    amount = rng.integers(500, 20000, n)
    duration = rng.integers(6, 72, n)
    employment = rng.integers(0, 40, n)
    existing = rng.integers(1, 5, n)
    housing = rng.choice(["own", "rent", "free"], n, p=[0.6, 0.3, 0.1])
    purpose = rng.choice(["car", "education", "business", "furniture"], n)
    logit = (
        -0.5 + 0.00008 * amount + 0.02 * duration - 0.03 * employment
        - 0.02 * (age - 30) + (housing == "rent") * 0.5
    )
    bad = rng.uniform(size=n) < 1 / (1 + np.exp(-logit))
    return pd.DataFrame(
        {
            "age": age, "credit_amount": amount, "duration_months": duration,
            "employment_years": employment, "existing_credits": existing,
            "housing": housing, "purpose": purpose,
            TARGET: np.where(bad, "bad", "good"),
        }
    )


def load() -> pd.DataFrame:
    csv = DATA / "credit_g.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        return df if TARGET in df.columns else synthetic()
    return synthetic()
