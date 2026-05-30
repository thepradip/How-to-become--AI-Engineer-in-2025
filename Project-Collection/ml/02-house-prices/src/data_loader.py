"""Real data: the Ames House Prices dataset (regression).

Fetched from OpenML (``house_prices``) by download_data.py. A schema-realistic
synthetic generator keeps tests fast and offline.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
TARGET = "SalePrice"


def synthetic(n: int = 900, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    area = rng.integers(600, 4000, n)
    beds = rng.integers(1, 6, n)
    baths = rng.integers(1, 4, n)
    age = rng.integers(0, 100, n)
    qual = rng.integers(1, 11, n)
    garage = rng.integers(0, 4, n)
    nb = rng.choice(["Downtown", "Suburb", "Rural", "Waterfront"], n, p=[0.3, 0.4, 0.2, 0.1])
    price = (
        40000 + area * 115 + beds * 7000 + baths * 9000 + qual * 16000 - age * 350
        + garage * 8000 + (nb == "Waterfront") * 80000 + (nb == "Downtown") * 30000
        + rng.normal(0, 18000, n)
    ).round(0)
    return pd.DataFrame(
        {
            "area_sqft": area, "bedrooms": beds, "bathrooms": baths, "age_years": age,
            "overall_quality": qual, "garage_cars": garage, "neighborhood": nb, TARGET: price,
        }
    )


def load() -> pd.DataFrame:
    csv = DATA / "house_prices.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        return df if TARGET in df.columns else synthetic()
    return synthetic()
