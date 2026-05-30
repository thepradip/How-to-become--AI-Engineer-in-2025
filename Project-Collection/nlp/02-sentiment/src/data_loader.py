"""Real data: IMDB movie reviews (positive/negative). Offline → synthetic reviews."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
TEXT, LABEL = "review", "sentiment"

_POS = ["loved it, a great movie", "excellent acting and wonderful story", "best film this year highly recommend",
        "beautiful, moving and brilliant", "fantastic direction, truly enjoyable", "a masterpiece, superb performances"]
_NEG = ["terrible, a waste of time", "boring and dull, awful acting", "worst movie ever, do not recommend",
        "painfully slow and predictable", "disappointing and forgettable", "bad script, poor pacing"]


def synthetic(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        pos = rng.random() < 0.5
        base = rng.choice(_POS if pos else _NEG)
        extra = " ".join(rng.choice(base.split(), size=rng.integers(0, 3)).tolist())
        rows.append({TEXT: f"{base} {extra}".strip(), LABEL: "positive" if pos else "negative"})
    return pd.DataFrame(rows)


def load() -> pd.DataFrame:
    csv = DATA / "imdb.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        if {TEXT, LABEL}.issubset(df.columns):
            return df
    return synthetic()
