"""Real data: Fake/Real news (Kaggle). Offline → synthetic headlines."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
TEXT, LABEL = "text", "label"

_REAL = ["officials confirmed the policy after a formal review", "the study published in the journal reported",
         "the central bank adjusted interest rates today", "the company released its quarterly earnings",
         "researchers documented the findings in a paper", "the ministry announced new guidelines"]
_FAKE = ["shocking secret they dont want you to know", "this miracle cure doctors hate", "you wont believe what happened next",
         "aliens secretly control the government", "click here to claim your free fortune", "one weird trick changes everything"]


def synthetic(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        real = rng.random() < 0.5
        base = rng.choice(_REAL if real else _FAKE)
        extra = " ".join(rng.choice(base.split(), size=rng.integers(0, 4)).tolist())
        rows.append({TEXT: f"{base} {extra}".strip(), LABEL: "real" if real else "fake"})
    return pd.DataFrame(rows)


def load() -> pd.DataFrame:
    csv = DATA / "news.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        if {TEXT, LABEL}.issubset(df.columns):
            return df
    return synthetic()
