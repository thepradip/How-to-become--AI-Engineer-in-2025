"""Real data: the UCI SMS Spam Collection (spam vs ham).

Download the real CSV from UCI/Kaggle into ./data (see README). Offline → a synthetic
generator with distinct spam/ham vocabularies so the classifiers are learnable.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
TEXT, LABEL = "message", "label"

_SPAM = ["WIN a free prize now", "Claim your FREE cash reward", "URGENT call now to win",
         "You won a £1000 voucher click here", "Free entry text WIN to claim",
         "Congratulations you have been selected", "Cheap loans apply now limited offer"]
_HAM = ["Hey are we still on for lunch", "I'll call you after the meeting", "Can you send me the notes",
        "Running late, see you soon", "Happy birthday! see you tonight", "Did you finish the report",
        "Let's catch up this weekend"]


def synthetic(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        if rng.random() < 0.4:
            base = rng.choice(_SPAM); label = "spam"
        else:
            base = rng.choice(_HAM); label = "ham"
        extra = " ".join(rng.choice(base.split(), size=rng.integers(0, 3)).tolist())
        rows.append({TEXT: f"{base} {extra}".strip(), LABEL: label})
    return pd.DataFrame(rows)


def load() -> pd.DataFrame:
    csv = DATA / "sms_spam.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        if {TEXT, LABEL}.issubset(df.columns):
            return df
    return synthetic()
