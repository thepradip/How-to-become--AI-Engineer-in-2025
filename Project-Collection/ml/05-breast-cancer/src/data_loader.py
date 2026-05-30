"""Real data: the UCI Wisconsin Breast Cancer Diagnostic dataset.

Shipped inside scikit-learn (``load_breast_cancer``) — it's the genuine UCI data
(569 biopsies, 30 features), so this project needs no download and runs offline.
Target: malignant vs. benign tumour.
"""

from __future__ import annotations

import pandas as pd

TARGET = "diagnosis"
POSITIVE = "malignant"  # the class we care about catching


def load() -> pd.DataFrame:
    from sklearn.datasets import load_breast_cancer

    bunch = load_breast_cancer(as_frame=True)
    df = bunch.frame.copy()
    # sklearn target: 0 = malignant, 1 = benign → human-readable labels.
    df[TARGET] = df["target"].map({0: "malignant", 1: "benign"})
    return df.drop(columns=["target"])
