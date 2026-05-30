"""Fetch the real Credit-Card Fraud dataset into ./data (OpenML 'creditcard').

Note: the real file is ~150 MB and very imbalanced. Offline → synthetic fallback.
"""

from __future__ import annotations

import pathlib

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
CSV = DATA / "creditcard.csv"


def fetch() -> pathlib.Path:
    DATA.mkdir(parents=True, exist_ok=True)
    if CSV.exists():
        print(f"✓ already present: {CSV}")
        return CSV
    try:
        from sklearn.datasets import fetch_openml

        frame = fetch_openml("creditcard", as_frame=True).frame
        frame.to_csv(CSV, index=False)
        print(f"✓ downloaded real fraud data → {CSV} ({len(frame)} rows)")
    except Exception as exc:
        from data_loader import synthetic  # type: ignore

        synthetic().to_csv(CSV, index=False)
        print(f"⚠ OpenML unavailable ({exc}); wrote a synthetic imbalanced sample.")
    return CSV


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    fetch()
