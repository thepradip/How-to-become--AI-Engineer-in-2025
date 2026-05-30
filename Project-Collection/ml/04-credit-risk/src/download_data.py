"""Fetch the real German Credit dataset into ./data (OpenML 'credit-g')."""

from __future__ import annotations

import pathlib

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
CSV = DATA / "credit_g.csv"


def fetch() -> pathlib.Path:
    DATA.mkdir(parents=True, exist_ok=True)
    if CSV.exists():
        print(f"✓ already present: {CSV}")
        return CSV
    try:
        from sklearn.datasets import fetch_openml

        frame = fetch_openml("credit-g", as_frame=True).frame
        frame.to_csv(CSV, index=False)
        print(f"✓ downloaded real German Credit data → {CSV} ({len(frame)} rows)")
    except Exception as exc:
        from data_loader import synthetic  # type: ignore

        synthetic().to_csv(CSV, index=False)
        print(f"⚠ OpenML unavailable ({exc}); wrote a synthetic sample. Re-run online for real data.")
    return CSV


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    fetch()
