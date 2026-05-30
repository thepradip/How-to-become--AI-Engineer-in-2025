"""Fetch the real IBM Telco Customer Churn dataset into ./data.

Source (real, open): OpenML — https://www.openml.org/search?type=data&q=Telco-Customer-Churn
(The same data is mirrored on Kaggle: `blastchar/telco-customer-churn`.)

Run:  python src/download_data.py
We never commit the CSV — it lands in ./data which is git-ignored.
"""

from __future__ import annotations

import pathlib
import sys

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
CSV = DATA / "telco_churn.csv"
OPENML_NAMES = ["Telco-Customer-Churn", "telco-customer-churn"]


def fetch() -> pathlib.Path:
    """Download the real dataset to ./data/telco_churn.csv (cached)."""
    DATA.mkdir(parents=True, exist_ok=True)
    if CSV.exists():
        print(f"✓ already present: {CSV}")
        return CSV

    from sklearn.datasets import fetch_openml

    last_err = None
    for name in OPENML_NAMES:
        try:
            frame = fetch_openml(name, as_frame=True).frame
            frame.to_csv(CSV, index=False)
            print(f"✓ downloaded real data via OpenML '{name}' → {CSV}  ({len(frame)} rows)")
            return CSV
        except Exception as exc:  # try next name / fall through
            last_err = exc

    # Offline / unavailable: write a schema-compatible sample so the app still runs,
    # but make it loud so the learner knows to get the real data when online.
    from model import synthetic_churn  # type: ignore

    synthetic_churn(800).to_csv(CSV, index=False)
    print(
        "⚠ could not reach OpenML (",
        last_err,
        ")\n  Wrote a SYNTHETIC sample so you can run the app. "
        "Re-run online for the real dataset.",
    )
    return CSV


if __name__ == "__main__":
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    fetch()
