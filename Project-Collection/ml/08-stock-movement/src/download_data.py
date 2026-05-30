"""Fetch real daily prices via yfinance into ./data (default: SPY)."""

from __future__ import annotations

import pathlib
import sys

DATA = pathlib.Path(__file__).resolve().parent.parent / "data"
CSV = DATA / "prices.csv"


def fetch(ticker: str = "SPY", period: str = "5y") -> pathlib.Path:
    DATA.mkdir(parents=True, exist_ok=True)
    if CSV.exists():
        print(f"✓ already present: {CSV}")
        return CSV
    try:
        import yfinance as yf

        hist = yf.download(ticker, period=period, progress=False)
        df = hist.reset_index()[["Date", "Close"]].rename(columns={"Date": "date", "Close": "close"})
        df.to_csv(CSV, index=False)
        print(f"✓ downloaded {ticker} ({len(df)} rows) → {CSV}")
    except Exception as exc:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        from stocks import synthetic_prices  # type: ignore

        synthetic_prices().to_csv(CSV, index=False)
        print(f"⚠ yfinance unavailable ({exc}); wrote a synthetic price series.")
    return CSV


if __name__ == "__main__":
    fetch(sys.argv[1] if len(sys.argv) > 1 else "SPY")
