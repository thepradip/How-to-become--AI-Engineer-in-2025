"""Real-dataset helpers — no website scraping, no copied code.

Every project loads *real* public data through legitimate dataset registries and
APIs (OpenML, scikit-learn's bundled sets, the UCI repo, yfinance, Hugging Face).
We never scrape arbitrary web pages, and we never commit large data to git — these
helpers download into the project's local ``data/`` folder on first run and cache.

Functions degrade gracefully: if a download fails (e.g. offline CI), callers can
fall back to a small bundled or synthetic sample so tests still pass — but the
README always documents the real source so learners use real data when online.
"""

from __future__ import annotations

import pathlib
from typing import Optional

import pandas as pd


def cache_dir(project_file: str, name: str = "data") -> pathlib.Path:
    """Return (and create) the ``data/`` dir next to a project file."""
    d = pathlib.Path(project_file).resolve().parent / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_openml(name: str, version: Optional[int] = None, cache_to: Optional[pathlib.Path] = None) -> pd.DataFrame:
    """Load a real dataset from OpenML (https://www.openml.org) as a DataFrame.

    OpenML is an open dataset registry — this is an API call, not scraping.
    Result is cached to ``cache_to/<name>.parquet`` when provided.
    """
    if cache_to is not None:
        cached = cache_to / f"{name}.parquet"
        if cached.exists():
            return pd.read_parquet(cached)

    from sklearn.datasets import fetch_openml

    kwargs = {"as_frame": True}
    if version is not None:
        kwargs["version"] = version
    bunch = fetch_openml(name, **kwargs)
    df = bunch.frame
    if cache_to is not None:
        try:
            df.to_parquet(cache_to / f"{name}.parquet")
        except Exception:
            pass  # parquet engine optional; caching is best-effort
    return df


def download_url(url: str, dest: pathlib.Path) -> pathlib.Path:
    """Download a file from a direct dataset URL to ``dest`` (cached)."""
    if dest.exists():
        return dest
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)  # noqa: S310 - documented dataset URLs only
    return dest
