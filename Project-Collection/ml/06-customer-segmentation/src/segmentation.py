"""Customer segmentation via RFM features + multiple clustering algorithms.

Business problem
----------------
Marketing can't treat every customer the same. By scoring each customer on
**Recency** (days since last purchase), **Frequency** (number of orders) and
**Monetary** value (total spend) — the classic "RFM" model — and clustering them,
the team can target "champions", "at-risk", "new", etc. with the right message.

Algorithms compared (per the "use the algorithms" goal)
------------------------------------------------------
KMeans, Agglomerative (hierarchical) and DBSCAN, scored by **silhouette**. KMeans
also picks ``k`` by sweeping a range and taking the best silhouette.

Real data: the UCI **Online Retail** transactions set (download_data.py). A synthetic
transaction generator keeps tests offline and fast.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

RFM_COLS = ["recency", "frequency", "monetary"]


@dataclass
class SegmentResult:
    rfm: pd.DataFrame          # per-customer RFM + assigned segment
    leaderboard: pd.DataFrame  # algorithm -> silhouette, n_clusters
    best_algo: str
    best_k: int
    profiles: pd.DataFrame     # mean RFM per segment + size + label


def synthetic_transactions(n_customers: int = 400, seed: int = 0) -> pd.DataFrame:
    """Generate synthetic order history with a few latent customer types."""
    rng = np.random.default_rng(seed)
    rows = []
    for cid in range(n_customers):
        kind = rng.choice(["champion", "regular", "lapsed"], p=[0.25, 0.5, 0.25])
        if kind == "champion":
            orders, recency_base, spend = rng.integers(8, 25), rng.integers(1, 30), rng.uniform(80, 300)
        elif kind == "regular":
            orders, recency_base, spend = rng.integers(2, 8), rng.integers(20, 120), rng.uniform(30, 120)
        else:
            orders, recency_base, spend = rng.integers(1, 3), rng.integers(150, 365), rng.uniform(10, 60)
        for _ in range(orders):
            rows.append({"customer_id": cid,
                         "days_ago": int(max(0, recency_base + rng.integers(-15, 15))),
                         "amount": round(float(spend * rng.uniform(0.5, 1.5)), 2)})
    return pd.DataFrame(rows)


def build_rfm(tx: pd.DataFrame) -> pd.DataFrame:
    """Aggregate transactions into per-customer Recency/Frequency/Monetary."""
    g = tx.groupby("customer_id")
    rfm = pd.DataFrame({
        "recency": g["days_ago"].min(),
        "frequency": g.size(),
        "monetary": g["amount"].sum().round(2),
    })
    return rfm


def _fit_score(X, algo: str, k: int):
    if algo == "KMeans":
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    elif algo == "Agglomerative":
        labels = AgglomerativeClustering(n_clusters=k).fit_predict(X)
    else:  # DBSCAN finds its own cluster count
        labels = DBSCAN(eps=0.8, min_samples=5).fit_predict(X)
    n = len(set(labels) - {-1})
    score = silhouette_score(X, labels) if n > 1 and len(set(labels)) < len(X) else -1.0
    return labels, n, round(float(score), 3)


def segment(rfm: pd.DataFrame, k_range=range(2, 7)) -> SegmentResult:
    """Compare clustering algorithms and label the winning segmentation."""
    X = StandardScaler().fit_transform(rfm[RFM_COLS])
    rows, best = [], None
    for algo in ("KMeans", "Agglomerative"):
        for k in k_range:
            labels, n, score = _fit_score(X, algo, k)
            rows.append({"algorithm": algo, "k": k, "silhouette": score})
            if best is None or score > best[3]:
                best = (algo, k, labels, score)
    # DBSCAN once (auto-k)
    dbl, dbn, dbscore = _fit_score(X, "DBSCAN", 0)
    rows.append({"algorithm": "DBSCAN", "k": dbn, "silhouette": dbscore})
    if dbscore > best[3]:
        best = ("DBSCAN", dbn, dbl, dbscore)

    algo, k, labels, _ = best
    rfm = rfm.copy()
    rfm["segment"] = labels
    profiles = _profiles(rfm)
    lb = (pd.DataFrame(rows).sort_values("silhouette", ascending=False).reset_index(drop=True))
    return SegmentResult(rfm=rfm, leaderboard=lb, best_algo=algo, best_k=int(k), profiles=profiles)


def _profiles(rfm: pd.DataFrame) -> pd.DataFrame:
    prof = rfm.groupby("segment")[RFM_COLS].mean().round(1)
    prof["size"] = rfm.groupby("segment").size()
    # Heuristic labels from RFM.
    labels = []
    for _, r in prof.iterrows():
        if r["frequency"] >= prof["frequency"].median() and r["recency"] <= prof["recency"].median():
            labels.append("Champion")
        elif r["recency"] > prof["recency"].median():
            labels.append("At-risk / Lapsed")
        else:
            labels.append("Regular")
    prof["label"] = labels
    return prof.reset_index()
