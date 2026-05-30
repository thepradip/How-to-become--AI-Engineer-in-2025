"""Movie recommender comparing baselines vs. matrix factorization.

Business problem
----------------
Recommendations drive a huge share of engagement on streaming/e-commerce. We predict
how a user would rate unseen items and recommend the top ones.

Approaches compared (held-out RMSE)
-----------------------------------
* Global-mean baseline   – predict the overall average rating
* Item-mean baseline      – predict each movie's average rating
* SVD matrix factorization – learn latent user/item factors (collaborative filtering)

Real data: **MovieLens 100K** (download_data.py). Offline → synthetic ratings generated
from latent factors so the structure is genuinely learnable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error


@dataclass
class RecoResult:
    leaderboard: pd.DataFrame   # method -> RMSE
    best_method: str
    user_item: pd.DataFrame     # fitted matrix (item means filled)
    svd_pred: np.ndarray        # reconstructed rating matrix


def synthetic_ratings(n_users: int = 300, n_items: int = 120, k: int = 5, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n_users, k))
    V = rng.normal(size=(n_items, k))
    rows = []
    for u in range(n_users):
        seen = rng.choice(n_items, size=rng.integers(15, 40), replace=False)
        for i in seen:
            r = 3 + U[u] @ V[i] / k  # latent signal centred near 3
            rating = float(np.clip(round(r + rng.normal(0, 0.4), 0), 1, 5))
            rows.append({"user_id": u, "item_id": int(i), "rating": rating})
    return pd.DataFrame(rows)


def _matrix(ratings: pd.DataFrame) -> pd.DataFrame:
    return ratings.pivot_table(index="user_id", columns="item_id", values="rating")


def evaluate(ratings: pd.DataFrame, test_frac: float = 0.2, n_factors: int = 10, seed: int = 0) -> RecoResult:
    """Hold out a fraction of ratings; compare methods by RMSE on them."""
    rng = np.random.default_rng(seed)
    mask = rng.uniform(size=len(ratings)) < test_frac
    train, test = ratings[~mask], ratings[mask]

    global_mean = train["rating"].mean()
    item_mean = train.groupby("item_id")["rating"].mean()

    # SVD on the (item-mean-filled) user-item matrix.
    ui = _matrix(train)
    filled = ui.apply(lambda col: col.fillna(item_mean.get(col.name, global_mean)), axis=0)
    filled = filled.fillna(global_mean)
    k = min(n_factors, min(filled.shape) - 1)
    svd = TruncatedSVD(n_components=max(1, k), random_state=seed)
    recon = svd.inverse_transform(svd.fit_transform(filled.values))
    recon_df = pd.DataFrame(recon, index=filled.index, columns=filled.columns)

    def pred_svd(u, i):
        if u in recon_df.index and i in recon_df.columns:
            return float(recon_df.loc[u, i])
        return item_mean.get(i, global_mean)

    y = test["rating"].values
    rows = [
        {"method": "Global mean", "RMSE": _rmse(y, np.full(len(test), global_mean))},
        {"method": "Item mean", "RMSE": _rmse(y, test["item_id"].map(item_mean).fillna(global_mean).values)},
        {"method": "SVD (matrix factorization)",
         "RMSE": _rmse(y, np.array([pred_svd(u, i) for u, i in zip(test["user_id"], test["item_id"])]))},
    ]
    lb = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    return RecoResult(leaderboard=lb, best_method=lb.iloc[0]["method"],
                      user_item=ui, svd_pred=recon_df.values)


def recommend(res: RecoResult, user_idx: int = 0, k: int = 5) -> pd.DataFrame:
    """Top-k unseen items for a user, by reconstructed score."""
    ui = res.user_item
    if user_idx >= len(ui):
        user_idx = 0
    seen = ui.iloc[user_idx].dropna().index
    scores = pd.Series(res.svd_pred[user_idx], index=ui.columns)
    scores = scores.drop(index=seen, errors="ignore").sort_values(ascending=False)
    return scores.head(k).rename("predicted_rating").reset_index()


def _rmse(a, b) -> float:
    return round(float(np.sqrt(mean_squared_error(a, b))), 3)
