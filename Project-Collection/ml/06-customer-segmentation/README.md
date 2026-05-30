# ML 06 · Customer Segmentation (RFM) 🟡

**Problem.** Group customers so marketing can target each group differently — champions, regulars,
at-risk. This is unsupervised learning (no labels): we discover the segments from behaviour.

**What you build.** An **RFM** (Recency / Frequency / Monetary) feature build from raw transactions,
then a comparison of **KMeans, Agglomerative and DBSCAN** scored by **silhouette**, with labelled
segment profiles — in the shared chat UI.

## Dataset (real)
UCI **Online Retail** transactions (`download_data.py`). Offline → a synthetic transaction generator
with three latent customer types.

## Approach
1. Aggregate transactions → per-customer RFM. 2. Standardise. 3. Sweep ``k`` for KMeans &
Agglomerative, run DBSCAN, pick the best by **silhouette**. 4. Profile & label each segment.

## Run it
```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py     # "segment" → "profiles" → "demo"
```

## What you learned
Unsupervised learning · the RFM framework · choosing cluster count with silhouette · turning clusters
into business-readable segments.

## Tested on
CPU (Python 3.11), fully offline on synthetic transactions. No GPU. ✅

> **Freelance relevance.** Customer segmentation for CRM/marketing is a frequent analytics gig.
