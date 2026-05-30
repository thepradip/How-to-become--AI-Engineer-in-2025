# ML 09 · Movie Recommender (collaborative filtering) 🟡

**Problem.** Predict how a user would rate movies they haven't seen, then recommend the best ones —
the engine behind Netflix/Amazon-style suggestions.

**What you build.** A comparison of **global-mean** and **item-mean** baselines against **SVD matrix
factorization** (latent-factor collaborative filtering), judged by held-out **RMSE**, plus a top-k
recommender — in the shared chat UI.

## Dataset (real)
**MovieLens 100K** (`download_data.py`). Offline → synthetic ratings generated from latent factors
(so the collaborative structure is genuinely learnable).

## Approach
Build a user–item rating matrix → hold out 20% of ratings → predict them with each method → compare
RMSE → recommend a user's top unseen items from the SVD reconstruction.

## Run it
```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py     # "train" → "recommend"
```

## What you learned
Collaborative filtering · why strong baselines matter · matrix factorization with `TruncatedSVD` ·
evaluating recommenders. **Extensions:** implicit feedback, ranking metrics (precision@k, NDCG),
the `surprise`/`implicit` libraries.

## Tested on
CPU (Python 3.11), fully offline on synthetic ratings (SVD beats the global-mean baseline). No GPU. ✅

> **Freelance relevance.** Recommendation engines for shops/content sites are common product work.
