# Build Progress

Built in reviewable batches. ✅ = code + shared chat UI + tests + README, tested in a venv.

| Batch | Scope | Status |
|-------|-------|--------|
| B0 | Scaffold, shared chat UI, getting-started + env guide, website link | ✅ |
| B1 | ML 01–05 | ✅ (16 tests green) |
| B2 | ML 06–10 | ✅ (15 tests green) |
| B3 | Deep Learning 01–05 | ⬜ |
| B4 | Deep Learning 06–10 | ⬜ |
| B5 | NLP 01–05 | ⬜ |
| B6 | NLP 06–10 | ⬜ |
| B7 | GenAI 01–05 | ⬜ |
| B8 | GenAI 06–10 | ⬜ |
| B9 | Agents 01–05 | ⬜ |
| B10 | Agents 06–09 | ⬜ |
| B11 | Agents 10–13 | ⬜ |

## Shipped projects
All use multiple algorithms + a stacking ensemble via the shared engine (`_shared/tabular.py`),
the shared chat UI, and real datasets. Tested on CPU.
- `ml/01-customer-churn` — Telco churn (LogReg/RF/GB + stacking). ✅ 6 tests
- `ml/02-house-prices` — Ames regression (Ridge/RF/GB + stacking). ✅ 2 tests
- `ml/03-fraud-detection` — imbalanced + SMOTE + ensemble. ✅ 3 tests
- `ml/04-credit-risk` — German Credit classification. ✅ 2 tests
- `ml/05-breast-cancer` — UCI Wisconsin (real, offline). ✅ 3 tests
- `ml/06-customer-segmentation` — RFM + KMeans/Agglomerative/DBSCAN (silhouette). ✅ 3 tests
- `ml/07-demand-forecasting` — lag/calendar features, seasonal-naive vs ML models. ✅ 3 tests
- `ml/08-stock-movement` — technical features, chronological split + backtest. ✅ 3 tests
- `ml/09-recommender` — baselines vs SVD matrix factorization (MovieLens). ✅ 4 tests
- `ml/10-automl-optuna` — Optuna tuning + MLflow (optional) capstone. ✅ 2 tests
