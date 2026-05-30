# Build Progress

Built in reviewable batches. ✅ = code + shared chat UI + tests + README, tested in a venv.

| Batch | Scope | Status |
|-------|-------|--------|
| B0 | Scaffold, shared chat UI, getting-started + env guide, website link | ✅ |
| B1 | ML 01–05 | ✅ (16 tests green) |
| B2 | ML 06–10 | ✅ (15 tests green) |
| B3 | Deep Learning 01–05 | ✅ (15 tests green) |
| B4 | Deep Learning 06–10 | ✅ (16 tests green) |
| B5 | NLP 01–05 | ✅ (11 tests green) |
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
- `deep-learning/01-cnn-from-scratch` — PyTorch CNN, Fashion-MNIST. ✅ 3 tests
- `deep-learning/02-transfer-learning` — ResNet-18 fine-tune. ✅ 3 tests
- `deep-learning/03-pneumonia-xray` — ResNet + **Grad-CAM**. ✅ 4 tests
- `deep-learning/04-object-detection` — **YOLO11** (ultralytics). ✅ 2 tests
- `deep-learning/05-unet-segmentation` — **U-Net** pixel segmentation. ✅ 3 tests
- `deep-learning/06-traffic-signs` — GTSRB CNN (43 classes). ✅ 3 tests
- `deep-learning/07-lstm-forecasting` — LSTM multi-step forecast. ✅ 3 tests
- `deep-learning/08-speech-emotion` — audio features + MLP (librosa opt). ✅ 4 tests
- `deep-learning/09-face-verification` — embeddings + cosine verify. ✅ 3 tests
- `deep-learning/10-vit-finetune` — ViT fine-tune + W&B (optional). ✅ 3 tests
- `nlp/01-sms-spam` — TF-IDF + NB/LogReg/SVM (shared text engine). ✅ 2 tests
- `nlp/02-sentiment` — review sentiment (+DistilBERT path). ✅ 2 tests
- `nlp/03-fake-news` — real/fake text classification. ✅ 2 tests
- `nlp/04-ner` — rule-based NER (+transformer path). ✅ 2 tests
- `nlp/05-summarization` — extractive (+BART path). ✅ 3 tests
