# NLP 02 · Sentiment Analysis 🟢

**Problem.** Decide whether text expresses a positive or negative opinion — used for product reviews,
social listening and support triage.

**What you build.** A TF-IDF classifier comparison (NB / LogReg / Linear SVM) with a chat UI that
scores any review. The README shows the **DistilBERT fine-tune** path for state-of-the-art accuracy.

## Dataset (real)
**IMDB** 50k reviews (Hugging Face `datasets`/Kaggle). Put `imdb.csv` (`review,sentiment`) in `./data`.
Offline → synthetic positive/negative reviews.

## Run it
```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py      # "train", then paste a review
```
Transformer path (higher accuracy, GPU recommended):
```python
# pip install transformers datasets
from transformers import pipeline
clf = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
clf("an absolute masterpiece")
```

## Tested on
CPU (Python 3.11), offline on synthetic data (sklearn baseline). DistilBERT path requires
`transformers` + a model download and is faster on a T4/M2 GPU.

> **Freelance relevance.** Review/feedback sentiment dashboards are a common, quick-to-ship deliverable.
