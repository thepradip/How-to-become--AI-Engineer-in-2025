# NLP 01 · SMS Spam Classifier 🟢

**Problem.** Filter spam from legitimate text messages — the gentle introduction to NLP: turn raw text
into numbers (TF-IDF) and classify.

**What you build.** A TF-IDF pipeline comparing **Naive Bayes, Logistic Regression and Linear SVM**;
then a chat UI where you paste any message and get spam/ham + confidence.

## Dataset (real)
**UCI SMS Spam Collection** (5,574 messages). Put `sms_spam.csv` (columns `message,label`) in `./data`.
Offline → a synthetic generator with distinct spam/ham vocabularies.

## Run it
```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py      # "train", then paste a message
```

## What you learned
TF-IDF features · n-grams & stop-words · comparing linear/NB/SVM text classifiers · precision vs.
recall for a filter. **Extension:** fine-tune **DistilBERT** for higher accuracy (see NLP 02).

## Tested on
CPU (Python 3.11), fully offline on synthetic data. No GPU.

> **Freelance relevance.** Text moderation / spam / intent classification is everyday NLP contract work.
