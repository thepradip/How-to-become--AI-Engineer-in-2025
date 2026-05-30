# NLP 03 · Fake News Detection 🟡

**Problem.** Flag likely misinformation from article text/headlines. A real, socially important task —
and a cautionary one: linguistic cues help but true fact-checking needs far more than text style.

**What you build.** A TF-IDF classifier comparison (NB / LogReg / Linear SVM) with a chat UI that
scores any headline.

## Dataset (real)
**Fake and Real News** (Kaggle `clmentbisaillon/fake-and-real-news-dataset`). Put `news.csv`
(`text,label`) in `./data`. Offline → synthetic real/clickbait headlines.

## Run it
```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py
```

## What you learned
Text classification on longer documents · why surface cues over-fit · the ethics/limits of automated
fake-news detection. **Extension:** fine-tune BERT; add source/metadata features.

## Tested on
CPU (Python 3.11), offline on synthetic data. No GPU.

> ⚠️ Educational — not a production fact-checker.
