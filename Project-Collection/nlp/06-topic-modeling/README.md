# NLP 06 · Topic Modeling 🟡

**Problem.** Automatically discover the themes in a large pile of documents (support tickets, reviews,
research papers) without labels — for exploration, tagging and routing.

**What you build.** A comparison of **NMF** (on TF-IDF) and **LDA** (on counts), showing the top words
per topic and matching new documents to a topic — in the shared chat UI.

## Dataset
Synthetic multi-topic documents (sports/tech/food/finance) so it runs offline. Point it at a real
corpus (20 Newsgroups, your tickets) by replacing `synthetic_docs`.

## Run it
```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py      # "discover", or paste a document
```

## What you learned
Unsupervised topic discovery · NMF vs. LDA · interpreting topics by top words. **Modern upgrade:**
**BERTopic** (sentence-transformer embeddings + UMAP + HDBSCAN) for far better topics on real text.

## Tested on
CPU (Python 3.11), fully offline (sklearn). No GPU.

> **Why it matters.** "What are people talking about in this data?" — a common analytics ask.
