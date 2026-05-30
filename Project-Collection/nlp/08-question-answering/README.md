# NLP 08 · Extractive Question Answering 🟡

**Problem.** Given a passage and a question, return the answer found *in the text* — the core of doc
search, support bots and the "answer" step of RAG.

**What you build.** An offline TF-IDF sentence-relevance baseline that returns the best-matching
sentence, wired to the shared chat UI (set your own `context:`), plus the **SQuAD BERT** span-extraction
path.

## Run it
```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py      # ask about the built-in passage, or "context: <your text>"
```
Exact span extraction:
```python
# pip install transformers
from src.qa import answer_transformer
answer_transformer("What does RAG reduce?")   # distilbert-squad
```

## What you learned
Extractive QA vs. generative answers · sentence relevance with TF-IDF cosine · how QA plugs into RAG.
**Extension:** fine-tune on SQuAD, return character-level spans, add confidence calibration.

## Tested on
CPU (Python 3.11), fully offline (TF-IDF baseline). The transformer path needs `transformers` + a
model download.

> **Freelance relevance.** "Answer questions over our docs" — the most requested LLM feature, and QA is
> its evaluation backbone.
