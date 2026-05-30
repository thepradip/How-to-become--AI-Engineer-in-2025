# NLP 05 · Text Summarization 🟡

**Problem.** Condense long text into a short summary — for news digests, meeting notes, document
triage. Two families: **extractive** (pick key sentences) and **abstractive** (write new ones).

**What you build.** A fast, faithful **extractive** summarizer (sentence scoring by normalised word
frequency) in the shared chat UI, plus the documented **BART abstractive** path.

## Run it
```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py      # paste a passage, or "demo" (sentence count in sidebar)
```
Abstractive (fluent, needs a model):
```python
# pip install transformers
from src.summarize import abstractive_summary
abstractive_summary(long_text)   # facebook/bart-large-cnn
```

## What you learned
Extractive vs. abstractive summarization · sentence scoring · faithfulness vs. fluency trade-offs.
**Extension:** TextRank, fine-tune BART/Pegasus, ROUGE evaluation.

## Tested on
CPU (Python 3.11), fully offline (extractive). No GPU. BART path needs `transformers` + a download.

> **Why it matters.** Summarization/condensation features are common in document and content tools.
