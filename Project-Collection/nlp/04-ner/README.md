# NLP 04 · Named Entity Recognition 🟡

**Problem.** Pull structured facts — people, organisations, money, dates, emails, percentages — out of
unstructured text. Powers resume parsers, contract review, and clinical/financial document mining.

**What you build.** A dependency-free rule-based extractor (regex + capitalisation heuristics) wired
into the shared chat UI: paste text, get a table of entities. The README shows the production
**fine-tuned BERT / spaCy** path.

## Data
Paste any text (or type `demo`). For training a real model, use **CoNLL-2003** (general) or a domain
set (resumes, clinical notes).

## Run it
```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py      # paste text, or "demo"
```
Production path:
```python
# pip install transformers
from src.ner import extract_with_transformer
extract_with_transformer("Tim Cook leads Apple in California")
# or: python -m spacy download en_core_web_sm ; spacy.load(...)
```

## What you learned
Rule-based vs. learned NER · entity types & spans · when regex is enough vs. when you need a fine-tuned
transformer. **Extension:** fine-tune `bert-base` for token classification on your domain.

## Tested on
CPU (Python 3.11), fully offline (rule-based extractor). No GPU. The transformer path needs
`transformers` + a model download.

> **Freelance relevance.** "Extract fields X/Y/Z from these documents" is one of the most common NLP gigs.
