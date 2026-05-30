# NLP 10 · Semantic Résumé ↔ Job Search 🟡

**Problem.** Match résumés to jobs (or jobs to candidates) by *meaning*, not just keyword overlap —
the engine behind talent platforms and recommendation/search features.

**What you build.** A TF-IDF cosine matcher that ranks jobs for a résumé in the shared chat UI, plus
the **sentence-transformer** dense-embedding upgrade for true semantic matching.

## Run it
```bash
pip install -r requirements.txt
pytest -q
streamlit run app.py      # paste skills/résumé, or "demo"
```
Dense embeddings (semantic):
```python
# pip install sentence-transformers
from src.search import rank_semantic, SAMPLE_JOBS
rank_semantic("python pytorch llm engineer", SAMPLE_JOBS)   # all-MiniLM-L6-v2
```

## What you learned
Sparse (TF-IDF) vs. dense (embedding) retrieval · cosine similarity ranking · the basis of semantic
search & RAG retrieval. **Extension:** a vector DB (Chroma/Qdrant), re-ranking, hybrid search.

## Tested on
CPU (Python 3.11), fully offline (TF-IDF). The embedding path needs `sentence-transformers` + a model
download.

> **Why it matters.** Semantic search / matching / recommendation features are in constant demand.
