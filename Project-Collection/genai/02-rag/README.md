# GenAI 02 · RAG — Chat With Your Documents 🟡

**Problem.** LLMs don't know your private docs and can hallucinate. **Retrieval-Augmented Generation**
fixes both: retrieve the most relevant chunks of *your* documents and have the model answer grounded
in them — with **citations** so answers are verifiable. The #1 production LLM pattern.

**What you build.** A "chat with your docs/PDF" app: chunk → TF-IDF retrieve → answer with citations,
via the shared LLM client. Load text with `docs:` or read PDFs with `pypdf`.

## Run it
```bash
pip install -r requirements.txt
pytest -q                 # offline (TF-IDF retrieval + mock LLM)
streamlit run app.py      # ask about sample docs, or `docs: <your text>`
```
Read a PDF / use production retrieval:
```python
from src.rag import load_pdf, build_index, answer       # pip install pypdf
idx = build_index([load_pdf("contract.pdf")]); answer(idx, "what is the notice period?")
# Production: LlamaIndex/LangChain + Chroma + embeddings + OpenAI (see below)
```

## What you learned
Chunking · retrieval (TF-IDF now, embeddings later) · grounding + **citations** · the full RAG loop.
**Upgrade path:** embeddings (sentence-transformers), a vector DB (Chroma/Qdrant), re-ranking, and an
eval harness (see GenAI 10).

## Tested on
CPU (Python 3.11), fully offline (TF-IDF retrieval + mock LLM). Real answers need an API key/Ollama;
PDFs need `pypdf`.

> **Why it matters.** "Chat with our documents/PDFs" is the single most-requested LLM build.
