"""Retrieval-Augmented Generation over your own documents (with citations).

The dominant pattern for production LLM apps: chunk documents, retrieve the most
relevant chunks for a question, and have the LLM answer **grounded in those chunks** —
returning citations so answers are verifiable. This is a "chat with your PDF/docs" app.

Offline core uses TF-IDF retrieval + the shared LLM client (mock answers extractively).
The README documents the production stack: **LlamaIndex/LangChain + Chroma + embeddings**.
PDFs are read with ``pypdf`` when installed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SAMPLE_DOCS = [
    "Our refund policy allows returns within 30 days of purchase with a receipt. "
    "Refunds are processed to the original payment method within 5 business days.",
    "Shipping is free for orders over $50. Standard delivery takes 3-5 business days; "
    "express delivery is 1-2 business days for an extra fee.",
    "The premium plan costs $20 per month and includes priority support, unlimited "
    "projects, and a 99.9% uptime SLA. You can cancel anytime.",
]


@dataclass
class Index:
    chunks: list[str]
    vectorizer: TfidfVectorizer
    matrix: object


def chunk_text(text: str, max_words: int = 60) -> list[str]:
    """Split text into ~max_words chunks on sentence boundaries."""
    sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    chunks, cur = [], []
    for s in sents:
        cur.append(s)
        if sum(len(c.split()) for c in cur) >= max_words:
            chunks.append(" ".join(cur)); cur = []
    if cur:
        chunks.append(" ".join(cur))
    return chunks or [text]


def build_index(docs: list[str]) -> Index:
    chunks = [c for d in docs for c in chunk_text(d)]
    vec = TfidfVectorizer(stop_words="english")
    return Index(chunks=chunks, vectorizer=vec, matrix=vec.fit_transform(chunks))


def retrieve(index: Index, query: str, k: int = 3) -> list[tuple[int, str, float]]:
    sims = cosine_similarity(index.vectorizer.transform([query]), index.matrix)[0]
    order = sims.argsort()[::-1][:k]
    return [(int(i), index.chunks[i], round(float(sims[i]), 3)) for i in order if sims[i] > 0]


def answer(index: Index, query: str) -> dict:
    """Retrieve, then answer grounded in the retrieved chunks (with citations)."""
    from _shared.llm import complete

    hits = retrieve(index, query)
    context = "\n".join(f"[{i}] {text}" for i, text, _ in hits)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer using only the context and cite [chunk] numbers."
    res = complete(prompt)
    return {"answer": res["text"], "provider": res["provider"],
            "sources": [{"chunk": i, "score": s, "text": t} for i, t, s in hits]}


def load_pdf(path: str) -> str:
    """Extract text from a PDF (requires pypdf)."""
    from pypdf import PdfReader

    return "\n".join((page.extract_text() or "") for page in PdfReader(path).pages)
